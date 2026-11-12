import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import HTTPException

from comfyui_anima_api.config import CIVITAI_CONFIG
from comfyui_anima_api.loras import aria2c, civitai
from comfyui_anima_api.loras.verification import verify
from comfyui_anima_api.schemas import DownloadParams

logger = logging.getLogger(__name__)


class _FileLock:
    """per-file 下载锁 + 引用计数：无人持有/等待即从表里回收，避免锁表无限增长。"""

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.users = 0


_LOCK_GUARD = asyncio.Lock()
_FILE_LOCKS: dict[str, _FileLock] = {}


@asynccontextmanager
async def _file_lock(name: str) -> AsyncIterator[None]:
    async with _LOCK_GUARD:
        entry = _FILE_LOCKS.setdefault(name, _FileLock())
        entry.users += 1
    async with entry.lock:
        try:
            yield
        finally:
            async with _LOCK_GUARD:
                entry.users -= 1
                if entry.users == 0:
                    _FILE_LOCKS.pop(name, None)


async def run(
    p: DownloadParams, on_progress: aria2c.ProgressCallback | None = None, cancel: asyncio.Event | None = None
) -> dict:
    """从 Civitai 下载 LoRA 并落盘，返回响应字典。"""

    started = time.monotonic()

    # 检测 aria2c 可用性
    if not aria2c.is_available():
        raise HTTPException(status_code=503, detail=aria2c.MISSING_MESSAGE)

    logger.info("lora/download: version_id=%s filename=%s", p.version_id, p.filename)
    info = await civitai.fetch_version(p.version_id)
    header_args = aria2c.build_headers(civitai.build_request_headers())

    download_dir = CIVITAI_CONFIG.loras_download_dir
    os.makedirs(download_dir, exist_ok=True)

    async with _file_lock(p.filename):
        tmp_path = os.path.join(download_dir, p.filename + ".part")

        if on_progress is not None and info.size_bytes:
            # 续传场景先把已有残片量报出去，调用方立刻有进度可显示
            done = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
            await on_progress(
                {
                    "bytes": done,
                    "total": info.size_bytes,
                    "percent": round(done * 100 / info.size_bytes, 1),
                    "speed_bps": 0,
                    "eta_s": None,
                }
            )

        try:
            await aria2c.run(
                civitai.build_download_url(p.version_id),
                header_args,
                download_dir,
                p.filename + ".part",
                timeout=CIVITAI_CONFIG.download_timeout,
                on_progress=on_progress,
                cancel=cancel,
                proxy=CIVITAI_CONFIG.proxy,
            )
        except (TimeoutError, RuntimeError) as e:
            detail = f"civitai download failed ({e}); partial file kept, retry to resume"
            if not CIVITAI_CONFIG.api_key:
                detail += "; no CIVITAI_API_KEY configured, gated models cannot be downloaded"
            raise HTTPException(status_code=502, detail=detail) from e

        # 全量 SHA256 对大文件是秒级同步 I/O：下放线程池，别卡住事件循环
        await asyncio.to_thread(verify, tmp_path, info.sha256, info.size_bytes)
        size = os.path.getsize(tmp_path)  # 必须在 replace 前取：之后 tmp_path 已不存在
        os.replace(tmp_path, os.path.join(download_dir, p.filename))
        logger.info("lora/download: saved %s (%d bytes, %.1fs)", p.filename, size, time.monotonic() - started)
        return {"filename": p.filename, "version_id": p.version_id}
