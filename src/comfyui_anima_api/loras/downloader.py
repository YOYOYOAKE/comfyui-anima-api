import asyncio
import contextlib
import logging
import os
import re
import shutil
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import HTTPException

from comfyui_anima_api.config import ARIA2_CONFIG, CIVITAI_CONFIG
from comfyui_anima_api.loras.registry import update_registry
from comfyui_anima_api.loras.verification import verify_lora_file
from comfyui_anima_api.schemas import DownloadParams

logger = logging.getLogger(__name__)

ARIA2C_MISSING = "aria2c is not installed on the server; /lora/download is disabled (install aria2 and restart)"

# 下载进度回调：{"bytes", "total", "percent", "speed_bps", "eta_s"}（eta_s/percent 可能为 None）
_ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]]

# aria2c 汇总进度行（--summary-interval 输出）：
# [#8b1f3c 45MiB/1.9GiB(2%) CN:8 DL:5.2MiB ETA:5m54s]
_ARIA2_PROGRESS_RE = re.compile(
    r"\[#\w+\s+([\d.]+[KMGTP]?i?B)/([\d.]+[KMGTP]?i?B)\((\d+)%\)\s+CN:\d+\s+DL:([\d.]+[KMGTP]?i?B)(?:\s+ETA:(\S+))?\]"
)
_SIZE_MULT = {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3, "TiB": 1024**4, "PiB": 1024**5}


def _parse_aria2_size(token: str) -> int | None:
    m = re.fullmatch(r"([\d.]+)([KMGTP]?i?B)", token)
    if m is None:
        return None
    return int(float(m.group(1)) * _SIZE_MULT[m.group(2)])


def _parse_aria2_eta(token: str | None) -> float | None:
    m = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", token or "")
    if m is None or not any(m.groups()):
        return None  # n/a / unknown / 空串
    h, mi, s = (int(g) if g else 0 for g in m.groups())
    return float(h * 3600 + mi * 60 + s)


def _parse_aria2_progress(line: str) -> dict[str, Any] | None:
    """解析一行 aria2 输出为进度字典；不是进度行（横幅/结果行等）返回 None。"""
    m = _ARIA2_PROGRESS_RE.search(line)
    if m is None:
        return None
    done, total, speed = _parse_aria2_size(m.group(1)), _parse_aria2_size(m.group(2)), _parse_aria2_size(m.group(4))
    if done is None or total is None or speed is None or total <= 0:
        return None
    return {
        "bytes": done,
        "total": total,
        "percent": round(done * 100 / total, 1),
        "speed_bps": speed,
        "eta_s": _parse_aria2_eta(m.group(5)),
    }


def aria2c_available() -> bool:
    """PATH 里能否找到 aria2c。缺失时启动告警且 /lora/download 软禁用（503），其余端点不受影响。"""
    return shutil.which("aria2c") is not None


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


def _metadata_headers() -> dict[str, str]:
    """元数据请求与 aria2 下载同权：API Key 一并带上，受限资源在 precheck 即可给出鉴权报错。"""
    headers = {"User-Agent": CIVITAI_CONFIG.ua, **CIVITAI_CONFIG.extra_headers}
    if CIVITAI_CONFIG.api_key:
        headers["Authorization"] = f"Bearer {CIVITAI_CONFIG.api_key}"
    return headers


async def _precheck_version(client: httpx.AsyncClient, version_id: int) -> tuple[str | None, int | None]:
    """下载前校验 version 是否存在、是否为 Anima 底模，返回 (SHA256, 文件总字节数)（都可能为 None）。"""
    r = await client.get(f"{CIVITAI_CONFIG.base_url}/api/v1/model-versions/{version_id}")
    if r.status_code == 404:
        raise HTTPException(status_code=404, detail=f"unknown civitai version_id {version_id}")
    if r.status_code in (401, 403):
        if CIVITAI_CONFIG.api_key:
            hint = "CIVITAI_API_KEY may be invalid"
        else:
            hint = "this resource requires authentication; set CIVITAI_API_KEY on the server and retry"
        raise HTTPException(status_code=502, detail=f"civitai rejected the request (HTTP {r.status_code}): {hint}")
    if r.status_code != 200:
        logger.error("lora/download: metadata HTTP %d", r.status_code)
        raise HTTPException(status_code=502, detail=f"civitai metadata request failed with HTTP {r.status_code}")
    try:
        meta = r.json()
    except ValueError:
        raise HTTPException(status_code=502, detail="civitai returned non-JSON metadata") from None
    base_model = str(meta.get("baseModel") or "")
    logger.info(
        "lora/download: model=%r version=%r baseModel=%r",
        (meta.get("model") or {}).get("name"),
        meta.get("name"),
        base_model,
    )
    if "anima" not in base_model.lower():
        raise HTTPException(
            status_code=400, detail=f"civitai baseModel is '{base_model or 'unknown'}', only Anima LoRAs are supported"
        )
    files = [f for f in meta.get("files", []) if f.get("type") == "Model"] or meta.get("files") or [{}]
    # 多文件 version（fp16/pruned 变体）时对准 primary：/api/download 端点实际服务的
    # 就是它——拿错文件的 SHA 会把下载成功误判成哈希不符（假阳性 502）
    pick = next((f for f in files if f.get("primary")), files[0])
    size_kb = pick.get("sizeKB")
    total_bytes = int(size_kb * 1024) if isinstance(size_kb, (int, float)) else None
    return (pick.get("hashes") or {}).get("SHA256"), total_bytes


async def download(
    p: DownloadParams, on_progress: _ProgressCallback | None = None, cancel: asyncio.Event | None = None
) -> dict:
    """
    从 Civitai 下载 LoRA 并落盘登记，返回响应字典。
    on_progress 收到解析后的 aria2 进度（每秒一条）；cancel 置位即中止下载
    （残片保留可续传），以 CancelledError 通知调用方。
    """

    started = time.monotonic()

    # 文件名合法性由 DownloadParams schema 保证（与 loras.name 同一正则）；此处只查服务可用性
    if not aria2c_available():
        raise HTTPException(status_code=503, detail=ARIA2C_MISSING)

    header_args: list[str] = []
    for k, v in [("User-Agent", CIVITAI_CONFIG.ua), *CIVITAI_CONFIG.extra_headers.items()]:
        header_args += ["--header", f"{k}: {v}"]
    api_key = CIVITAI_CONFIG.api_key
    if api_key:
        header_args += ["--header", f"Authorization: Bearer {api_key}"]

    logger.info("lora/download: version_id=%s filename=%s trigger=%r", p.version_id, p.filename, p.trigger)
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(CIVITAI_CONFIG.connect_timeout, read=CIVITAI_CONFIG.read_timeout),
            proxy=CIVITAI_CONFIG.proxy,
            headers=_metadata_headers(),
            follow_redirects=True,
        ) as client:
            expected_sha, total_bytes = await _precheck_version(client, p.version_id)
    except httpx.HTTPError as e:
        logger.error("lora/download: civitai request failed: %r", e)
        raise HTTPException(status_code=502, detail="cannot reach civitai") from e

    download_dir = CIVITAI_CONFIG.loras_download_dir
    os.makedirs(download_dir, exist_ok=True)

    async with _file_lock(p.filename):
        tmp_path = os.path.join(download_dir, p.filename + ".part")

        if on_progress is not None and total_bytes:
            # 续传场景先把已有残片量报出去，调用方立刻有进度可显示
            done = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
            await on_progress(
                {
                    "bytes": done,
                    "total": total_bytes,
                    "percent": round(done * 100 / total_bytes, 1),
                    "speed_bps": 0,
                    "eta_s": None,
                }
            )

        try:
            await _run_aria2(
                f"{CIVITAI_CONFIG.base_url}/api/download/models/{p.version_id}",
                header_args,
                download_dir,
                p.filename + ".part",
                on_progress=on_progress,
                cancel=cancel,
            )
        except (TimeoutError, RuntimeError) as e:
            detail = f"civitai download failed ({e}); partial file kept, retry to resume"
            if not api_key:
                detail += "; no CIVITAI_API_KEY configured, gated models cannot be downloaded"
            raise HTTPException(status_code=502, detail=detail) from e

        # HTTP 200 + 错误体：Cloudflare 质询页 / 鉴权跳转页 / 运营商劫持页 / 自建镜像的兜底页
        # 都会以 200 返回 HTML，aria2 会把它当文件落盘并报"下载成功"。嗅探开头字节 + 与元数据
        # 声明的大小对账，当场点名"下到的不是模型文件"，而不是让 SHA 校验报出误导性的 mismatch。
        size = os.path.getsize(tmp_path)
        with open(tmp_path, "rb") as f:
            looks_like_page = f.read(64).lstrip().startswith(b"<")
        size_off = total_bytes is not None and abs(size - total_bytes) > max(total_bytes * 0.02, 1024)
        if looks_like_page or size_off:
            os.remove(tmp_path)
            detail = (
                "download url returned an HTML page instead of the model file"
                if looks_like_page
                else f"downloaded {size} bytes but civitai metadata says {total_bytes} bytes"
            )
            raise HTTPException(
                status_code=502,
                detail=f"{detail}; the download endpoint did not serve the model file — common causes: "
                "an anti-bot/Cloudflare challenge page, a gated model needing CIVITAI_API_KEY, or a "
                "CIVITAI_BASE_URL/CIVITAI_PROXY mirror not forwarding the 302-to-CDN download redirect",
            )

        # 全量 SHA256 对大文件是秒级同步 I/O：下放线程池，别卡住事件循环
        await asyncio.to_thread(verify_lora_file, tmp_path, expected_sha)
        size = os.path.getsize(tmp_path)  # 必须在 replace 前取：之后 tmp_path 已不存在
        os.replace(tmp_path, os.path.join(download_dir, p.filename))
        update_registry(p.filename, p.version_id, p.trigger)
        logger.info("lora/download: saved %s (%d bytes, %.1fs)", p.filename, size, time.monotonic() - started)
        return {"filename": p.filename, "version_id": p.version_id, "trigger": p.trigger}


async def _run_aria2(
    url: str,
    header_args: list[str],
    dest_dir: str,
    out_name: str,
    on_progress: _ProgressCallback | None = None,
    cancel: asyncio.Event | None = None,
) -> None:
    """
    调用 aria2 下载；汇总行解析后交给 on_progress，cancel 置位即杀进程（残片保留续传），
    超时/非零退出抛 TimeoutError/RuntimeError。
    """

    cmd = [
        "stdbuf",
        "-oL",
        "aria2c",
        "--continue",
        "--dir",
        dest_dir,
        "--out",
        out_name,
        "--max-connection-per-server",
        str(ARIA2_CONFIG.connections),
        "--split",
        str(ARIA2_CONFIG.connections),
        "--timeout",
        str(ARIA2_CONFIG.connect_timeout),
        "--max-tries",
        str(ARIA2_CONFIG.max_tries),
        "--retry-wait",
        str(ARIA2_CONFIG.retry_wait),
        "--lowest-speed-limit",
        ARIA2_CONFIG.lowest_speed_limit,
        "--file-allocation",
        "none",
        "--auto-file-renaming=false",
        "--auto-save-interval=1",
        # 每秒一条汇总行：既是日志也是进度源（--summary-interval 1）
        "--summary-interval",
        "1",
        *header_args,
        url,
    ]
    if CIVITAI_CONFIG.proxy:
        cmd += ["--all-proxy", CIVITAI_CONFIG.proxy]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )
    except FileNotFoundError as e:
        # 与启动时的软禁用同文案：PATH 变化等极端情况下的兜底
        raise HTTPException(status_code=503, detail=ARIA2C_MISSING) from e

    async def drain() -> None:
        if proc.stdout is None:  # pragma: no cover - PIPE 下必非 None
            raise RuntimeError("aria2c stdout is missing")
        async for line in proc.stdout:
            text = line.decode(errors="replace").replace("\r", "\n").strip()
            if not text:
                continue
            logger.info("lora/download: aria2: %s", text)
            if on_progress is not None:
                parsed = _parse_aria2_progress(text)
                if parsed is not None:
                    await on_progress(parsed)
        rc = await proc.wait()
        if rc != 0:
            raise RuntimeError(f"aria2c exit code {rc}")

    # 读输出、等取消、等总超时三件事竞速：谁先到处理谁
    reader = asyncio.create_task(drain())
    canceller = asyncio.create_task(cancel.wait()) if cancel is not None else None
    try:
        pending: set[asyncio.Task[Any]] = {reader}
        if canceller is not None:
            pending.add(canceller)
        done, _ = await asyncio.wait(
            pending, timeout=CIVITAI_CONFIG.download_timeout, return_when=asyncio.FIRST_COMPLETED
        )
        if canceller in done:
            raise asyncio.CancelledError  # 调用方主动取消：残片保留，交上层转 cancelled
        if reader in done:
            await reader  # 非零退出在这里抛 RuntimeError
            return  # 正常完成
        raise TimeoutError
    finally:
        if canceller is not None:
            canceller.cancel()
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
        if not reader.done():
            reader.cancel()
        # 收尾阶段的取消/被杀退出码不上抛；真正要抛的异常已在上面 await 时抛出
        with contextlib.suppress(asyncio.CancelledError, RuntimeError):
            await reader
