import asyncio
import contextlib
import logging
import re
import shutil
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from fastapi import HTTPException

from comfyui_anima_api.config import ARIA2_CONFIG

logger = logging.getLogger(__name__)

MISSING_MESSAGE = "aria2c is not installed on the server; /lora/download is disabled (install aria2 and restart)"

# 下载进度回调：aria2 汇总行解析出的进度字典
ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]]

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


def is_available() -> bool:
    return shutil.which("aria2c") is not None


def build_headers(headers: Mapping[str, str]) -> list[str]:
    """把 HTTP 头字典转换为 aria2c 的 --header 参数对。"""
    args: list[str] = []
    for key, value in headers.items():
        args.extend(["--header", f"{key}: {value}"])
    return args


async def run(
    url: str,
    header_args: list[str],
    dest_dir: str,
    out_name: str,
    *,
    timeout: float,
    on_progress: ProgressCallback | None = None,
    cancel: asyncio.Event | None = None,
    proxy: str | None = None,
) -> None:
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
        "--summary-interval",
        "1",
        *header_args,
        url,
    ]
    if proxy:
        cmd += ["--all-proxy", proxy]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=MISSING_MESSAGE) from e

    async def drain() -> None:
        if proc.stdout is None:
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

    reader = asyncio.create_task(drain())
    canceller = asyncio.create_task(cancel.wait()) if cancel is not None else None
    try:
        pending: set[asyncio.Task[Any]] = {reader}
        if canceller is not None:
            pending.add(canceller)
        done, _ = await asyncio.wait(pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        if canceller in done:
            raise asyncio.CancelledError
        if reader in done:
            await reader
            return
        raise TimeoutError
    finally:
        if canceller is not None:
            canceller.cancel()
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
        if not reader.done():
            reader.cancel()
        with contextlib.suppress(asyncio.CancelledError, RuntimeError):
            await reader
