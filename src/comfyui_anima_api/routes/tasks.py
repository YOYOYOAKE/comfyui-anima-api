"""任务查询与操控：状态快照 / SSE 事件流 / 图片 ZIP / 取消。业务端点只负责创建任务返回 202，进度与产物全部从这里取。"""

import asyncio
import io
import logging
import zipfile
from collections.abc import AsyncIterator

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from comfyui_anima_api.comfyui import fetch_image_bytes
from comfyui_anima_api.config import COMFYUI_CONFIG
from comfyui_anima_api.tasks import TERMINAL_STATUSES, cancel_task, get_store, sse_response

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/tasks/{task_id}")
async def task_snapshot(task_id: str) -> dict:
    """
    任务快照：状态 queued → running → done | error | cancelled

    params 为请求参数回显（b64 图片字段替换为占位符）；images 为已产出图片清单
    （index + filename，本体在 ComfyUI output/ 里，经 images.zip 取回）。
    """
    task = await get_store().get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"unknown task {task_id}")
    images = await get_store().images(task_id)
    return {
        "task_id": task["id"],
        "type": task["type"],
        "status": task["status"],
        "error": task["error"],
        "prompt_id": task["prompt_id"],
        "images_total": task["images_total"],
        "images_done": task["images_done"],
        "images": [{"index": img["idx"], "filename": img["filename"]} for img in images],
        "params": task["params"],
        "created_at": task["created_at"],
        "started_at": task["started_at"],
        "finished_at": task["finished_at"],
    }


@router.get("/tasks/{task_id}/events")
async def task_events(task_id: str, request: Request) -> Response:
    """
    SSE 进度流（text/event-stream），任务达到终态后关流

    事件三段式 id / event / data(JSON)，id 为全局自增 seq；断线重连带
    `Last-Event-ID` 请求头即可从断点续收（回放走 SQLite），手动重连可用
    `?lastEventId=<seq>` 等价指定；空闲每 15s 发一行 `: ping` 保活。

    事件类型与 data 载荷：

    - status: `{"status": "queued | running"}`
    - progress: `{"image": 1, "images_total": 1, "step": 14, "steps": 30}`
    - image_done: `{"image": 1, "images_total": 1, "filename": "001_00001_.png", "seconds": 41.2}`
    - lora_progress: `{"bytes": 47185920, "total": 2040109465, "percent": 2.3, "speed_bps": 5452595, "eta_s": 354.0}`
    - done: `{"status": "done", "images": 1, "total_seconds": 128.4}`（残批附 `"truncated": true`）
    - error: `{"status": "error", "http_status": 502, "message": "…"}`
    - cancelled: `{"status": "cancelled"}`
    """
    if await get_store().get_task(task_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown task {task_id}")
    raw = request.headers.get("last-event-id") or request.query_params.get("lastEventId") or "0"
    try:
        after_seq = max(int(raw), 0)
    except ValueError:
        after_seq = 0
    return await sse_response(task_id, after_seq)


class _ZipSink(io.RawIOBase):
    """zipfile 的同步写降落点：攒块，由流式生成器收走；不可寻址使 zipfile 走 data descriptor 路径，无需完整缓冲。"""

    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    def write(self, data: bytes) -> int:  # type: ignore[override]  # mmap/bytes 均按 bytes 收
        self.chunks.append(data)
        return len(data)


@router.get("/tasks/{task_id}/images.zip")
async def task_images_zip(task_id: str) -> StreamingResponse:
    """
    任务终态（done | error | cancelled）后的产物整包 ZIP：未终态一律 409——
    半套图片没有消费价值，调用方也不必处理"包不完整"的判断。终态但无图片
    （如 error 任务）→ 404。图片本体一直存在 ComfyUI output/ 里，本端点按
    task_images 登记表现取现打包。流式：逐张现取现写，内存占用 O(1)。
    代价是响应一经开始便无法改口状态码——打包途中取图失败以 _INCOMPLETE.txt
    条目收尾（客户端按 ZIP 完整性即可识别）。
    """
    task = await get_store().get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"unknown task {task_id}")
    if task["status"] not in TERMINAL_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=f"task {task_id} is still {task['status']}; images.zip is only available after the task "
            "reaches a terminal state (done | error | cancelled)",
        )
    images = await get_store().images(task_id)
    if not images:
        raise HTTPException(status_code=404, detail=f"task {task_id} has no images")

    async def stream() -> AsyncIterator[bytes]:
        sink = _ZipSink()
        # PNG 本身已压缩，ZIP_STORED 只打包不 deflate，省 CPU
        zf = zipfile.ZipFile(sink, "w", zipfile.ZIP_STORED)
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(COMFYUI_CONFIG.connect_timeout, read=COMFYUI_CONFIG.connect_timeout * 6)
        ) as client:
            for img in images:
                try:
                    blob = await fetch_image_bytes(client, img)
                except httpx.HTTPError as e:
                    logger.warning("tasks/%s/images.zip: fetch %s failed: %r", task_id, img["filename"], e)
                    await asyncio.to_thread(
                        zf.writestr, "_INCOMPLETE.txt", f"fetch failed at {img['filename']}: {e!r}\n"
                    )
                    break
                await asyncio.to_thread(zf.writestr, f"{img['idx']:03d}_{img['filename']}", blob)
                if sink.chunks:
                    yield b"".join(sink.chunks)
                    sink.chunks.clear()
        await asyncio.to_thread(zf.close)
        if sink.chunks:
            yield b"".join(sink.chunks)
            sink.chunks.clear()

    return StreamingResponse(
        stream(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{task_id}.zip"',
            "X-Images-Count": str(len(images)),
        },
    )


@router.post("/tasks/{task_id}/cancel")
async def task_cancel(task_id: str) -> dict:
    """
    取消任务（幂等）：置本地取消事件 + 撤 ComfyUI 侧 prompt（执行中 interrupt / 排队中出队）

    已终态的任务再取消直接返回当前状态，不报错。
    """
    return await cancel_task(task_id)
