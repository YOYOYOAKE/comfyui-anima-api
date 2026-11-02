"""
任务编排：三类任务（generate / upscale / lora_download）共用的创建、执行、取消与重启对账。
流程：入库 queued -> 发 status 事件 -> asyncio.create_task 起后台 runner；runner 只报
进度类事件（progress / image_done / lora_progress），终态（done / error / cancelled）统一由
本模块收口——HTTPException 转 error（保留原状态码语义），CancelledError 转 cancelled，
正常返回转 done（downloads runner 的返回值并入 done 事件）。
"""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from fastapi import HTTPException

from comfyui_anima_api.comfyui import cancel_prompt, collect_output_images, fetch_history_entry
from comfyui_anima_api.config import COMFYUI_CONFIG
from comfyui_anima_api.schemas import DownloadParams, GenerateParams, UpscaleParams
from comfyui_anima_api.tasks.bus import bus
from comfyui_anima_api.tasks.handle import TaskHandle
from comfyui_anima_api.tasks.store import TERMINAL_STATUSES, get_store

logger = logging.getLogger(__name__)

_TaskParams = GenerateParams | UpscaleParams | DownloadParams
_Runner = Callable[[TaskHandle, Any], Awaitable[dict[str, Any] | None]]

_LIVE: dict[str, asyncio.Event] = {}

# 后台执行协程的引用，避免任务 GC 回收
_BACKGROUND: set[asyncio.Task[None]] = set()


class _TaskHandleImpl:
    """TaskHandle（Protocol）的运行态实现：emit 进总线（先落库再扇出），任务行与图片引用落库。"""

    def __init__(self, task_id: str, cancel_event: asyncio.Event) -> None:
        self.task_id = task_id
        self.cancel_event = cancel_event

    async def emit(self, event_type: str, data: dict[str, Any]) -> None:
        await bus.emit(self.task_id, event_type, data)

    async def set_prompt_id(self, prompt_id: str | None) -> None:
        await get_store().update_task(self.task_id, prompt_id=prompt_id)

    async def add_image(self, image: dict[str, Any], prompt_id: str | None) -> int:
        return await get_store().add_image(self.task_id, image, prompt_id)


async def _run_download(handle: TaskHandle, p: DownloadParams) -> dict[str, Any]:
    """LoRA 下载 runner：aria2 进度原样转发为 lora_progress 事件。"""
    from comfyui_anima_api.loras import download

    async def on_progress(progress: dict[str, Any]) -> None:
        await handle.emit("lora_progress", progress)

    return await download(p, on_progress=on_progress, cancel=handle.cancel_event)


def _resolve_runner(task_type: str) -> _Runner:
    """
    域 runner 的运行时解析。刻意不在模块顶层导入各域：generation/upscale 的 service
    依赖 tasks.TaskHandle，runner 注册又反过来依赖它们——顶层互导成环，运行时导入斩断。
    """
    if task_type == "generate":
        from comfyui_anima_api.generation import run_generate

        return run_generate
    if task_type == "upscale":
        from comfyui_anima_api.upscale import run_upscale

        return run_upscale
    if task_type == "lora_download":
        return _run_download
    raise ValueError(f"unknown task type {task_type!r}")


def _http_message(e: HTTPException) -> str:
    if isinstance(e.detail, str):
        return e.detail
    return json.dumps(e.detail, ensure_ascii=False, default=str)[:1000]


def _sanitize_params(params: _TaskParams) -> dict[str, Any]:
    """入库参数清洗，避免 Base64 完整入库。"""

    data: dict[str, Any] = params.model_dump()
    if isinstance(params, GenerateParams):
        for group in ("base", "reference"):
            for field in ("image", "mask"):
                blob = data.get(group, {}).get(field)
                if blob:
                    data[group][field] = f"<base64 omitted: {len(blob)} chars>"
    elif isinstance(params, UpscaleParams):
        data["image"] = f"<base64 omitted: {len(data['image'])} chars>"
    return data


async def create_task(task_type: str, params: _TaskParams, images_total: int = 0) -> str:
    runner = _resolve_runner(task_type)
    task_id = uuid.uuid4().hex[:12]
    await get_store().create_task(task_id, task_type, _sanitize_params(params), images_total=images_total)
    cancel_event = asyncio.Event()
    _LIVE[task_id] = cancel_event
    await bus.emit(task_id, "status", {"status": "queued"})
    background = asyncio.create_task(
        _execute(task_id, runner, _TaskHandleImpl(task_id, cancel_event), params), name=f"task-{task_type}-{task_id}"
    )
    _BACKGROUND.add(background)
    background.add_done_callback(_BACKGROUND.discard)
    return task_id


async def _execute(task_id: str, runner: _Runner, handle: _TaskHandleImpl, params: Any) -> None:
    """后台执行一个任务并收口终态。异常全部就地消化（后台任务裸抛只会进 asyncio 日志）。"""
    store = get_store()
    await store.update_task(task_id, status="running", started_at=time.time())
    await bus.emit(task_id, "status", {"status": "running"})
    try:
        result = await runner(handle, params)
    except asyncio.CancelledError:
        await _finish(task_id, "cancelled")
    except HTTPException as e:
        if handle.cancel_event.is_set():
            # 取消途中的连锁失败（如 interrupt 触发的执行错误）归因为取消
            await _finish(task_id, "cancelled")
        else:
            await _finish(task_id, "error", error=_http_message(e), extra={"http_status": e.status_code})
    except Exception as e:
        logger.exception("task[%s]: runner crashed", task_id)
        await _finish(task_id, "error", error=f"internal error: {e!r}", extra={"http_status": 500})
    else:
        extra = {"result": result} if result else None
        await _finish(task_id, "done", extra=extra)
    finally:
        _LIVE.pop(task_id, None)


async def _finish(task_id: str, status: str, *, error: str | None = None, extra: dict[str, Any] | None = None) -> None:
    """终态收口（幂等：处于终态的任务不重复改写）：任务行 + done/error/cancelled 事件。"""
    store = get_store()
    task = await store.get_task(task_id)
    if task is None or task["status"] in TERMINAL_STATUSES:
        return
    fields: dict[str, Any] = {"status": status, "finished_at": time.time()}
    if error is not None:
        fields["error"] = error[:1000]
    await store.update_task(task_id, **fields)
    task = await store.get_task(task_id) or task
    data: dict[str, Any] = {"status": status}
    if status == "done":
        if task["type"] in ("generate", "upscale"):
            data["images"] = task["images_done"]
        # finished_at 刚在上方写入必非 None，但行类型是 float | None，用 or 兜底取 created_at
        finished = task["finished_at"] or task["created_at"]
        data["total_seconds"] = round(finished - (task["started_at"] or task["created_at"]), 1)
    if error is not None:
        data["message"] = fields["error"]
    if extra:
        data.update(extra)
    await bus.emit(task_id, status, data)


async def cancel_task(task_id: str) -> dict[str, str]:
    """取消任务（幂等）：置本地取消事件 + 撤 ComfyUI 侧 prompt（执行中 interrupt / 排队中出队）。"""
    task = await get_store().get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"unknown task {task_id}")
    if task["status"] in TERMINAL_STATUSES:
        return {"task_id": task_id, "status": task["status"]}
    cancel_event = _LIVE.get(task_id)
    if cancel_event is not None:
        cancel_event.set()
    if task["type"] in ("generate", "upscale") and task["prompt_id"]:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(COMFYUI_CONFIG.connect_timeout)) as client:
                await cancel_prompt(client, task["prompt_id"])
        except httpx.HTTPError as e:
            # 尽力而为：本地取消事件已置位，任务仍会终止，只是 ComfyUI 可能白跑完当前图
            logger.warning("task[%s]: comfyui-side cancel failed (best-effort): %r", task_id, e)
    return {"task_id": task_id, "status": "cancelling"}


async def reconcile() -> None:
    """
    启动对账：API 重启后把重启前未完的任务逐个收口。生成/超分若 ComfyUI 已完成
    （prompt_id 有 history），补登记图片并标 done；其余（还在跑/排队、下载中、查不到）
    一律标 error 说明因重启丢失——ComfyUI 侧 prompt 可能仍在跑，调用方可凭 prompt_id
    直查 ComfyUI /history 自行取回。
    """
    store = get_store()
    for task in await store.unfinished_tasks():
        task_id = task["id"]
        logger.info("reconcile[%s]: %s task left %s", task_id, task["type"], task["status"])
        entry = None
        if task["type"] in ("generate", "upscale") and task["prompt_id"]:
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(COMFYUI_CONFIG.health_timeout)) as client:
                    entry = await fetch_history_entry(client, task["prompt_id"])
            except httpx.HTTPError as e:
                logger.warning("reconcile[%s]: history lookup failed: %r", task_id, e)
        if entry is not None:
            if entry.get("status", {}).get("status_str") == "error":
                await _finish(
                    task_id,
                    "error",
                    error="execution failed (finalized after api restart)",
                    extra={"http_status": 500, "messages": entry.get("status", {}).get("messages")},
                )
                continue
            if entry.get("status", {}).get("completed"):
                for image in collect_output_images(entry):
                    await store.add_image(task_id, image, task["prompt_id"])
                # 批量任务重启前可能只完成部分张数：最后一个 prompt 完成不等于整批完成。
                # 仍标 done（已产出图片真实可取），done 事件附 truncated 供调用方区分整批/残批
                done_images = len(await store.images(task_id))
                extra: dict[str, Any] | None = {"truncated": True} if done_images < task["images_total"] else None
                await _finish(task_id, "done", extra=extra)
                continue
        hint = ""
        if task["prompt_id"]:
            hint = f"; prompt {task['prompt_id']} may still be running in ComfyUI, check its /history"
        if task["type"] == "lora_download":
            hint = "; partial file kept, resubmit the same request to resume"
        await _finish(task_id, "error", error=f"api restarted before task finished{hint}", extra={"http_status": 503})
