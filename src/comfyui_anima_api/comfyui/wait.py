"""
等待完成与结果收集：ws feed 优先、/history 轮询兜底的等待编排。
单条 prompt 的等待语义收口在 await_prompt：在线优先走共享 feed（步级进度回调），
断连或本就未连接则降级轮询；完成后的 entry 校验与图片清单收集也在这里。
"""

import asyncio
import time
from typing import Any

import httpx
from fastapi import HTTPException

from comfyui_anima_api.comfyui._shared import _json, _timeout
from comfyui_anima_api.comfyui.http import fetch_history_entry
from comfyui_anima_api.comfyui.ws import ProgressCallback, comfyui_progress
from comfyui_anima_api.config import COMFYUI_CONFIG


async def _feed_get(feed: asyncio.Queue, cancel: asyncio.Event | None, timeout: float) -> dict | None:
    """等下一条 feed 消息；cancel 置位返回 None，超时抛 TimeoutError。双任务竞速，长无消息期也能立刻响应取消。"""
    getter = asyncio.ensure_future(feed.get())
    waiter = asyncio.ensure_future(cancel.wait()) if cancel is not None else None
    try:
        pending: set[asyncio.Task[Any]] = {getter}
        if waiter is not None:
            pending.add(waiter)
        done, _ = await asyncio.wait(pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        if waiter in done:
            return None
        if getter in done:
            return getter.result()
        raise TimeoutError
    finally:
        for t in (getter, waiter):
            if t is not None and not t.done():
                t.cancel()


async def wait_for_history(
    client: httpx.AsyncClient, prompt_id: str, timeout: float, cancel: asyncio.Event | None = None
) -> dict | None:
    """
    轮询 /history/<id> 直到完成或超时；返回 history entry（完成或出错），超时返回 None。
    cancel 置位立即抛 CancelledError（响应延迟至多一个 poll_interval）。
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cancel is not None and cancel.is_set():
            raise asyncio.CancelledError
        r = await client.get(f"{COMFYUI_CONFIG.base_url}/history/{prompt_id}")
        entry = _json(r, "response to /history").get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error" or status.get("completed"):
                return entry
        if cancel is not None:
            try:
                if await asyncio.wait_for(cancel.wait(), COMFYUI_CONFIG.poll_interval):
                    raise asyncio.CancelledError
            except TimeoutError:
                pass  # 超时不是错误：这正是轮询节拍，继续下一轮
        else:
            await asyncio.sleep(COMFYUI_CONFIG.poll_interval)
    return None


async def _await_via_feed(
    client: httpx.AsyncClient,
    feed: asyncio.Queue,
    prompt_id: str,
    deadline: float,
    on_progress: ProgressCallback | None,
    cancel: asyncio.Event | None,
) -> dict | None:
    """
    走 ws feed 等待完成，返回 history entry；ws 断连返回 None（由调用方降级轮询）。
    progress 消息回调 on_progress；executing(node=None) / execution_error 即完成。
    """
    while True:
        if cancel is not None and cancel.is_set():
            raise asyncio.CancelledError
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _timeout(prompt_id)
        try:
            message = await _feed_get(feed, cancel, remaining)
        except TimeoutError:
            raise _timeout(prompt_id) from None
        if message is None:
            raise asyncio.CancelledError
        mtype = message.get("type")
        if mtype == "_disconnected":
            return None
        data = message.get("data") or {}
        # 同一条共享 feed 上还有别的并发任务的事件：只认自己的 prompt_id
        if data.get("prompt_id") != prompt_id:
            continue
        if mtype == "progress":
            if on_progress is not None:
                await on_progress(int(data.get("value", 0)), int(data.get("max", 0)))
        elif mtype == "execution_error" or (mtype == "executing" and data.get("node") is None):
            entry = await fetch_history_entry(client, prompt_id)
            if entry is None:
                raise HTTPException(
                    status_code=500, detail={"error": "prompt finished but history is missing", "prompt_id": prompt_id}
                )
            return entry


async def await_prompt(
    client: httpx.AsyncClient,
    feed: asyncio.Queue,
    prompt_id: str,
    timeout: float,
    on_progress: ProgressCallback | None = None,
    cancel: asyncio.Event | None = None,
) -> dict:
    """
    等待单个 prompt 执行完成并返回其 history entry（超时 504 / 取消 CancelledError）。
    ws 在线优先走 feed（步级进度回调），断连或本就未连接则 /history 轮询兜底。
    """
    deadline = time.monotonic() + timeout
    if comfyui_progress.is_connected():
        entry = await _await_via_feed(client, feed, prompt_id, deadline, on_progress, cancel)
        if entry is not None:
            return entry
    entry = await wait_for_history(client, prompt_id, deadline - time.monotonic(), cancel=cancel)
    if entry is None:
        raise _timeout(prompt_id)
    return entry


def validate_history_entry(entry: dict, prompt_id: str) -> None:
    """history entry 完成态校验：执行失败抛 500（携带 ComfyUI 原始 messages）。"""
    if entry.get("status", {}).get("status_str") == "error":
        raise HTTPException(
            status_code=500,
            detail={
                "error": "execution failed",
                "prompt_id": prompt_id,
                "messages": entry.get("status", {}).get("messages"),
            },
        )


def collect_output_images(entry: dict) -> list[dict]:
    """收集 history entry 的全部输出图片（跨节点、按节点顺序），过滤无文件名项。"""
    images: list[dict] = []
    for node_output in entry.get("outputs", {}).values():
        images.extend(node_output.get("images") or [])
    return [img for img in images if img.get("filename")]
