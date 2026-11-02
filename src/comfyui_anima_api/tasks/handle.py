"""
TaskHandle：任务 runner 与任务框架之间的接口（Protocol，运行态由 tasks.service 提供）。
generation / upscale / loras 三个域只依赖本模块，不反向依赖 tasks.service，避免环引用。
"""

import asyncio
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TaskHandle(Protocol):
    """
    runner 视角的任务句柄：task_id 用于命名输出子目录；emit 上报进度事件
    （progress / image_done / lora_progress / status 相位）；add_image 登记一张
    产出图片并返回其 1 起始序号；cancel_event 置位表示调用方请求取消，runner
    应尽快抛 CancelledError 退出（框架统一转为 cancelled 终态）。
    """

    @property
    def task_id(self) -> str: ...

    @property
    def cancel_event(self) -> asyncio.Event: ...

    async def emit(self, event_type: str, data: dict[str, Any]) -> None: ...

    async def set_prompt_id(self, prompt_id: str | None) -> None: ...

    async def add_image(self, image: dict[str, Any], prompt_id: str | None) -> int: ...
