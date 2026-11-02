"""
SSE 输出：text/event-stream 流式响应（手写格式化，不引第三方库）。
- 事件三段式 id / event / data(JSON)：id 是全局自增 seq，调用方断线重连带
  Last-Event-ID 请求头即可从断点续收（回放走总线内存缓冲，任务在服务端继续跑；
  API 重启后缓冲不复存在，库里仅存的终态事件行同样足以收尾）；
- 空闲每 15s 发一行注释 ": ping" 保活，防中间设备掐空闲连接；
- 推到终态事件（done/error/cancelled）后流正常关闭；客户端中途断开由生成器
  取消时的 finally 注销订阅，不影响任务执行。
"""

import asyncio
import json
import typing
from collections.abc import Mapping
from typing import Any

from fastapi.responses import StreamingResponse

from comfyui_anima_api.tasks.bus import TERMINAL_EVENTS, bus

HEARTBEAT_SECONDS = 15.0


def _format_event(event: Mapping[str, Any]) -> str:
    return f"id: {event['seq']}\nevent: {event['type']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"


async def _sse_stream(task_id: str, after_seq: int) -> typing.AsyncIterator[str]:
    replay, queue, task_terminal = await bus.subscribe(task_id, after_seq)
    try:
        for event in replay:
            yield _format_event(event)
            if event["type"] in TERMINAL_EVENTS:
                return
        if task_terminal:
            # 回放不含终态而任务已终态：after_seq 已越过终态事件（EventSource 正常
            # 关流后仍会自动重连），没有后续事件，立即关流而非挂着发心跳
            return
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), HEARTBEAT_SECONDS)
            except TimeoutError:
                yield ": ping\n\n"
                continue
            yield _format_event(event)
            if event["type"] in TERMINAL_EVENTS:
                return
    finally:
        bus.unsubscribe(task_id, queue)


async def sse_response(task_id: str, after_seq: int) -> StreamingResponse:
    # X-Accel-Buffering: no —— nginx 反代默认缓冲响应会攒成一坨，必须显式关掉
    return StreamingResponse(
        _sse_stream(task_id, after_seq),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
