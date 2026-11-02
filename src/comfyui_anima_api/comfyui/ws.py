"""
WebSocket 进度通道：/ws 单连接管理与订阅扇出。

进度通道是单条共享 ws 连接（按 client_id 路由），订阅者拿到全量消息队列、
自己按 prompt_id 过滤——提交前先订阅，事件天然不丢；ws 断线自动重连，
等待方收到 _disconnected 即降级 /history 轮询（此间无步级进度，只有完成态）。
"""

import asyncio
import contextlib
import json
import logging
from collections.abc import Awaitable, Callable

import websockets

from comfyui_anima_api.config import COMFYUI_CONFIG

logger = logging.getLogger(__name__)

# 步级进度回调：(当前步, 总步数)；由 wait.await_prompt 在收到 ws progress 消息时调用
ProgressCallback = Callable[[int, int], Awaitable[None]]


class ComfyUIProgress:
    """
    ComfyUI /ws 单连接管理：懒启动（首个订阅者到来才连）、断线指数退避重连
    （1s 起翻倍、封顶 30s）。订阅者收到的是全量消息的副本队列；连接断开/恢复以
    合成消息 _disconnected / _connected 通知，等待方据此降级或恢复。
    """

    def __init__(self) -> None:
        self._subs: set[asyncio.Queue] = set()
        self._task: asyncio.Task[None] | None = None
        self._connected: bool = False

    def is_connected(self) -> bool:
        return self._connected

    def _url(self) -> str:
        base = COMFYUI_CONFIG.base_url.replace("http://", "ws://").replace("https://", "wss://")
        return f"{base}/ws?clientId={COMFYUI_CONFIG.client_id}"

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="comfyui-ws")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self._connected = False
        self._subs.clear()

    async def subscribe(self) -> asyncio.Queue:
        await self.start()
        # 封顶：停滞的等待方不会无限累积步级进度（正常负载每秒几条，256 绰绰有余）；
        # 挤掉最老一条的代价是漏看进度，漏掉的完成信号由 /history 轮询兜底
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._subs.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subs.discard(queue)

    def _broadcast(self, message: dict) -> None:
        for queue in self._subs:
            try:
                queue.put_nowait(dict(message))
            except asyncio.QueueFull:
                with contextlib.suppress(asyncio.QueueEmpty, ValueError):
                    queue.get_nowait()
                queue.put_nowait(dict(message))

    async def _run(self) -> None:
        backoff = 1.0
        while True:
            was_connected = self._connected
            try:
                async with websockets.connect(
                    self._url(), open_timeout=COMFYUI_CONFIG.health_timeout, max_size=2**24
                ) as ws:
                    logger.info("comfyui ws: connected (%s)", self._url())
                    self._connected = True
                    backoff = 1.0
                    self._broadcast({"type": "_connected"})
                    async for message in ws:
                        # 二进制帧是预览图（本 API 的工作流不产出），直接丢弃
                        if isinstance(message, str):
                            self._dispatch(message)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("comfyui ws: dropped (%r), reconnecting in %.0fs", e, backoff)
            finally:
                if was_connected or self._connected:
                    self._connected = False
                    self._broadcast({"type": "_disconnected"})
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    def _dispatch(self, message: str) -> None:
        try:
            parsed = json.loads(message)
        except ValueError:
            logger.debug("comfyui ws: non-JSON message dropped: %.200s", message)
            return
        if isinstance(parsed, dict):
            self._broadcast(parsed)


# 全局单例：应用生命周期内唯一的一条 ws 连接（app lifespan 负责 start/stop）
comfyui_progress = ComfyUIProgress()
