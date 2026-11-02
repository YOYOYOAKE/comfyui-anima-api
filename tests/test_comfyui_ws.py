"""
comfyui 包 ws 进度通道的测试：共享 feed 过滤、断连降级与广播封顶。不连真 ws：
_await_via_feed 直接喂假 feed 队列；ComfyUIProgress 用假 websockets.connect 驱动。
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
import respx
from fastapi import HTTPException

import comfyui_anima_api.comfyui.ws as ws_mod
from comfyui_anima_api.comfyui import ComfyUIProgress
from comfyui_anima_api.comfyui.wait import _await_via_feed
from comfyui_anima_api.config import COMFYUI_CONFIG

COMFY = "http://comfy-ws-test:8188"

ENTRY: dict[str, Any] = {"status": {"status_str": "success", "completed": True}, "outputs": {}}


def ws_msg(msg_type: str, **data: Any) -> str:
    return json.dumps({"type": msg_type, "data": data})


class TestAwaitViaFeed:
    """_await_via_feed：走共享 ws feed 等待单个 prompt 完成。"""

    @pytest.fixture(autouse=True)
    def _patch_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(COMFYUI_CONFIG, "base_url", COMFY)

    def test_progress_filtered_and_completion_fetched(self) -> None:
        async def scenario() -> tuple[dict[str, Any] | None, list[tuple[int, int]]]:
            feed: asyncio.Queue = asyncio.Queue()
            feed.put_nowait({"type": "progress", "data": {"prompt_id": "pid", "value": 3, "max": 30}})
            feed.put_nowait({"type": "progress", "data": {"prompt_id": "other", "value": 9, "max": 9}})
            feed.put_nowait({"type": "executing", "data": {"prompt_id": "pid", "node": None}})
            seen: list[tuple[int, int]] = []

            async def on_step(value: int, max_value: int) -> None:
                seen.append((value, max_value))

            async with httpx.AsyncClient() as client:
                entry = await _await_via_feed(client, feed, "pid", time.monotonic() + 5, on_step, None)
            return entry, seen

        with respx.mock:
            respx.get(f"{COMFY}/history/pid").mock(return_value=httpx.Response(200, json={"pid": ENTRY}))
            entry, seen = asyncio.run(scenario())
        # 只认自己 prompt_id 的进度；完成信号触发 /history 补取并返回 entry
        assert seen == [(3, 30)]
        assert entry == ENTRY

    def test_execution_error_also_completes(self) -> None:
        async def scenario() -> dict[str, Any] | None:
            feed: asyncio.Queue = asyncio.Queue()
            feed.put_nowait({"type": "execution_error", "data": {"prompt_id": "pid"}})
            async with httpx.AsyncClient() as client:
                return await _await_via_feed(client, feed, "pid", time.monotonic() + 5, None, None)

        with respx.mock:
            respx.get(f"{COMFY}/history/pid").mock(return_value=httpx.Response(200, json={"pid": ENTRY}))
            assert asyncio.run(scenario()) == ENTRY

    def test_disconnect_degrades_to_none(self) -> None:
        async def scenario() -> dict[str, Any] | None:
            feed: asyncio.Queue = asyncio.Queue()
            feed.put_nowait({"type": "_disconnected", "data": {}})
            async with httpx.AsyncClient() as client:
                return await _await_via_feed(client, feed, "pid", time.monotonic() + 5, None, None)

        # ws 断连返回 None：由调用方降级 /history 轮询（await_prompt 的兜底路径）
        assert asyncio.run(scenario()) is None

    def test_deadline_passed_raises_504(self) -> None:
        async def scenario() -> dict[str, Any] | None:
            feed: asyncio.Queue = asyncio.Queue()
            async with httpx.AsyncClient() as client:
                return await _await_via_feed(client, feed, "pid", time.monotonic() - 1, None, None)

        with pytest.raises(HTTPException) as ei:
            asyncio.run(scenario())
        assert ei.value.status_code == 504

    def test_cancel_raises_cancelled_error(self) -> None:
        async def scenario() -> dict[str, Any] | None:
            feed: asyncio.Queue = asyncio.Queue()
            cancel = asyncio.Event()
            cancel.set()
            async with httpx.AsyncClient() as client:
                return await _await_via_feed(client, feed, "pid", time.monotonic() + 5, None, cancel)

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(scenario())


class _FakeWS:
    """假 ws 连接：按序吐出 messages，耗尽后抛 ConnectionError 模拟断线。"""

    def __init__(self, messages: list[str]) -> None:
        self._messages = messages

    def __aiter__(self) -> "_FakeWS":
        return self

    async def __anext__(self) -> str:
        if self._messages:
            return self._messages.pop(0)
        raise ConnectionError("fake ws dropped")


class TestComfyUIProgress:
    def test_dispatch_drop_and_reconnect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """连上 -> 吐一条进度 + 一条非 JSON（丢弃）-> 断连 -> 退避后重连。"""
        messages = [
            "not-json-{",  # _dispatch 解析失败：debug 日志后丢弃，不影响后续
            ws_msg("progress", prompt_id="p1", value=1, max=30),
        ]
        connects: list[str] = []

        @asynccontextmanager
        async def fake_connect(url: str, **kwargs: Any) -> AsyncIterator[_FakeWS]:
            connects.append(url)
            yield _FakeWS(list(messages) if len(connects) == 1 else [])

        monkeypatch.setattr(ws_mod.websockets, "connect", fake_connect)

        async def scenario() -> list[dict[str, Any]]:
            progress = ComfyUIProgress()
            queue = await progress.subscribe()
            got: list[dict[str, Any]] = []
            for _ in range(4):
                got.append(await asyncio.wait_for(queue.get(), 5))
            await progress.stop()
            return got

        got = asyncio.run(scenario())
        # 断线退避 1s 后重连，第四条是第二次 _connected
        assert [g["type"] for g in got] == ["_connected", "progress", "_disconnected", "_connected"]
        assert got[1]["data"] == {"prompt_id": "p1", "value": 1, "max": 30}
        assert connects, "url should carry clientId"
        assert "clientId=" in connects[0]

    def test_broadcast_drops_oldest_when_full(self) -> None:
        progress = ComfyUIProgress()
        queue: asyncio.Queue = asyncio.Queue(maxsize=2)
        progress._subs.add(queue)
        for i in range(4):
            progress._broadcast({"type": "progress", "data": {"value": i}})
        drained = [queue.get_nowait()["data"]["value"] for _ in range(queue.qsize())]
        assert drained == [2, 3]  # 封顶后挤掉最老两条，滞后的订阅者拿最新的
        progress._subs.clear()
