"""
任务事件总线：进度类事件只进内存回放缓冲并扇出——终态后即无用，落库纯属写放大
（30 步 × 8 张 = 240 次写库/任务）；仅终态事件额外落库一条，供 API 重启后的
断线重连回放收尾。seq 是全局单调计数（SSE 事件 id / Last-Event-ID 凭它续收），
起始值取库中最大 seq，跨重启仍大于旧客户端手里的断点。subscribe 在锁内完成
"回放 + 注册"，与 emit 互斥，不丢事件、不重复推送。
"""

import asyncio
import contextlib
import logging
import time
from typing import Any, cast

from comfyui_anima_api.tasks.store import TERMINAL_STATUSES, EventRow, get_store

logger = logging.getLogger(__name__)

# 终态事件类型
TERMINAL_EVENTS = frozenset({"done", "error", "cancelled"})

_QUEUE_MAX = 1024
# 每任务回放缓冲上限：只服务进程存活期的断线重连，超出挤掉最老（更早断点的
# 重连拿到尾部 + 实时流）。常规负载（30 步 × 8 张 ≈ 250 条）远碰不到
_REPLAY_MAX = 4096


class _TaskState:
    """单任务的内存态：回放缓冲 + 终态标记；终态且无人订阅即整体回收。"""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.terminal = False


class _TaskBus:
    def __init__(self) -> None:
        self._states: dict[str, _TaskState] = {}
        self._subs: dict[str, list[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()
        self._seq: int | None = None  # 懒初始化：首个事件到来时取库中最大 seq

    @staticmethod
    def _put(queue: asyncio.Queue, event: dict[str, Any]) -> None:
        try:
            queue.put_nowait(dict(event))
        except asyncio.QueueFull:
            with contextlib.suppress(asyncio.QueueEmpty, ValueError):
                queue.get_nowait()
            queue.put_nowait(dict(event))

    async def _next_seq(self) -> int:
        if self._seq is None:
            self._seq = await get_store().max_event_seq()
        self._seq += 1
        return self._seq

    async def emit(self, task_id: str, event_type: str, data: dict[str, Any]) -> int:
        """进度类事件：内存 append + 扇出，零写库；终态事件：额外落库一条（尽力而为，
        失败只影响重启后的回放收尾，不影响本次投递）。返回事件 seq。"""
        async with self._lock:
            event: dict[str, Any] = {
                "seq": await self._next_seq(), "type": event_type, "data": data, "ts": time.time()
            }
            state = self._states.setdefault(task_id, _TaskState())
            state.events.append(event)
            if len(state.events) > _REPLAY_MAX:
                del state.events[: len(state.events) - _REPLAY_MAX]
            for queue in self._subs.get(task_id, []):
                self._put(queue, event)
            if event_type in TERMINAL_EVENTS:
                state.terminal = True
                try:
                    await get_store().insert_event(task_id, event["seq"], event_type, data)
                except Exception:
                    logger.warning("bus: persist terminal event for %s failed", task_id, exc_info=True)
                self._gc(task_id)
        return event["seq"]

    async def subscribe(self, task_id: str, after_seq: int = 0) -> tuple[list[EventRow], asyncio.Queue, bool]:
        """
        返回（after_seq 之后的历史事件, 实时队列, 任务是否已终态）。回放与注册
        在同一把锁下完成，中间不可能插入新事件，回放与实时无缝衔接。terminal 供
        订阅方处理"回放不含终态但任务已终态"的情形——即 after_seq 已越过终态事件
        （EventSource 在服务端正常关流后仍会自动重连一次），此时应立即收尾而非挂着等事件。
        本进程有内存态则回放走内存；没有（重启后的旧任务）回落库里仅存的终态行。
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_MAX)
        replay: list[EventRow]
        async with self._lock:
            state = self._states.get(task_id)
            if state is not None:
                # 内存事件与 EventRow 同构（seq/type/data/ts），cast 只过 TypedDict 形参这一关
                replay = cast(list[EventRow], [e for e in state.events if e["seq"] > after_seq])
                terminal = state.terminal
            else:
                replay = await get_store().events_after(task_id, after_seq)
                task = await get_store().get_task(task_id)
                terminal = task is not None and task["status"] in TERMINAL_STATUSES
            self._subs.setdefault(task_id, []).append(queue)
        return replay, queue, terminal

    def unsubscribe(self, task_id: str, queue: asyncio.Queue) -> None:
        # 同步且无 await 点 = 相对事件循环原子，不必抢异步锁
        subs = self._subs.get(task_id)
        if subs is not None and queue in subs:
            subs.remove(queue)
        self._gc(task_id)

    def _gc(self, task_id: str) -> None:
        state = self._states.get(task_id)
        if state is not None and state.terminal and not self._subs.get(task_id):
            self._states.pop(task_id, None)
            self._subs.pop(task_id, None)


bus = _TaskBus()
