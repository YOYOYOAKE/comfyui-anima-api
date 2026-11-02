"""
任务框架：store.py（SQLite 持久化）-> bus.py（事件总线）-> sse.py（SSE 输出）->
service.py（创建/执行/取消/重启对账）。handle.py 是 runner 视角的接口协议；
各域 runner 由 service 在运行时解析（_resolve_runner），任务框架与各域保持无环。
bus 实例刻意不经此门面导出，避免遮蔽 tasks.bus 子模块。
"""

from comfyui_anima_api.tasks.handle import TaskHandle
from comfyui_anima_api.tasks.service import cancel_task, create_task, reconcile
from comfyui_anima_api.tasks.sse import sse_response
from comfyui_anima_api.tasks.store import TERMINAL_STATUSES, get_store

__all__ = [
    "TERMINAL_STATUSES",
    "TaskHandle",
    "cancel_task",
    "create_task",
    "get_store",
    "reconcile",
    "sse_response",
]
