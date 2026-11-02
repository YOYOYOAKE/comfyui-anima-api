"""精确取消：按 prompt_id 撤销 ComfyUI 侧的执行或排队。"""

from typing import Any

import httpx

from comfyui_anima_api.comfyui._shared import _json
from comfyui_anima_api.config import COMFYUI_CONFIG


def _queue_prompt_ids(entries: Any) -> set[str]:
    """
    提取 /queue 条目里的 prompt_id，兼容三种形态：现行数组 [number, prompt_id, ...]、
    老版数组 [prompt_id, ...]、对象 {"prompt_id": ...}。前两种按"前两个元素里的字符串"取。
    """
    ids: set[str] = set()
    for entry in entries or []:
        if isinstance(entry, dict):
            pid = entry.get("prompt_id")
        elif isinstance(entry, (list, tuple)):
            pid = next((e for e in entry[:2] if isinstance(e, str)), None)
        else:
            pid = None
        if pid:
            ids.add(str(pid))
    return ids


async def cancel_prompt(client: httpx.AsyncClient, prompt_id: str) -> None:
    """
    按 prompt 精确取消：执行中 POST /interrupt（ComfyUI 的中断是全局的，先确认
    正在跑的就是它才动手）；还在排队 POST /queue delete（只删这一条）；已不在队
    则不动作（已完成或已消失）。
    """
    r = await client.get(f"{COMFYUI_CONFIG.base_url}/queue")
    queue = _json(r, "response to /queue")
    if prompt_id in _queue_prompt_ids(queue.get("queue_running")):
        await client.post(f"{COMFYUI_CONFIG.base_url}/interrupt")
    elif prompt_id in _queue_prompt_ids(queue.get("queue_pending")):
        await client.post(f"{COMFYUI_CONFIG.base_url}/queue", json={"delete": [prompt_id]})
