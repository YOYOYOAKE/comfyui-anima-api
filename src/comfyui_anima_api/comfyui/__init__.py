"""
ComfyUI 接入层：HTTP（提交/上传/历史/取图）+ WebSocket（步级进度）+ 等待编排 + 精确取消。
门面（facade）：消费方一律 `from comfyui_anima_api.comfyui import X`，内部布局不影响外部导入路径。

子模块按职责划分：

- ``_shared`` —— _json（非 JSON 响应转 502）与 _timeout（504）小件；
- ``ws``      —— /ws 单连接管理（ComfyUIProgress + comfyui_progress 单例 + ProgressCallback）；
- ``http``    —— 无状态单次请求：submit / upload_image / fetch_history_entry / fetch_image_bytes；
- ``wait``    —— 等待编排：await_prompt（feed 优先、轮询兜底）、entry 校验与图片清单收集；
- ``cancel``  —— cancel_prompt（执行中 interrupt / 排队中出队）。
"""

from comfyui_anima_api.comfyui.cancel import cancel_prompt
from comfyui_anima_api.comfyui.http import fetch_history_entry, fetch_image_bytes, submit, upload_image
from comfyui_anima_api.comfyui.wait import (
    await_prompt,
    collect_output_images,
    validate_history_entry,
    wait_for_history,
)
from comfyui_anima_api.comfyui.ws import ComfyUIProgress, ProgressCallback, comfyui_progress

__all__ = [
    "ComfyUIProgress",
    "ProgressCallback",
    "await_prompt",
    "cancel_prompt",
    "collect_output_images",
    "comfyui_progress",
    "fetch_history_entry",
    "fetch_image_bytes",
    "submit",
    "upload_image",
    "validate_history_entry",
    "wait_for_history",
]
