"""纯放大域（POST /generation/upscale）：service.py 业务编排 -> workflows.py 工作流构建（纯函数）。"""

from comfyui_anima_api.generation.upscale.service import run_upscale, validate_upscale
from comfyui_anima_api.generation.upscale.workflows import build_upscale_workflow

__all__ = ["build_upscale_workflow", "run_upscale", "validate_upscale"]
