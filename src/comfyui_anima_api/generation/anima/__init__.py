"""Anima 生图域（POST /generation/anima）：service.py 业务编排 -> workflows.py 工作流构建（纯函数）。"""

from comfyui_anima_api.generation.anima.service import run_generate, validate_generate
from comfyui_anima_api.generation.anima.workflows import build_generate_workflow

__all__ = ["build_generate_workflow", "run_generate", "validate_generate"]
