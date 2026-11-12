"""生成域门面：anima/（扩散生图，POST /generation/anima）与 upscale/（纯放大，POST /generation/upscale）。

两个子域各自 service.py 业务编排 -> workflows.py 工作流构建（纯函数），彼此完全独立。
"""

from comfyui_anima_api.generation.anima import (
    build_generate_workflow,
    run_generate,
    validate_generate,
)
from comfyui_anima_api.generation.upscale import (
    build_upscale_workflow,
    run_upscale,
    validate_upscale,
)

__all__ = [
    "build_generate_workflow",
    "build_upscale_workflow",
    "run_generate",
    "run_upscale",
    "validate_generate",
    "validate_upscale",
]
