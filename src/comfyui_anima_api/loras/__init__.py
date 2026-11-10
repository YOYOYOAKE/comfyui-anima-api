"""LoRA 域：downloader.py（Civitai 元数据 + aria2 下载 + 校验）、registry.py（loras.json 登记/枚举）
与 migrate.py（loras.json 版本迁移，应用启动时经 run() 执行）。"""

from comfyui_anima_api.loras.downloader import ARIA2C_MISSING, aria2c_available, download
from comfyui_anima_api.loras.registry import list_loras

__all__ = ["ARIA2C_MISSING", "aria2c_available", "download", "list_loras"]
