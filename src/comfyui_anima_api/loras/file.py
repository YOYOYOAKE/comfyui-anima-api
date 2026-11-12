import logging
import os

from comfyui_anima_api.config import CIVITAI_CONFIG

logger = logging.getLogger(__name__)


def list_all() -> list[str]:
    download_dir = CIVITAI_CONFIG.loras_download_dir
    try:
        names = os.listdir(download_dir)
    except FileNotFoundError:
        names = []

    items: list[tuple[float, str]] = []
    for name in names:
        if not name.endswith(".safetensors"):
            continue
        path = os.path.join(download_dir, name)
        try:
            st = os.stat(path)
        except FileNotFoundError:
            continue
        if not os.path.isfile(path):
            continue
        items.append((st.st_mtime, name))

    items.sort(reverse=True)
    result = [name for _, name in items]
    logger.info("lora/list: %d items", len(result))
    return result
