import json
import logging
import os
import threading

from comfyui_anima_api.config import CIVITAI_CONFIG

logger = logging.getLogger(__name__)

# loras.json 全局写锁：下载侧的 per-file 锁保护不了共享登记表的 load -> merge -> write
# （不同文件名并发下载时交错写会丢条目）
_REGISTRY_LOCK = threading.Lock()


def _download_dir() -> str:
    """LoRA 目录唯一真相源是 CIVITAI_CONFIG，调用时现取，不存在需要同步的第二份常量。"""
    return CIVITAI_CONFIG.loras_download_dir


def _loras_json() -> str:
    return os.path.join(_download_dir(), "loras.json")


def _load_registry() -> dict:
    try:
        with open(_loras_json(), encoding="utf-8") as f:
            registry = json.load(f)
        if not isinstance(registry, dict):
            registry = {}
    except (FileNotFoundError, ValueError):
        registry = {}
    return registry


def update_registry(filename: str, version_id: int | None, trigger: str) -> None:
    """把下载到的 LoRA 信息记入 <loras 目录>/loras.json。"""

    with _REGISTRY_LOCK:
        registry = _load_registry()
        registry[filename] = {"filename": filename, "version_id": version_id, "trigger": trigger}
        os.makedirs(_download_dir(), exist_ok=True)
        tmp = _loras_json() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, _loras_json())


def list_loras() -> dict:
    """列出已落盘的全部 LoRA（仅 *.safetensors）。"""

    try:
        names = os.listdir(_download_dir())
    except FileNotFoundError:
        names = []
    registry = _load_registry()
    entries = []
    for name in names:
        path = os.path.join(_download_dir(), name)
        if not name.endswith(".safetensors") or not os.path.isfile(path):
            continue
        try:
            st = os.stat(path)
        except FileNotFoundError:
            continue  # listdir 与 stat 之间文件被移走：跳过而非 500
        record = registry.get(name) or {}
        entries.append((st.st_mtime, name, record.get("version_id"), record.get("trigger")))
    entries.sort(key=lambda e: (e[0], e[1]), reverse=True)
    items = [{"filename": name, "version_id": vid, "trigger": trig} for _, name, vid, trig in entries]
    logger.info("lora/list: %d items", len(items))
    return {"counts": len(items), "loras": items}
