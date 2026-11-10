import json
import logging
import os
import threading

from comfyui_anima_api.config import CIVITAI_CONFIG

logger = logging.getLogger(__name__)

# loras.json 全局写锁：下载侧的 per-file 锁保护不了共享登记表的 load -> merge -> write
# （不同文件名并发下载时交错写会丢条目）
_REGISTRY_LOCK = threading.Lock()

REGISTRY_VERSION = 1


def _download_dir() -> str:
    """LoRA 目录唯一真相源是 CIVITAI_CONFIG，调用时现取，不存在需要同步的第二份常量。"""
    return CIVITAI_CONFIG.loras_download_dir


def _loras_json() -> str:
    return os.path.join(_download_dir(), "loras.json")


def _read_file() -> object:
    try:
        with open(_loras_json(), encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return None


def _entries_from_raw(raw: object) -> list[dict]:
    """v1 登记表内容 -> 标准五字段条目列表；坏条目跳过，缺字段按未设置（None）。"""
    if not isinstance(raw, dict):
        return []
    version = raw.get("version")
    if version != REGISTRY_VERSION:
        if version is not None:
            logger.warning(
                "loras.json version=%r is newer than supported v%d; ignoring its content", version, REGISTRY_VERSION
            )
        return []
    raw_entries = raw.get("loras")
    if not isinstance(raw_entries, list):
        return []
    return [
        {
            "filename": e["filename"],
            "civitai_version_id": e.get("civitai_version_id"),
            "trigger": e.get("trigger"),
            "type": e.get("type"),
            "note": e.get("note"),
        }
        for e in raw_entries
        if isinstance(e, dict) and isinstance(e.get("filename"), str)
    ]


def _load_registry() -> list[dict]:
    return _entries_from_raw(_read_file())


def _write_registry(entries: list[dict]) -> None:
    """全量原子落盘（tmp + os.replace）；调用方必须已持有 _REGISTRY_LOCK。"""
    os.makedirs(_download_dir(), exist_ok=True)
    tmp = _loras_json() + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"version": REGISTRY_VERSION, "loras": entries}, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, _loras_json())


def update_registry(
    filename: str, civitai_version_id: int | None, trigger: str, type: str = "other", note: str = ""
) -> None:
    """把下载到的 LoRA 信息记入 <loras 目录>/loras.json（同名条目原位覆盖，其余不动）。"""

    with _REGISTRY_LOCK:
        entries = _entries_from_raw(_read_file())
        entry = {
            "filename": filename,
            "civitai_version_id": civitai_version_id,
            "trigger": trigger,
            "type": type,
            "note": note,
        }
        for i, existing in enumerate(entries):
            if existing["filename"] == filename:
                entries[i] = entry  # 原位覆盖：其余条目与相对顺序不动
                break
        else:
            entries.append(entry)
        _write_registry(entries)


def list_loras() -> dict:
    """列出已落盘的全部 LoRA（仅 *.safetensors）。"""
    try:
        names = os.listdir(_download_dir())
    except FileNotFoundError:
        names = []
    entries = _entries_from_raw(_read_file())
    by_name = {e["filename"]: e for e in entries}
    items = []
    for name in names:
        path = os.path.join(_download_dir(), name)
        if not name.endswith(".safetensors") or not os.path.isfile(path):
            continue
        try:
            st = os.stat(path)
        except FileNotFoundError:
            continue  # listdir 与 stat 之间文件被移走：跳过而非 500
        record = by_name.get(name) or {}
        items.append(
            (
                st.st_mtime,
                name,
                record.get("civitai_version_id"),
                record.get("trigger"),
                record.get("type"),
                record.get("note"),
            )
        )
    items.sort(key=lambda e: (e[0], e[1]), reverse=True)
    loras = [
        {"filename": name, "civitai_version_id": vid, "trigger": trig, "type": typ, "note": note}
        for _, name, vid, trig, typ, note in items
    ]
    logger.info("lora/list: %d items", len(loras))
    return {"counts": len(loras), "loras": loras}
