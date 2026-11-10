"""loras.json 版本迁移：v0（最初测试版，扁平 dict）-> v1（{version, loras}）。

迁移时机在应用启动时（app.lifespan 调 run()），registry 的读写只认 v1。
"""

import logging

from comfyui_anima_api.loras.registry import _REGISTRY_LOCK, REGISTRY_VERSION, _read_file, _write_registry
from comfyui_anima_api.schemas import LORA_TYPES

logger = logging.getLogger(__name__)


def run() -> None:
    _migrate_from_0_to_1()


def _migrate_from_0_to_1() -> None:
    raw = _read_file()
    # 文件存在但读不到 version 字段即为 v0
    if not isinstance(raw, dict) or "version" in raw:
        return
    entries = []
    for filename, record in raw.items():
        if not isinstance(filename, str) or not isinstance(record, dict):
            continue
        known_type = record.get("type")
        if known_type not in LORA_TYPES:
            # v0 文件没有 type 字段：按文件名前缀粗分类，未命中的归 other
            known_type = next(
                (
                    t
                    for prefix, t in {
                        "character_": "character",
                        "artist_": "style",
                        "style_": "style",
                        "concept_": "concept",
                    }.items()
                    if filename.startswith(prefix)
                ),
                "other",
            )
        entries.append(
            {
                "filename": filename,
                "civitai_version_id": record.get("version_id"),
                "trigger": record.get("trigger"),
                "type": known_type,
                "note": record.get("note", ""),
            }
        )
    with _REGISTRY_LOCK:
        _write_registry(entries)
    logger.info("loras.json migrated v0 -> v%d: %d entries", REGISTRY_VERSION, len(entries))
