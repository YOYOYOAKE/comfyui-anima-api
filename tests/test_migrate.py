"""migrate.py 单测：v0 扁平 dict -> v1 数组的启动迁移与各"不动"分支。"""

import json
from pathlib import Path
from typing import Any

import pytest

from comfyui_anima_api.config import CIVITAI_CONFIG
from comfyui_anima_api.loras import migrate


def patch_loras_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # 目录唯一真相源是 CIVITAI_CONFIG，migrate 经 registry 取用
    monkeypatch.setattr(CIVITAI_CONFIG, "loras_download_dir", str(tmp_path))


def read_file(tmp_path: Path) -> Any:
    return json.loads((tmp_path / "loras.json").read_text(encoding="utf-8"))


def write_raw(tmp_path: Path, data: Any) -> None:
    (tmp_path / "loras.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class TestMigrateFrom0To1:
    def test_v0_flat_dict_migrated_with_prefix_types(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_loras_paths(monkeypatch, tmp_path)
        write_raw(
            tmp_path,
            {
                "character_a.safetensors": {"filename": "character_a.safetensors", "version_id": 7, "trigger": "ta"},
                "artist_finch.safetensors": {
                    "filename": "artist_finch.safetensors",
                    "version_id": None,
                    "trigger": "@f",
                },
                "style_x.safetensors": {"filename": "style_x.safetensors", "version_id": 3094325, "trigger": "sx"},
                "concept_y.safetensors": {"filename": "concept_y.safetensors", "version_id": None, "trigger": "cy"},
                "moss_z.safetensors": {"filename": "moss_z.safetensors", "version_id": None, "trigger": "cz"},
            },
        )
        migrate._migrate_from_0_to_1()
        data = read_file(tmp_path)
        assert data["version"] == 1
        by_name = {e["filename"]: e for e in data["loras"]}
        assert by_name["character_a.safetensors"]["type"] == "character"
        assert by_name["artist_finch.safetensors"]["type"] == "style"
        assert by_name["style_x.safetensors"]["type"] == "style"
        assert by_name["concept_y.safetensors"]["type"] == "concept"
        assert by_name["moss_z.safetensors"]["type"] == "other"  # 前缀未命中
        # version_id -> civitai_version_id、trigger 原样、note 补空串
        assert by_name["character_a.safetensors"] == {
            "filename": "character_a.safetensors",
            "civitai_version_id": 7,
            "trigger": "ta",
            "type": "character",
            "note": "",
        }
        assert by_name["artist_finch.safetensors"]["trigger"] == "@f"

    def test_existing_valid_type_in_v0_record_respected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_loras_paths(monkeypatch, tmp_path)
        write_raw(
            tmp_path,
            {"x.safetensors": {"filename": "x.safetensors", "version_id": None, "trigger": "t", "type": "style"}},
        )
        migrate._migrate_from_0_to_1()
        assert read_file(tmp_path)["loras"][0]["type"] == "style"

    def test_bad_entries_skipped(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_loras_paths(monkeypatch, tmp_path)
        write_raw(tmp_path, {"a.safetensors": {"version_id": 1, "trigger": "t"}, "broken": 42})
        migrate._migrate_from_0_to_1()
        data = read_file(tmp_path)
        assert [e["filename"] for e in data["loras"]] == ["a.safetensors"]

    def test_missing_file_noop(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_loras_paths(monkeypatch, tmp_path)
        migrate._migrate_from_0_to_1()
        assert not (tmp_path / "loras.json").exists()

    def test_bad_json_noop(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """坏 JSON 无法证明是 v0：原样不动。"""
        patch_loras_paths(monkeypatch, tmp_path)
        (tmp_path / "loras.json").write_text("{not json", encoding="utf-8")
        migrate._migrate_from_0_to_1()
        assert (tmp_path / "loras.json").read_text(encoding="utf-8") == "{not json"

    def test_v1_untouched(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_loras_paths(monkeypatch, tmp_path)
        raw = {"version": 1, "loras": [{"filename": "x.safetensors"}]}
        write_raw(tmp_path, raw)
        migrate._migrate_from_0_to_1()
        assert read_file(tmp_path) == raw

    def test_newer_version_untouched(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_loras_paths(monkeypatch, tmp_path)
        raw = {"version": 99, "loras": [{"filename": "x.safetensors"}]}
        write_raw(tmp_path, raw)
        migrate._migrate_from_0_to_1()
        assert read_file(tmp_path) == raw
