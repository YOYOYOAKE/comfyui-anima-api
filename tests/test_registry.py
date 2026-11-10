"""registry.py 单测：loras.json（v1 数组格式）的原子写、合并与 /lora/list 枚举。

v0 -> v1 的启动迁移在 tests/test_migrate.py。
"""

import json
import os
from pathlib import Path

import pytest

from comfyui_anima_api.config import CIVITAI_CONFIG
from comfyui_anima_api.loras import registry


def patch_loras_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # 目录唯一真相源是 CIVITAI_CONFIG，registry 调用时现取
    monkeypatch.setattr(CIVITAI_CONFIG, "loras_download_dir", str(tmp_path))


def read_file(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "loras.json").read_text(encoding="utf-8"))


def write_raw(tmp_path: Path, data: dict) -> None:
    (tmp_path / "loras.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class TestRegistryUpdate:
    def test_write_and_reload(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_loras_paths(monkeypatch, tmp_path)
        registry.update_registry("a.safetensors", 123, "trigger a", type="style", note="n")
        assert read_file(tmp_path) == {
            "version": 1,
            "loras": [
                {
                    "filename": "a.safetensors",
                    "civitai_version_id": 123,
                    "trigger": "trigger a",
                    "type": "style",
                    "note": "n",
                }
            ],
        }

    def test_defaults_type_other_note_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_loras_paths(monkeypatch, tmp_path)
        registry.update_registry("a.safetensors", None, "t")
        entry = read_file(tmp_path)["loras"][0]
        assert entry["type"] == "other"
        assert entry["note"] == ""

    def test_merges_without_clobbering_existing_entries(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_loras_paths(monkeypatch, tmp_path)
        write_raw(
            tmp_path,
            {
                "version": 1,
                "loras": [
                    {
                        "filename": "old.safetensors",
                        "civitai_version_id": None,
                        "trigger": "old",
                        "type": "character",
                        "note": "keep",
                    }
                ],
            },
        )
        registry.update_registry("new.safetensors", 456, "new trigger")
        data = read_file(tmp_path)
        assert [e["filename"] for e in data["loras"]] == ["old.safetensors", "new.safetensors"]  # 追加不动旧条目
        assert data["loras"][0]["trigger"] == "old"
        assert data["loras"][0]["note"] == "keep"
        assert data["loras"][1]["civitai_version_id"] == 456
        # 原子替换后不残留 .tmp
        assert not (tmp_path / "loras.json.tmp").exists()

    def test_same_name_replaced_in_place(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_loras_paths(monkeypatch, tmp_path)
        registry.update_registry("a.safetensors", 1, "t1")
        registry.update_registry("b.safetensors", 2, "t2")
        registry.update_registry("a.safetensors", 11, "t11", type="concept", note="updated")
        data = read_file(tmp_path)
        assert [e["filename"] for e in data["loras"]] == ["a.safetensors", "b.safetensors"]  # 原位覆盖不追加
        assert data["loras"][0] == {
            "filename": "a.safetensors",
            "civitai_version_id": 11,
            "trigger": "t11",
            "type": "concept",
            "note": "updated",
        }

    def test_no_tmp_left_after_write(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_loras_paths(monkeypatch, tmp_path)
        registry.update_registry("a.safetensors", None, "t")
        assert sorted(p.name for p in tmp_path.iterdir()) == ["loras.json"]


class TestLoadRegistry:
    def test_missing_file_is_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_loras_paths(monkeypatch, tmp_path)
        assert registry._load_registry() == []

    def test_corrupted_json_is_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_loras_paths(monkeypatch, tmp_path)
        (tmp_path / "loras.json").write_text("{not json", encoding="utf-8")
        assert registry._load_registry() == []

    def test_non_object_json_is_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_loras_paths(monkeypatch, tmp_path)
        (tmp_path / "loras.json").write_text("[1, 2]", encoding="utf-8")
        assert registry._load_registry() == []

    def test_v1_bad_entries_skipped_and_fields_normalized(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_loras_paths(monkeypatch, tmp_path)
        write_raw(
            tmp_path,
            {
                "version": 1,
                "loras": [
                    "not a dict",
                    {"trigger": "no filename"},
                    {"filename": "ok.safetensors"},  # 缺字段按 None 处理
                ],
            },
        )
        assert registry._load_registry() == [
            {
                "filename": "ok.safetensors",
                "civitai_version_id": None,
                "trigger": None,
                "type": None,
                "note": None,
            }
        ]

    def test_newer_version_ignored_and_never_overwritten(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """未来版本：当空表处理，且读路径绝不落盘覆写。"""
        patch_loras_paths(monkeypatch, tmp_path)
        raw = {"version": 99, "loras": [{"filename": "x.safetensors"}]}
        write_raw(tmp_path, raw)
        assert registry._load_registry() == []
        registry.list_loras()
        assert read_file(tmp_path) == raw  # 原样保留，等支持它的版本来读

    def test_v1_missing_loras_list_is_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_loras_paths(monkeypatch, tmp_path)
        write_raw(tmp_path, {"version": 1})
        assert registry._load_registry() == []


class TestListLoras:
    def test_counts_order_and_unregistered_nulls(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_loras_paths(monkeypatch, tmp_path)
        for name, mtime in [("old.safetensors", 1000), ("new.safetensors", 2000)]:
            (tmp_path / name).write_bytes(b"x")
            os.utime(tmp_path / name, (mtime, mtime))
        (tmp_path / "partial.safetensors.part").write_bytes(b"x")  # 下载中的分片不算
        (tmp_path / "notes.txt").write_text("x", encoding="utf-8")  # 非 safetensors 不算
        registry.update_registry("old.safetensors", 7, "old trigger", type="style", note="moss 出品")

        result = registry.list_loras()
        assert result["counts"] == 2
        assert [e["filename"] for e in result["loras"]] == ["new.safetensors", "old.safetensors"]  # mtime 新→旧
        by_name = {e["filename"]: e for e in result["loras"]}
        assert by_name["old.safetensors"] == {
            "filename": "old.safetensors",
            "civitai_version_id": 7,
            "trigger": "old trigger",
            "type": "style",
            "note": "moss 出品",
        }
        assert by_name["new.safetensors"]["civitai_version_id"] is None
        assert by_name["new.safetensors"]["trigger"] is None
        assert by_name["new.safetensors"]["type"] is None
        assert by_name["new.safetensors"]["note"] is None

    def test_missing_dir_is_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_loras_paths(monkeypatch, tmp_path / "nonexistent")
        assert registry.list_loras() == {"counts": 0, "loras": []}
