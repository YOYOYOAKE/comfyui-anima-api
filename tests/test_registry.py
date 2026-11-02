"""registry.py 单测：loras.json 登记表的原子写、合并与 /lora/list 枚举。"""

import json
import os
from pathlib import Path

import pytest

from comfyui_anima_api.config import CIVITAI_CONFIG
from comfyui_anima_api.loras import registry


def patch_loras_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # 目录唯一真相源是 CIVITAI_CONFIG，registry 调用时现取
    monkeypatch.setattr(CIVITAI_CONFIG, "loras_download_dir", str(tmp_path))


class TestRegistryUpdate:
    def test_write_and_reload(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_loras_paths(monkeypatch, tmp_path)
        registry.update_registry("a.safetensors", 123, "trigger a")
        data = json.loads((tmp_path / "loras.json").read_text(encoding="utf-8"))
        assert data == {"a.safetensors": {"filename": "a.safetensors", "version_id": 123, "trigger": "trigger a"}}

    def test_merges_without_clobbering_existing_entries(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_loras_paths(monkeypatch, tmp_path)
        (tmp_path / "loras.json").write_text(
            json.dumps(
                {"old.safetensors": {"filename": "old.safetensors", "version_id": None, "trigger": "old"}},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        registry.update_registry("new.safetensors", 456, "new trigger")
        data = json.loads((tmp_path / "loras.json").read_text(encoding="utf-8"))
        assert data["old.safetensors"]["trigger"] == "old"
        assert data["new.safetensors"]["version_id"] == 456
        # 原子替换后不残留 .tmp
        assert not (tmp_path / "loras.json.tmp").exists()

    def test_no_tmp_left_after_write(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_loras_paths(monkeypatch, tmp_path)
        registry.update_registry("a.safetensors", None, "t")
        assert sorted(p.name for p in tmp_path.iterdir()) == ["loras.json"]


class TestLoadRegistry:
    def test_missing_file_is_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_loras_paths(monkeypatch, tmp_path)
        assert registry._load_registry() == {}

    def test_corrupted_json_is_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_loras_paths(monkeypatch, tmp_path)
        (tmp_path / "loras.json").write_text("{not json", encoding="utf-8")
        assert registry._load_registry() == {}

    def test_non_object_json_is_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_loras_paths(monkeypatch, tmp_path)
        (tmp_path / "loras.json").write_text("[1, 2]", encoding="utf-8")
        assert registry._load_registry() == {}


class TestListLoras:
    def test_counts_order_and_unregistered_nulls(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_loras_paths(monkeypatch, tmp_path)
        for name, mtime in [("old.safetensors", 1000), ("new.safetensors", 2000)]:
            (tmp_path / name).write_bytes(b"x")
            os.utime(tmp_path / name, (mtime, mtime))
        (tmp_path / "partial.safetensors.part").write_bytes(b"x")  # 下载中的分片不算
        (tmp_path / "notes.txt").write_text("x", encoding="utf-8")  # 非 safetensors 不算
        registry.update_registry("old.safetensors", 7, "old trigger")

        result = registry.list_loras()
        assert result["counts"] == 2
        assert [e["filename"] for e in result["loras"]] == ["new.safetensors", "old.safetensors"]  # mtime 新→旧
        by_name = {e["filename"]: e for e in result["loras"]}
        assert by_name["old.safetensors"]["version_id"] == 7
        assert by_name["new.safetensors"]["version_id"] is None
        assert by_name["new.safetensors"]["trigger"] is None

    def test_missing_dir_is_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_loras_paths(monkeypatch, tmp_path / "nonexistent")
        assert registry.list_loras() == {"counts": 0, "loras": []}
