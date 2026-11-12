"""file.py 单测：/lora/list 的目录枚举规则。"""

import os
from pathlib import Path

import pytest

from comfyui_anima_api import loras
from comfyui_anima_api.config import CIVITAI_CONFIG


def patch_loras_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(CIVITAI_CONFIG, "loras_download_dir", str(tmp_path))


class TestListAll:
    def test_missing_dir_is_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_loras_paths(monkeypatch, tmp_path / "nonexistent")
        assert loras.list_all() == []

    def test_empty_dir_is_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_loras_paths(monkeypatch, tmp_path)
        assert loras.list_all() == []

    def test_only_safetensors_files_sorted_newest_first(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_loras_paths(monkeypatch, tmp_path)
        for name, mtime in [("old.safetensors", 1000), ("new.safetensors", 2000)]:
            (tmp_path / name).write_bytes(b"x")
            os.utime(tmp_path / name, (mtime, mtime))
        (tmp_path / "partial.safetensors.part").write_bytes(b"x")  # 下载中的分片不算
        (tmp_path / "notes.txt").write_text("x", encoding="utf-8")  # 非 safetensors 不算
        (tmp_path / "dir.safetensors").mkdir()  # 目录不算

        assert loras.list_all() == ["new.safetensors", "old.safetensors"]
