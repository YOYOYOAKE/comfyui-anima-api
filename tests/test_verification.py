"""verification.py 单测：HTML 页/大小对账、SHA256、safetensors 布局与文件名正则。"""

import json
import os
import struct
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException

import comfyui_anima_api.loras.verification as verification
from comfyui_anima_api import loras
from comfyui_anima_api.schemas import LORA_FILENAME_RE as LORA_NAME_RE


def write_safetensors(path: Path, header: dict[str, Any]) -> None:
    """构造最小 safetensors 文件：8 字节小端头长度 + JSON 头（无实际权重数据）。"""
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)


class TestLoraNameRE:
    def test_valid_names(self) -> None:
        for name in ["a.safetensors", "my lora v2(final)[fix].safetensors", "X-1.5.safetensors"]:
            assert LORA_NAME_RE.fullmatch(name)

    def test_rejects_path_traversal_and_empty_stem(self) -> None:
        assert not LORA_NAME_RE.fullmatch("../evil.safetensors")
        assert not LORA_NAME_RE.fullmatch("a/b.safetensors")
        assert not LORA_NAME_RE.fullmatch("a\\b.safetensors")
        assert not LORA_NAME_RE.fullmatch(".safetensors")
        assert not LORA_NAME_RE.fullmatch("model.bin")
        assert not LORA_NAME_RE.fullmatch("model.safetensors.txt")


class TestGetSafetensorsMaxBlock:
    def test_picks_largest_block(self, tmp_path: Path) -> None:
        p = tmp_path / "x.safetensors"
        write_safetensors(
            p,
            {
                "lora_transformer.blocks.0.attn.q": {},
                "lora_transformer.blocks.27.attn.v": {},
                "lora_transformer.blocks.5.mlp": {},
            },
        )
        assert verification._get_safetensors_max_block(str(p)) == 27

    def test_underscore_variant_and_llm_adapter_excluded(self, tmp_path: Path) -> None:
        # blocks_9 变体也要识别；llm_adapter（text encoder 侧）不计入 transformer 布局判断
        p = tmp_path / "x.safetensors"
        write_safetensors(p, {"lora_transformer.blocks_9.attn": {}, "lora_llm_adapter.blocks.99.proj": {}})
        assert verification._get_safetensors_max_block(str(p)) == 9

    def test_metadata_and_no_blocks_return_none(self, tmp_path: Path) -> None:
        # __metadata__ 不是块 key；无任何 transformer block key 视为非 LoRA 布局
        p = tmp_path / "x.safetensors"
        write_safetensors(p, {"unrelated.key": {}, "__metadata__": {"ss": "1"}})
        assert verification._get_safetensors_max_block(str(p)) is None


class TestVerifyDownloadedFile:
    def _file(self, tmp_path: Path, header: dict[str, Any]) -> str:
        p = tmp_path / "l.safetensors"
        write_safetensors(p, header)
        return str(p)

    def test_html_page_rejected_and_discarded(self, tmp_path: Path) -> None:
        # HTTP 200 + HTML 错误体（反代兜底页 / CF 挑战页 / 鉴权页等）：aria2 会当文件存下
        p = tmp_path / "l.safetensors"
        p.write_bytes(b"<!doctype html><html>frontend</html>")
        with pytest.raises(HTTPException) as ei:
            loras.verify(str(p), None, None)
        assert ei.value.status_code == 502
        # 文案直连场景优先（Cloudflare 质询在前），镜像/代理转发问题在后——一般情况优先
        assert "HTML page" in ei.value.detail and "Cloudflare" in ei.value.detail
        assert not p.exists()

    def test_size_mismatch_rejected_and_discarded(self, tmp_path: Path) -> None:
        # 元数据声明远大于实得大小：对账即拒（容忍 2%/1KB 的 sizeKB 舍入）
        p = self._file(tmp_path, {"lora_transformer.blocks.0.a": {}})
        with pytest.raises(HTTPException) as ei:
            loras.verify(p, None, 44914 * 1024)
        assert ei.value.status_code == 502
        assert "metadata says" in ei.value.detail
        assert not os.path.exists(p)

    def test_size_within_tolerance_passes(self, tmp_path: Path) -> None:
        p = self._file(tmp_path, {"lora_transformer.blocks.0.a": {}})
        expected = os.path.getsize(p) + 1000  # 容差 max(2%, 1KB)
        loras.verify(p, None, expected)
        assert os.path.exists(p)

    def test_sha_mismatch_discards_file(self, tmp_path: Path) -> None:
        p = self._file(tmp_path, {"lora_transformer.blocks.0.a": {}})
        with pytest.raises(HTTPException) as ei:
            loras.verify(p, "deadbeef", None)
        assert ei.value.status_code == 502
        assert not os.path.exists(p)

    def test_sha_match_or_absent_passes(self, tmp_path: Path) -> None:
        p = self._file(tmp_path, {"lora_transformer.blocks.3.a": {}})
        loras.verify(p, verification._sha256(p), None)
        loras.verify(p, None, None)
        assert os.path.exists(p)

    def test_no_block_keys_rejected(self, tmp_path: Path) -> None:
        p = self._file(tmp_path, {"unrelated.key": {}})
        with pytest.raises(HTTPException) as ei:
            loras.verify(p, None, None)
        assert ei.value.status_code == 400
        assert not os.path.exists(p)

    def test_beyond_2b_layout_rejected(self, tmp_path: Path) -> None:
        p = self._file(tmp_path, {"lora_transformer.blocks.39.a": {}})
        with pytest.raises(HTTPException) as ei:
            loras.verify(p, None, None)
        assert ei.value.status_code == 400
        assert not os.path.exists(p)
