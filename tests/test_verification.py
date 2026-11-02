"""verification.py 单测：safetensors 头解析、block 布局判断、下载产物校验门禁、文件名正则。"""

import json
import os
import struct
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException

from comfyui_anima_api.loras.verification import (
    _get_safetensors_max_block,
    _sha256,
    verify_lora_file,
)
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
        assert _get_safetensors_max_block(str(p)) == 27

    def test_underscore_variant_and_llm_adapter_excluded(self, tmp_path: Path) -> None:
        # blocks_9 变体也要识别；llm_adapter（text encoder 侧）不计入 transformer 布局判断
        p = tmp_path / "x.safetensors"
        write_safetensors(p, {"lora_transformer.blocks_9.attn": {}, "lora_llm_adapter.blocks.99.proj": {}})
        assert _get_safetensors_max_block(str(p)) == 9

    def test_metadata_and_no_blocks_return_none(self, tmp_path: Path) -> None:
        # __metadata__ 不是块 key；无任何 transformer block key 视为非 LoRA 布局
        p = tmp_path / "x.safetensors"
        write_safetensors(p, {"unrelated.key": {}, "__metadata__": {"ss": "1"}})
        assert _get_safetensors_max_block(str(p)) is None


class TestVerifyLoraFile:
    def _file(self, tmp_path: Path, header: dict[str, Any]) -> str:
        p = tmp_path / "l.safetensors"
        write_safetensors(p, header)
        return str(p)

    def test_sha_mismatch_discards_file(self, tmp_path: Path) -> None:
        p = self._file(tmp_path, {"lora_transformer.blocks.0.a": {}})
        with pytest.raises(HTTPException) as ei:
            verify_lora_file(p, "deadbeef")
        assert ei.value.status_code == 502
        assert not os.path.exists(p)

    def test_sha_match_or_absent_passes(self, tmp_path: Path) -> None:
        p = self._file(tmp_path, {"lora_transformer.blocks.3.a": {}})
        assert verify_lora_file(p, _sha256(p)) is True
        assert verify_lora_file(p, None) is True
        assert os.path.exists(p)

    def test_no_block_keys_rejected(self, tmp_path: Path) -> None:
        p = self._file(tmp_path, {"unrelated.key": {}})
        with pytest.raises(HTTPException) as ei:
            verify_lora_file(p, None)
        assert ei.value.status_code == 400
        assert not os.path.exists(p)

    def test_beyond_2b_layout_rejected(self, tmp_path: Path) -> None:
        p = self._file(tmp_path, {"lora_transformer.blocks.39.a": {}})
        with pytest.raises(HTTPException) as ei:
            verify_lora_file(p, None)
        assert ei.value.status_code == 400
        assert not os.path.exists(p)
