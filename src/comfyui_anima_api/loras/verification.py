import hashlib
import json
import logging
import os
import re
import struct

from fastapi import HTTPException

logger = logging.getLogger(__name__)

# transformer block key 形如 lora_transformer.blocks.0.xxx / blocks_0.xxx 等
BLOCK_RE = re.compile(r"blocks([._])(\d+)(?=[._])")


def _get_safetensors_max_block(path: str) -> int | None:
    """读 safetensors 文件头并取 transformer block 的最大块号。"""

    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        keys = json.loads(f.read(n))
    idx = [int(m.group(2)) for k in keys if "llm_adapter" not in k for m in [BLOCK_RE.search(k)] if m]
    return max(idx) if idx else None


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_lora_file(tmp_path: str, expected_sha: str | None) -> bool:
    """校验下载 LoRA 是否合法：SHA256 一致且为 Anima Base 布局。"""

    sha_hex = _sha256(tmp_path)

    if expected_sha and sha_hex.upper() != expected_sha.upper():
        logger.error("lora/download: sha256 mismatch, expected=%s got=%s", expected_sha, sha_hex.upper())
        os.remove(tmp_path)
        raise HTTPException(status_code=502, detail="downloaded file sha256 mismatch, discarded")

    # 校验方式：由于Anima Base 有且仅有 28 块布局，
    # 因此只要出现一个块的块号 >27，就足以证明是其他架构，如 Anima 2.9B / 3.8B
    # 这些模型产出的 LoRA 在 Civitai 上只会被标记为 Anima 底模，无法在下载前排除
    max_idx = _get_safetensors_max_block(tmp_path)
    if max_idx is None:
        os.remove(tmp_path)
        raise HTTPException(
            status_code=400, detail="file has no transformer block keys, not recognizable as an Anima LoRA"
        )
    if max_idx > 27:
        os.remove(tmp_path)
        raise HTTPException(
            status_code=400,
            detail=f"unsupported Anima LoRA layout (max block index {max_idx}, 2.9B/40-block); "
            f"the base model is the official 2B (28-block), only 2B LoRAs are accepted",
        )
    return True
