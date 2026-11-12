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

_NOT_MODEL_FILE_HINT = (
    "; the download endpoint did not serve the model file — common causes: "
    "an anti-bot/Cloudflare challenge page, a gated model needing CIVITAI_API_KEY, or a "
    "CIVITAI_BASE_URL/CIVITAI_PROXY mirror not forwarding the 302-to-CDN download redirect"
)


def verify(path: str, expected_sha: str | None, expected_size: int | None) -> None:
    _reject_html_page(path)
    _reject_size_mismatch(path, expected_size)
    _verify_sha(path, expected_sha)
    _verify_layout(path)


def _reject_html_page(path: str) -> None:
    """HTTP 200 + 错误体（CF 挑战页/鉴权页/镜像兜底页）判断"""
    with open(path, "rb") as f:
        looks_like_page = f.read(64).lstrip().startswith(b"<")
    if not looks_like_page:
        return
    os.remove(path)
    raise HTTPException(
        status_code=502,
        detail="download url returned an HTML page instead of the model file" + _NOT_MODEL_FILE_HINT,
    )


def _reject_size_mismatch(path: str, expected_size: int | None) -> None:
    """与元数据中声明的文件大小对比"""
    if expected_size is None:
        return
    size = os.path.getsize(path)
    # 容忍 KB 舍入（2% 或 1KB）
    if abs(size - expected_size) <= max(expected_size * 0.02, 1024):
        return
    os.remove(path)
    raise HTTPException(
        status_code=502,
        detail=f"downloaded {size} bytes but civitai metadata says {expected_size} bytes" + _NOT_MODEL_FILE_HINT,
    )


def _verify_sha(path: str, expected_sha: str | None) -> None:
    if not expected_sha:
        return
    sha_hex = _sha256(path)
    if sha_hex.upper() == expected_sha.upper():
        return
    logger.error("lora/download: sha256 mismatch, expected=%s got=%s", expected_sha, sha_hex.upper())
    os.remove(path)
    raise HTTPException(status_code=502, detail="downloaded file sha256 mismatch, discarded")


def _verify_layout(path: str) -> None:
    # 校验方式：由于 Anima Base 有且仅有 28 块布局，
    # 因此只要出现一个块的块号 >27，就足以证明是其他架构，如 Anima 2.9B / 3.8B
    # 这些模型产出的 LoRA 在 Civitai 上只会被标记为 Anima 底模，无法在下载前排除
    max_idx = _get_safetensors_max_block(path)
    if max_idx is None:
        os.remove(path)
        raise HTTPException(
            status_code=400, detail="file has no transformer block keys, not recognizable as an Anima LoRA"
        )
    if max_idx > 27:
        os.remove(path)
        raise HTTPException(
            status_code=400,
            detail=f"unsupported Anima LoRA layout (max block index {max_idx}, 2.9B/40-block); "
            f"the base model is the official 2B (28-block), only 2B LoRAs are accepted",
        )


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
