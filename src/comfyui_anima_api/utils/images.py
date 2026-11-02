"""
图片数据解码与格式嗅探：generate / upscale 两个域共用的内容级校验。
按 magic bytes 识别 PNG/JPG/WebP：decode 预检即拒绝其它内容；upload 按嗅探结果定扩展名与 MIME。
"""

import base64

from fastapi import HTTPException


def sniff_image(data: bytes, field: str) -> tuple[str, str]:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png", "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg", "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp", "image/webp"

    raise HTTPException(status_code=400, detail=f"invalid {field}, must be PNG/JPG/WebP image data")


def decode_b64_image(data: str, field: str) -> bytes:
    b64 = data.partition(",")[2] if data.startswith("data:") else data

    try:
        raw = base64.b64decode(b64, validate=True)
    except ValueError:
        raise HTTPException(
            status_code=400, detail=f"invalid {field}, must be base64 (or a data: URL) of PNG/JPG/WebP"
        ) from None

    if not raw:
        raise HTTPException(status_code=400, detail=f"invalid {field}, base64 payload is empty")

    sniff_image(raw, field)

    return raw
