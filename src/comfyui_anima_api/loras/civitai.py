import logging
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import HTTPException

from comfyui_anima_api.config import CIVITAI_CONFIG

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VersionInfo:
    version_id: int
    base_model: str
    sha256: str | None
    size_bytes: int | None


def build_request_headers() -> dict[str, str]:
    headers = {"User-Agent": CIVITAI_CONFIG.ua, **CIVITAI_CONFIG.extra_headers}
    if CIVITAI_CONFIG.api_key:
        headers["Authorization"] = f"Bearer {CIVITAI_CONFIG.api_key}"
    return headers


def build_download_url(version_id: int) -> str:
    return f"{CIVITAI_CONFIG.base_url}/api/download/models/{version_id}"


def _pick_primary_file(meta: dict[str, Any]) -> dict[str, Any]:
    """version 中存在多个文件时仅选 primary 文件"""
    raw_files = meta.get("files")
    files = [f for f in raw_files if isinstance(f, dict)] if isinstance(raw_files, list) else []
    model_files = [f for f in files if f.get("type") == "Model"]
    candidates = model_files or files or [{}]
    return next((f for f in candidates if f.get("primary")), candidates[0])


async def fetch_version(version_id: int) -> VersionInfo:
    # 拉取 version 元数据
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(CIVITAI_CONFIG.connect_timeout, read=CIVITAI_CONFIG.read_timeout),
            proxy=CIVITAI_CONFIG.proxy,
            headers=build_request_headers(),
            follow_redirects=True,
        ) as client:
            response = await client.get(f"{CIVITAI_CONFIG.base_url}/api/v1/model-versions/{version_id}")
    except httpx.HTTPError as e:
        logger.error("lora/download: civitai request failed: %r", e)
        raise HTTPException(status_code=502, detail="cannot reach civitai") from e

    # version_id 不存在
    if response.status_code == 404:
        raise HTTPException(status_code=404, detail=f"unknown civitai version_id {version_id}")

    # 鉴权问题
    if response.status_code in (401, 403):
        if CIVITAI_CONFIG.api_key:
            hint = "CIVITAI_API_KEY may be invalid"
        else:
            hint = "this resource requires authentication; set CIVITAI_API_KEY on the server and retry"
        raise HTTPException(
            status_code=502, detail=f"civitai rejected the request (HTTP {response.status_code}): {hint}"
        )

    # 其余非 200 视为 Civitai 侧故障
    if response.status_code != 200:
        logger.error("lora/download: metadata HTTP %d", response.status_code)
        raise HTTPException(status_code=502, detail=f"civitai metadata request failed with HTTP {response.status_code}")

    # 解析元数据
    try:
        meta = response.json()
    except ValueError:
        raise HTTPException(status_code=502, detail="civitai returned non-JSON metadata") from None
    if not isinstance(meta, dict):
        raise HTTPException(status_code=502, detail="civitai returned non-object metadata")

    # 判断底模是否为 Anima
    base_model = str(meta.get("baseModel") or "")
    model = meta.get("model")
    logger.info(
        "lora/download: model=%r version=%r baseModel=%r",
        model.get("name") if isinstance(model, dict) else None,
        meta.get("name"),
        base_model,
    )
    if "anima" not in base_model.lower():
        raise HTTPException(
            status_code=400, detail=f"civitai baseModel is '{base_model or 'unknown'}', only Anima LoRAs are supported"
        )

    # 一个 version 可能挂多个文件（full/pruned/fp16 等），默认下载 primary 文件
    primary = _pick_primary_file(meta)

    # 提取 primary 文件的 SHA256 与大小
    size_kb = primary.get("sizeKB")
    hashes = primary.get("hashes")
    size_bytes = int(size_kb * 1024) if isinstance(size_kb, (int, float)) else None
    sha256 = hashes.get("SHA256") if isinstance(hashes, dict) else None
    if not isinstance(sha256, str):
        sha256 = None

    # 返回领域对象
    return VersionInfo(version_id=version_id, base_model=base_model, sha256=sha256, size_bytes=size_bytes)
