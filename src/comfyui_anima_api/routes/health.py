import logging

import httpx
from fastapi import APIRouter, HTTPException

from comfyui_anima_api.config import COMFYUI_CONFIG

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    """健康检查：当且仅当 API 服务与 ComfyUI 同时可用时返回 200；ComfyUI 不可达或异常返回 503。"""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(COMFYUI_CONFIG.health_timeout)) as client:
            r = await client.get(f"{COMFYUI_CONFIG.base_url}/system_stats")
            r.raise_for_status()
    except httpx.HTTPError as e:
        logger.warning("health: comfyui probe failed: %r", e)
        raise HTTPException(status_code=503, detail=f"comfyui unhealthy at {COMFYUI_CONFIG.base_url}: {e!r}") from e

    return {"status": "ok", "comfyui": "ok"}
