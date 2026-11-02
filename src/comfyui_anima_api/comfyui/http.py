"""ComfyUI HTTP 请求小件：提交工作流 / 上传图片 / 查历史 / 取图片字节。全部是无状态单次请求；等待编排见 wait.py。"""

import logging
from collections.abc import Mapping
from typing import Any

import httpx
from fastapi import HTTPException

from comfyui_anima_api.comfyui._shared import _json
from comfyui_anima_api.config import COMFYUI_CONFIG
from comfyui_anima_api.utils import sniff_image

logger = logging.getLogger(__name__)


async def submit(client: httpx.AsyncClient, workflow: dict) -> str:
    r = await client.post(
        f"{COMFYUI_CONFIG.base_url}/prompt", json={"prompt": workflow, "client_id": COMFYUI_CONFIG.client_id}
    )
    resp = _json(r, "response to /prompt")
    if r.status_code != 200 or resp.get("node_errors"):
        raise HTTPException(status_code=400, detail={"error": "workflow rejected", "detail": resp})
    prompt_id = resp.get("prompt_id")
    if not prompt_id:
        raise HTTPException(
            status_code=502, detail={"error": "comfyui accepted the workflow but returned no prompt_id", "detail": resp}
        )
    return prompt_id


async def upload_image(client: httpx.AsyncClient, name: str, data: bytes) -> str:
    """
    上传图片到 ComfyUI temp/（容器内，自动清理，不进 bind mount 的 input/），返回 LoadImage 引用名。
    name 传不含扩展名的基名：扩展名与 MIME 按 magic bytes 定（JPG/WebP 不会误贴 png 标签）。
    """
    ext, mime = sniff_image(data, "upload")
    r = await client.post(
        f"{COMFYUI_CONFIG.base_url}/upload/image",
        files={"image": (f"{name}.{ext}", data, mime)},
        data={"type": "temp", "overwrite": "true"},
    )
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail={"error": "upload rejected", "detail": _json(r, "error response")})
    uploaded = _json(r, "response to /upload/image").get("name")
    if not uploaded:
        raise HTTPException(
            status_code=502, detail={"error": "comfyui upload returned no filename", "detail": r.text[:200]}
        )
    logger.info("upload: %d bytes as %s", len(data), uploaded)
    # ComfyUI 标注格式为 "filename [temp]"（方括号前必须有空格）
    return uploaded + " [temp]"


async def fetch_history_entry(client: httpx.AsyncClient, prompt_id: str) -> dict | None:
    """单次查询 /history/<id>（ws 通知完成后取完整 entry 用）。"""
    r = await client.get(f"{COMFYUI_CONFIG.base_url}/history/{prompt_id}")
    return _json(r, "response to /history").get(prompt_id)


async def fetch_image_bytes(client: httpx.AsyncClient, image: Mapping[str, Any]) -> bytes:
    """按 history 里的图片引用（filename/subfolder/type）从 /view 取回字节。"""
    params = {
        "filename": image["filename"],
        "subfolder": image.get("subfolder", ""),
        "type": image.get("type", "output"),
    }
    r = await client.get(f"{COMFYUI_CONFIG.base_url}/view", params=params)
    r.raise_for_status()
    return r.content
