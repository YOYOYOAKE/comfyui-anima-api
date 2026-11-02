"""ComfyUI 接入层共享小件：响应解析与超时错误的统一出口。"""

import httpx
from fastapi import HTTPException


def _json(r: httpx.Response, what: str) -> dict:
    """解析 ComfyUI 响应体；非 JSON（如反代错误页）统一转 502，而非裸 500。"""
    try:
        return r.json()
    except ValueError as e:
        raise HTTPException(
            status_code=502, detail={"error": f"comfyui returned non-JSON {what}", "status": r.status_code}
        ) from e


def _timeout(prompt_id: str) -> HTTPException:
    return HTTPException(status_code=504, detail={"error": "timed out waiting for completion", "prompt_id": prompt_id})
