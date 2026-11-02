from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from comfyui_anima_api.loras import ARIA2C_MISSING, aria2c_available, list_loras
from comfyui_anima_api.schemas import DownloadParams
from comfyui_anima_api.tasks import create_task

router = APIRouter()


@router.get("/lora/list")
async def lora_list() -> dict:
    """
    列出已落盘的 LoRA（仅 *.safetensors）

    已下载但未经本 API 登记的 LoRA 拿不到触发词与 version_id（trigger 为空）。
    """
    return list_loras()


@router.post("/lora/download")
async def lora_download(p: DownloadParams) -> JSONResponse:
    """
    新建 Civitai LoRA 下载任务，落盘即对 ComfyUI 可见

    返回 202 与 task_id；下载进度（速度/ETA）走 `GET /tasks/{task_id}/events`。
    仅接受 Anima Base 2B 底模 LoRA（下载前校验 baseModel，下载后校验 SHA256 与
    transformer block 布局，不合法即丢弃并报错）。文件名合法性由请求 schema 保证
    （纯文件名 + .safetensors，同名覆盖）；鉴权凭据只认服务端环境变量
    CIVITAI_API_KEY，请求体传入即未知字段报错。服务端缺 aria2c 时本端点 503。
    """
    if not aria2c_available():
        raise HTTPException(status_code=503, detail=ARIA2C_MISSING)
    task_id = await create_task("lora_download", p)
    return JSONResponse({"task_id": task_id}, status_code=202)
