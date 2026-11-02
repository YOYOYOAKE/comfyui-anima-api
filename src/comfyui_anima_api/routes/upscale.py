from fastapi import APIRouter
from fastapi.responses import JSONResponse

from comfyui_anima_api.schemas import UpscaleParams
from comfyui_anima_api.tasks import create_task
from comfyui_anima_api.upscale import validate_upscale

router = APIRouter()


@router.post("/upscale")
async def upscale(p: UpscaleParams) -> JSONResponse:
    """
    新建超分任务：底图经 4x-AnimeSharp 放大 2 倍（纯图像链，无扩散重绘）

    返回 202 与 task_id；进度与产物同 generate：`GET /tasks/{task_id}/events`、
    `GET /tasks/{task_id}/images.zip`。timeout 为执行秒数上限，超时任务转 error（504）。
    """
    validate_upscale(p)
    task_id = await create_task("upscale", p, images_total=1)
    return JSONResponse({"task_id": task_id}, status_code=202)
