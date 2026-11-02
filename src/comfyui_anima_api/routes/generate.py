from fastapi import APIRouter
from fastapi.responses import JSONResponse

from comfyui_anima_api.generation import validate_generate
from comfyui_anima_api.schemas import GenerateParams
from comfyui_anima_api.tasks import create_task

router = APIRouter()


@router.post("/anima/generate")
async def generate(p: GenerateParams) -> JSONResponse:
    """
    新建图像生成任务（文生图 / 图生图 / 局部重绘 / LLLite ControlNet 参考）

    返回 202 与 task_id。任务状态流转 queued → running → done | error | cancelled；
    进度走 `GET /tasks/{task_id}/events`，产物打包 `GET /tasks/{task_id}/images.zip`，
    取消 `POST /tasks/{task_id}/cancel`。

    字段约束与逐项语义见请求 schema（约束即声明：count 上限、seed 域 [0, 2^64-1]、
    strength 值域等）。模式级规则：count>1 时不可指定 seed；base.denoise / base.mask
    仅随 base.image 有效；reference 组内任何字段没有 reference.image 都无意义；
    reference.mask 仅 inpaint 模式有效（缺省回退 base.mask）。timeout 为单张图执行
    秒数上限，超时任务转 error（http_status 504）。
    """
    validate_generate(p)
    task_id = await create_task("generate", p, images_total=p.count)
    return JSONResponse({"task_id": task_id}, status_code=202)
