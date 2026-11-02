"""
POST /upscale 业务编排，任务化后分两层：
validate_upscale 同步校验（路由层即时 400，b64 预检）；
run_upscale 后台执行（上传 -> 提交 -> 等待 -> 登记），进度经 TaskHandle 上报。
与 generation 完全独立，互不引用；领域错误直接抛 HTTPException。
"""

import logging
import random
import time

import httpx
from fastapi import HTTPException

from comfyui_anima_api.comfyui import (
    await_prompt,
    collect_output_images,
    comfyui_progress,
    submit,
    upload_image,
    validate_history_entry,
)
from comfyui_anima_api.config import COMFYUI_CONFIG
from comfyui_anima_api.schemas import UpscaleParams
from comfyui_anima_api.tasks import TaskHandle
from comfyui_anima_api.upscale.workflows import build_upscale_workflow
from comfyui_anima_api.utils import decode_b64_image

logger = logging.getLogger(__name__)


def validate_upscale(p: UpscaleParams) -> None:
    decode_b64_image(p.image, "image")


async def run_upscale(handle: TaskHandle, p: UpscaleParams) -> None:
    started = time.monotonic()
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                COMFYUI_CONFIG.connect_timeout,
                read=p.timeout + COMFYUI_CONFIG.connect_timeout,
                write=p.timeout + COMFYUI_CONFIG.connect_timeout,
            )
        ) as client:
            stamp = f"{time.strftime('%y%m%d_%H%M%S')}_{random.randrange(2**32):08x}"
            image = await upload_image(client, f"anima_upscale_{stamp}", decode_b64_image(p.image, "image"))
            feed = await comfyui_progress.subscribe()

            try:
                prompt_id = await submit(client, build_upscale_workflow(image, f"upscale/{handle.task_id}/001"))
                await handle.set_prompt_id(prompt_id)
                logger.info("upscale[%s]: submitted prompt_id=%s", handle.task_id, prompt_id)
                entry = await await_prompt(client, feed, prompt_id, p.timeout, cancel=handle.cancel_event)
                validate_history_entry(entry, prompt_id)
                images = collect_output_images(entry)
                if not images:
                    raise HTTPException(
                        status_code=500, detail={"error": "execution produced no image", "prompt_id": prompt_id}
                    )
                for img in images:
                    idx = await handle.add_image(img, prompt_id)
                    await handle.emit(
                        "image_done",
                        {
                            "image": idx,
                            "images_total": 1,
                            "filename": img["filename"],
                            "seconds": round(time.monotonic() - started, 1),
                        },
                    )
            finally:
                comfyui_progress.unsubscribe(feed)

    except httpx.HTTPError as e:
        logger.error("upscale[%s]: comfyui request failed: %r", handle.task_id, e)
        raise HTTPException(status_code=502, detail=f"cannot reach ComfyUI at {COMFYUI_CONFIG.base_url}") from e
