"""
POST /anima/generate 业务编排，任务化后分两层：
validate_generate 同步校验（路由层即时 400，含底图/蒙版 b64 预检）；
run_generate 后台逐张执行（count 循环、独立 seed），进度经 TaskHandle 上报——
progress（第 i 张 + 当前步数）与 image_done（落盘登记）。
领域错误直接抛 HTTPException（状态码语义见 README），由任务框架统一转 error 终态。
"""

import asyncio
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
from comfyui_anima_api.config import COMFYUI_CONFIG, GENERATE_DEFAULTS, REFERENCE_STRENGTH_DEFAULTS
from comfyui_anima_api.generation.workflows import build_generate_workflow
from comfyui_anima_api.schemas import GenerateParams
from comfyui_anima_api.tasks import TaskHandle
from comfyui_anima_api.utils import decode_b64_image

logger = logging.getLogger(__name__)


def validate_generate(p: GenerateParams) -> str:
    positive = p.prompt.positive.strip()

    if p.prompt.positive.isspace():
        raise HTTPException(status_code=400, detail="prompt.positive is whitespace-only")

    if not positive:
        raise HTTPException(status_code=400, detail="prompt.positive is required")

    if p.count > 1 and p.seed is not None:
        raise HTTPException(
            status_code=400,
            detail="seed is not supported for batch generation (count>1); remove it and each image "
            "will use an independent random seed",
        )

    if p.base.image is not None:
        decode_b64_image(p.base.image, "base.image")
        if p.base.mask is not None:
            decode_b64_image(p.base.mask, "base.mask")
    else:
        if p.base.denoise is not None:
            raise HTTPException(status_code=400, detail="base.denoise is only valid together with base.image")
        if p.base.mask is not None:
            raise HTTPException(status_code=400, detail="base.mask is only valid together with base.image")

    if p.reference.image is not None:
        decode_b64_image(p.reference.image, "reference.image")
        if p.reference.mode == "inpaint":
            if p.reference.mask is not None:
                decode_b64_image(p.reference.mask, "reference.mask")
            elif p.base.mask is None:
                raise HTTPException(
                    status_code=400,
                    detail="reference.mode 'inpaint' requires reference.mask (or base.mask to fall back on)",
                )
        elif "mask" in p.reference.provided:
            raise HTTPException(
                status_code=400,
                detail=f"reference.mask is only valid with reference.mode 'inpaint' (got '{p.reference.mode}'); "
                "remove it or switch mode",
            )
    else:
        stale = sorted(p.reference.provided - {"image"})
        if stale:
            raise HTTPException(
                status_code=400,
                detail="reference fields have no effect without reference.image; "
                f"remove them or provide the image: {', '.join(f'reference.{k}' for k in stale)}",
            )

    return positive


async def _run_generate_one(
    handle: TaskHandle,
    p: GenerateParams,
    client: httpx.AsyncClient,
    feed: asyncio.Queue,
    *,
    i: int,
    positive: str,
    denoise: float,
    ref_strength: float,
    base_image: str | None,
    base_mask_image: str | None,
    ref_image: str | None,
    ref_mask_image: str | None,
) -> None:
    if handle.cancel_event.is_set():
        raise asyncio.CancelledError

    seed = p.seed if p.seed is not None else random.randrange(2**64)
    image_started = time.monotonic()
    workflow = build_generate_workflow(
        p,
        positive=positive,
        denoise=denoise,
        seed=seed,
        ref_strength=ref_strength,
        base_image=base_image,
        base_mask_image=base_mask_image,
        ref_image=ref_image,
        ref_mask_image=ref_mask_image,
        filename_prefix=f"anima/{handle.task_id}/{i:03d}",
    )
    prompt_id = await submit(client, workflow)
    await handle.set_prompt_id(prompt_id)
    logger.info("generate[%s]: image %d/%d submitted prompt_id=%s", handle.task_id, i, p.count, prompt_id)

    # i 是本函数的形参而非循环变量，回调直接闭包引用即可
    async def on_step(value: int, max_value: int) -> None:
        await handle.emit("progress", {"image": i, "images_total": p.count, "step": value, "steps": max_value})

    entry = await await_prompt(client, feed, prompt_id, p.timeout, on_progress=on_step, cancel=handle.cancel_event)
    validate_history_entry(entry, prompt_id)
    images = collect_output_images(entry)

    if not images:
        raise HTTPException(status_code=500, detail={"error": "execution produced no image", "prompt_id": prompt_id})
    for image in images:
        idx = await handle.add_image(image, prompt_id)
        await handle.emit(
            "image_done",
            {
                "image": idx,
                "images_total": p.count,
                "filename": image["filename"],
                "url": f"/tasks/{handle.task_id}/images/{idx}",
                "seconds": round(time.monotonic() - image_started, 1),
            },
        )


async def run_generate(handle: TaskHandle, p: GenerateParams) -> None:
    positive = p.prompt.positive.strip()

    denoise = (
        1.0
        if p.base.image is None
        else (p.base.denoise if p.base.denoise is not None else GENERATE_DEFAULTS.img2img_denoise)
    )
    # reference.strength 缺省按模式取默认：与 denoise 同款在执行时解析（不进 schema，保持 provided 纯净）
    ref_strength = (
        p.reference.strength
        if p.reference.strength is not None
        else getattr(REFERENCE_STRENGTH_DEFAULTS, f"{p.reference.mode.value}_strength")
    )

    mode = "t2i"
    if p.base.image is not None:
        mode = f"denoise={denoise:.2f}" + (",inpaint" if p.base.mask is not None else "")
    if p.reference.image is not None:
        mode += f",ref={p.reference.mode}"

    logger.info(
        "generate[%s]: positive=%d chars, size=%s, steps=%s, cfg=%s, sampler=%s/%s, mode=%s, loras=%d, count=%d, "
        "teacache=%s, sage=%s",
        handle.task_id,
        len(positive),
        p.size,
        p.advance.steps,
        p.advance.cfg,
        p.advance.sampler,
        p.advance.scheduler,
        mode,
        len(p.loras),
        p.count,
        p.advance.teacache,
        p.advance.sage_attention,
    )

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                COMFYUI_CONFIG.connect_timeout,
                read=p.timeout + COMFYUI_CONFIG.connect_timeout,
                write=p.timeout + COMFYUI_CONFIG.connect_timeout,
            )
        ) as client:
            # 随机文件名避免并发请求互相覆盖 temp/ 里的暂存图
            stamp = f"{time.strftime('%y%m%d_%H%M%S')}_{random.randrange(2**32):08x}"
            base_image = (
                await upload_image(client, f"anima_ref_{stamp}", decode_b64_image(p.base.image, "base.image"))
                if p.base.image
                else None
            )
            base_mask_image = (
                await upload_image(client, f"anima_mask_{stamp}", decode_b64_image(p.base.mask, "base.mask"))
                if p.base.mask
                else None
            )
            ref_image = (
                await upload_image(client, f"anima_cn_{stamp}", decode_b64_image(p.reference.image, "reference.image"))
                if p.reference.image
                else None
            )
            ref_mask_image = (
                await upload_image(
                    client, f"anima_cn_mask_{stamp}", decode_b64_image(p.reference.mask, "reference.mask")
                )
                if p.reference.mask
                else None
            )

            feed = await comfyui_progress.subscribe()

            try:
                for i in range(1, p.count + 1):
                    await _run_generate_one(
                        handle,
                        p,
                        client,
                        feed,
                        i=i,
                        positive=positive,
                        denoise=denoise,
                        ref_strength=ref_strength,
                        base_image=base_image,
                        base_mask_image=base_mask_image,
                        ref_image=ref_image,
                        ref_mask_image=ref_mask_image,
                    )
            finally:
                comfyui_progress.unsubscribe(feed)

    except httpx.HTTPError as e:
        logger.error("generate[%s]: comfyui request failed: %r", handle.task_id, e)
        raise HTTPException(status_code=502, detail=f"cannot reach ComfyUI at {COMFYUI_CONFIG.base_url}") from e
