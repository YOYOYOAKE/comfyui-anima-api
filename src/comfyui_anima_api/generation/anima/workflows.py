from typing import Any

from comfyui_anima_api.config import (
    ANIMA_BOOSTER_CONFIG,
    MODEL_CONFIG,
    SIZE_PRESETS,
    TEACACHE_CONFIG,
)
from comfyui_anima_api.schemas import GenerateParams


def build_generate_workflow(
    p: GenerateParams,
    *,
    positive: str,
    denoise: float,
    seed: int | None,
    ref_strength: float,
    base_image: str | None,
    base_mask_image: str | None,
    ref_image: str | None,
    ref_mask_image: str | None,
    filename_prefix: str = "anima",
) -> dict:
    """
    完整生图工作流组装

    - anima_*：工作流骨架节点
    - teacache：TeaCache 节点
    - base_img_*：底图节点
    - ref_img_*：参考图节点（ControlNet）
    - lora_*：LoRA 节点
    """

    workflow: dict[str, dict[str, Any]] = {}

    workflow["anima_booster"] = {
        "class_type": "AnimaBoosterLoader",
        "inputs": {
            "model_name": MODEL_CONFIG.anima_base.anima,
            "sage_attention": "auto" if p.advance.sage_attention else "disabled",
            "torch_compile": ANIMA_BOOSTER_CONFIG.torch_compile,
        },
    }

    # 链尾指针，供每个可选节点把自己挂上链并改写指针
    last_node = ["anima_booster", 0]

    if p.advance.teacache and p.advance.steps >= TEACACHE_CONFIG.min_steps:
        tc_inputs = TEACACHE_CONFIG.model_dump()
        tc_inputs.pop("min_steps")
        # 配置字段 version（env TEACACHE_VERSION）-> ComfyUI 节点输入名 teacache_version
        tc_inputs["teacache_version"] = tc_inputs.pop("version")
        workflow["teacache"] = {"class_type": "AnimaTeaCache", "inputs": {"model": last_node, **tc_inputs}}
        last_node = ["teacache", 0]

    for i, spec in enumerate(p.loras):
        workflow[f"lora_{i}"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": last_node, "lora_name": spec.name, "strength_model": spec.strength},
        }
        last_node = [f"lora_{i}", 0]

    # LLLite ControlNet
    if ref_image is not None:
        workflow["ref_img"] = {"class_type": "LoadImage", "inputs": {"image": ref_image}}
        cn_inputs = {
            "model": last_node,
            "lllite_name": getattr(MODEL_CONFIG.control_net, p.reference.mode.value),
            "image": ["ref_img", 0],
            "strength": ref_strength,
            "start_percent": 0.0,
            "end_percent": 1.0,
            "auto_remap": True,
            "manifest": "Auto (Recommended)",
            "extend_to_new_layers": False,
            "extend_strength": 0.5,
            "preserve_wrapper": p.reference.mode == "inpaint",
        }
        # inpaint 需要蒙版，缺独立蒙版时回退复用底图蒙版节点
        if p.reference.mode == "inpaint":
            if ref_mask_image is not None:
                workflow["ref_img_mask"] = {
                    "class_type": "LoadImageMask",
                    "inputs": {"image": ref_mask_image, "channel": "red"},
                }
                cn_inputs["mask"] = ["ref_img_mask", 0]
            else:
                cn_inputs["mask"] = ["base_img_mask", 0]

        workflow["ref_img_apply"] = {"class_type": "AnimaLLLiteRemapApply", "inputs": cn_inputs}
        last_node = ["ref_img_apply", 0]

    # ---- 提示词条件：CLIP -> 正/负编码（提示词零注入，原样进编码器）----
    workflow["anima_clip"] = {
        "class_type": "CLIPLoader",
        "inputs": {"clip_name": MODEL_CONFIG.anima_base.text_encoder, "type": "qwen_image", "device": "default"},
    }
    workflow["anima_pos"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["anima_clip", 0], "text": positive}}
    workflow["anima_neg"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["anima_clip", 0], "text": p.prompt.negative},
    }

    # ---- latent：底图编码（图生图，inpaint 叠噪声蒙版）或空 latent（文生图）----
    width, height = SIZE_PRESETS[p.size]
    workflow["anima_vae"] = {"class_type": "VAELoader", "inputs": {"vae_name": MODEL_CONFIG.anima_base.vae}}
    if base_image is None:
        workflow["anima_latent"] = {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        }
        latent = ["anima_latent", 0]
    else:
        workflow["base_img"] = {"class_type": "LoadImage", "inputs": {"image": base_image}}
        workflow["base_img_scale"] = {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["base_img", 0],
                "upscale_method": "lanczos",
                "width": width,
                "height": height,
                "crop": "disabled",
            },
        }
        workflow["base_img_encode"] = {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["base_img_scale", 0], "vae": ["anima_vae", 0]},
        }
        latent = ["base_img_encode", 0]
        if base_mask_image is not None:
            # LoadImageMask 内部统一转 RGB 后取指定通道；channel=red 时灰度蒙版（灰度值即红通道）与彩色蒙版都兼容。
            # SetLatentNoiseMask：采样时 mask 外区域每步拉回原图 latent；mask 尺寸由采样器自动插值到 latent
            workflow["base_img_mask"] = {
                "class_type": "LoadImageMask",
                "inputs": {"image": base_mask_image, "channel": "red"},
            }
            workflow["base_img_inpaint"] = {
                "class_type": "SetLatentNoiseMask",
                "inputs": {"samples": ["base_img_encode", 0], "mask": ["base_img_mask", 0]},
            }
            latent = ["base_img_inpaint", 0]

    # ---- 采样 -> 解码 -> 落盘（SaveImage 持久落 output/，事后可经 /view 取回）----
    workflow["anima_sampler"] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": seed,
            "steps": p.advance.steps,
            "cfg": p.advance.cfg,
            "sampler_name": p.advance.sampler,
            "scheduler": p.advance.scheduler,
            "denoise": denoise,
            "model": last_node,
            "positive": ["anima_pos", 0],
            "negative": ["anima_neg", 0],
            "latent_image": latent,
        },
    }
    workflow["anima_decode"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["anima_sampler", 0], "vae": ["anima_vae", 0]},
    }
    workflow["anima_save"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["anima_decode", 0], "filename_prefix": filename_prefix},
    }
    return workflow
