from typing import Any

from comfyui_anima_api.config import MODEL_CONFIG, UPSCALE_CONFIG


def build_upscale_workflow(image_name: str, filename_prefix: str = "upscale") -> dict:
    workflow: dict[str, dict[str, Any]] = {}
    workflow["img"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}

    workflow["us_model"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": MODEL_CONFIG.sharp}}
    workflow["us"] = {
        "class_type": "ImageUpscaleWithModel",
        "inputs": {"upscale_model": ["us_model", 0], "image": ["img", 0]},
    }
    workflow["us_half"] = {
        "class_type": "ImageScaleBy",
        "inputs": {
            "image": ["us", 0],
            "upscale_method": "lanczos",
            "scale_by": UPSCALE_CONFIG.sharp.upscale_by / 4,
        },
    }

    workflow["save"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["us_half", 0], "filename_prefix": filename_prefix},
    }
    return workflow
