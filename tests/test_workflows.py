"""generation/workflows.py 单测：生成图 builder 的全分支（各模式节点链）。"""

from typing import Any

from comfyui_anima_api.config import GENERATE_DEFAULTS, REFERENCE_STRENGTH_DEFAULTS
from comfyui_anima_api.generation import build_generate_workflow
from comfyui_anima_api.schemas import GenerateParams


def _ref_strength(p: GenerateParams) -> float:
    """与 generation.service.run_generate 同款解析：显式 strength 优先，缺省按 mode 取默认。"""
    return (
        p.reference.strength
        if p.reference.strength is not None
        else getattr(REFERENCE_STRENGTH_DEFAULTS, f"{p.reference.mode.value}_strength")
    )


def build(**overrides: Any) -> tuple[dict, GenerateParams]:
    """构造最小 t2i 请求并组装工作流；overrides 直接进请求 dict。返回 (工作流, 请求对象)。"""
    body = {"prompt": {"positive": "pos", "negative": "neg"}, **overrides}
    p = GenerateParams.model_validate(body)
    wf = build_generate_workflow(
        p,
        positive="pos",
        denoise=1.0,
        seed=42,
        ref_strength=_ref_strength(p),
        base_image=None,
        base_mask_image=None,
        ref_image=None,
        ref_mask_image=None,
    )
    return wf, p


def build_with_images(
    ref: str | None = None,
    mask: str | None = None,
    cn: str | None = None,
    cn_mask: str | None = None,
    **overrides: Any,
) -> dict:
    body = {"prompt": {"positive": "pos", "negative": "neg"}, **overrides}
    p = GenerateParams.model_validate(body)
    return build_generate_workflow(
        p,
        positive="pos",
        denoise=0.6,
        seed=42,
        ref_strength=_ref_strength(p),
        base_image=ref,
        base_mask_image=mask,
        ref_image=cn,
        ref_mask_image=cn_mask,
    )


class TestTxt2Img:
    def test_default_chain(self) -> None:
        wf, _ = build()
        assert wf["anima_latent"]["class_type"] == "EmptyLatentImage"
        assert (wf["anima_latent"]["inputs"]["width"], wf["anima_latent"]["inputs"]["height"]) == (832, 1216)
        ks = wf["anima_sampler"]["inputs"]
        assert ks["seed"] == 42 and ks["steps"] == GENERATE_DEFAULTS.steps and ks["cfg"] == 4.5
        assert ks["denoise"] == 1.0 and ks["sampler_name"] == "euler" and ks["scheduler"] == "simple"
        # 默认 teacache=True 且 30 步 >= 12 -> model 链头是 TeaCache
        assert ks["model"] == ["teacache", 0]
        assert ks["positive"] == ["anima_pos", 0] and ks["negative"] == ["anima_neg", 0]
        assert wf["anima_pos"]["inputs"]["text"] == "pos" and wf["anima_neg"]["inputs"]["text"] == "neg"
        assert wf["anima_booster"]["inputs"]["sage_attention"] == "auto"
        # 无 LoRA/CN/底图动态节点（us_model 属 upscale 域，防串扰）
        for absent in ("lora_0", "base_img", "ref_img", "ref_img_apply", "us_model"):
            assert absent not in wf

    def test_size_preset_mapping(self) -> None:
        wf, _ = build(size="large_landscape")
        assert wf["anima_latent"]["inputs"]["width"] == 1536 and wf["anima_latent"]["inputs"]["height"] == 1024

    def test_sampler_override(self) -> None:
        wf, _ = build(advance={"sampler": "res_multistep", "scheduler": "karras", "steps": 45, "cfg": 5.5})
        ks = wf["anima_sampler"]["inputs"]
        assert ks["sampler_name"] == "res_multistep" and ks["scheduler"] == "karras"
        assert ks["steps"] == 45 and ks["cfg"] == 5.5

    def test_sage_attention_disabled(self) -> None:
        wf, _ = build(advance={"sage_attention": False})
        assert wf["anima_booster"]["inputs"]["sage_attention"] == "disabled"


class TestTeaCache:
    def test_attached_when_enabled_and_enough_steps(self) -> None:
        wf, _ = build(
            advance={"steps": 30},
        )
        assert "teacache" in wf and wf["anima_sampler"]["inputs"]["model"] == ["teacache", 0]

    def test_skipped_on_short_sampling(self) -> None:
        wf, _ = build(advance={"steps": 10})
        assert "teacache" not in wf and wf["anima_sampler"]["inputs"]["model"] == ["anima_booster", 0]

    def test_skipped_when_disabled(self) -> None:
        wf, _ = build(advance={"teacache": False})
        assert "teacache" not in wf


class TestLoraChain:
    def test_specs_chain_in_order(self) -> None:
        # teacache 默认开启 -> 链：teacache -> lora_0 -> lora_1 -> KSampler
        wf, _ = build(loras=[{"name": "a.safetensors"}, {"name": "b.safetensors", "strength": 0.5}])
        assert wf["lora_0"]["inputs"] == {
            "model": ["teacache", 0],
            "lora_name": "a.safetensors",
            "strength_model": 1.0,
        }
        assert wf["lora_1"]["inputs"]["model"] == ["lora_0", 0]
        assert wf["lora_1"]["inputs"]["strength_model"] == 0.5
        assert wf["anima_sampler"]["inputs"]["model"] == ["lora_1", 0]

    def test_chain_starts_from_booster_without_teacache(self) -> None:
        wf, _ = build(loras=[{"name": "a.safetensors"}], advance={"teacache": False})
        assert wf["lora_0"]["inputs"]["model"] == ["anima_booster", 0]
        assert wf["anima_sampler"]["inputs"]["model"] == ["lora_0", 0]


class TestReference:
    def test_mode_maps_to_weight_and_default_strength(self) -> None:
        wf = build_with_images(cn="c [temp]", reference={"image": "c", "mode": "lineart"})
        ref_apply = wf["ref_img_apply"]["inputs"]
        assert ref_apply["lllite_name"] == "anima-lllite-lineart-1.safetensors"
        assert ref_apply["strength"] == 0.68 and ref_apply["preserve_wrapper"] is False

    def test_explicit_strength_overrides_default(self) -> None:
        wf = build_with_images(cn="c [temp]", reference={"image": "c", "mode": "pose", "strength": 0.9})
        ref_apply = wf["ref_img_apply"]["inputs"]
        assert ref_apply["lllite_name"] == "anima-lllite-pose-1.safetensors"
        assert ref_apply["strength"] == 0.9

    def test_inpaint_with_own_mask(self) -> None:
        wf = build_with_images(cn="c [temp]", cn_mask="cm [temp]", reference={"image": "c", "mode": "inpaint"})
        assert wf["ref_img_apply"]["inputs"]["mask"] == ["ref_img_mask", 0]
        assert wf["ref_img_apply"]["inputs"]["preserve_wrapper"] is True

    def test_inpaint_falls_back_to_base_mask_node(self) -> None:
        # 调用方保证：inpaint 无独立蒙版时 base.mask 必存在（mask_image 参数）
        wf = build_with_images(mask="m [temp]", cn="c [temp]", reference={"image": "c", "mode": "inpaint"})
        assert wf["ref_img_apply"]["inputs"]["mask"] == ["base_img_mask", 0]

    def test_cn_wraps_lora_chain(self) -> None:
        wf = build_with_images(cn="c [temp]", loras=[{"name": "a.safetensors"}])
        assert wf["ref_img_apply"]["inputs"]["model"] == ["lora_0", 0]


class TestImg2ImgAndInpaint:
    def test_img2img_replaces_empty_latent(self) -> None:
        wf = build_with_images(ref="r [temp]")
        assert "anima_latent" not in wf
        assert wf["base_img"]["inputs"]["image"] == "r [temp]"
        assert wf["base_img_scale"]["inputs"]["width"] == 832
        assert wf["anima_sampler"]["inputs"]["latent_image"] == ["base_img_encode", 0]

    def test_inpaint_hard_locks_with_mask(self) -> None:
        wf = build_with_images(ref="r [temp]", mask="m [temp]")
        assert wf["base_img_mask"]["inputs"]["image"] == "m [temp]"
        assert wf["anima_sampler"]["inputs"]["latent_image"] == ["base_img_inpaint", 0]
