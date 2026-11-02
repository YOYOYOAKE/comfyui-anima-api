"""schemas.py 单测：pydantic 请求契约的默认值回落、三条硬规则（forbid/null=unset/strict）与 provided 记账。"""

import pytest
from pydantic import ValidationError

from comfyui_anima_api.config import GENERATE_DEFAULTS, REFERENCE_STRENGTH_DEFAULTS
from comfyui_anima_api.schemas import (
    DownloadParams,
    GenerateParams,
    LoraSpec,
    UpscaleParams,
)

# 真 PNG 文件头的 b64：decode_b64_image 按 magic bytes 预检，裸 b"x" 之类会被 400
PNG_B64 = "iVBORw0KGgo="


def first_error(exc: ValidationError) -> tuple[object, tuple[object, ...], str]:
    err = exc.errors()[0]
    return err["type"], tuple(err["loc"]), err["msg"]


class TestGenerateParams:
    def test_all_defaults(self) -> None:
        p = GenerateParams.model_validate({})
        assert p.prompt.positive == ""
        assert p.prompt.negative == ""
        assert p.seed is None
        assert p.loras == []
        assert p.base.image is None and p.base.denoise is None and p.base.mask is None
        assert p.reference.mode.value == "mix"
        # strength 缺省不在此收敛（由 service 按 mode 解析，保持 provided 纯净）
        assert p.reference.strength is None
        assert p.advance.steps == GENERATE_DEFAULTS.steps and p.advance.cfg == GENERATE_DEFAULTS.cfg
        assert p.advance.sampler == GENERATE_DEFAULTS.sampler and p.advance.scheduler == GENERATE_DEFAULTS.scheduler
        assert p.size.value == "normal_portrait"
        assert p.advance.teacache is True and p.advance.sage_attention is True
        assert p.timeout == 600
        assert p.provided == frozenset()

    def test_explicit_null_resolves_to_defaults(self) -> None:
        # 显式传 null 视为未设置：丢键回落默认值，也不计入 provided
        p = GenerateParams.model_validate({"advance": {"steps": None, "sampler": None}})
        assert p.advance.steps == GENERATE_DEFAULTS.steps
        assert p.advance.sampler == GENERATE_DEFAULTS.sampler
        assert p.advance.provided == frozenset()

    def test_loras_objects_only(self) -> None:
        # 元素只认 {name, strength} 对象；字符串元素定位到 loras[0] 点名拒绝
        p = GenerateParams.model_validate(
            {"loras": [{"name": "a.safetensors"}, {"name": "b.safetensors", "strength": 0.5}]}
        )
        assert p.loras[0] == LoraSpec(name="a.safetensors")
        assert isinstance(p.loras[1], LoraSpec) and p.loras[1].strength == 0.5
        with pytest.raises(ValidationError) as ei:
            GenerateParams.model_validate({"loras": ["a.safetensors"]})
        typ, loc, _ = first_error(ei.value)
        assert typ == "model_type" and loc == ("loras", 0)

    def test_count_capped_at_8(self) -> None:
        # 批量上限 8：对单任务的 ComfyUI 排队时长封顶（images.zip 已流式化）
        with pytest.raises(ValidationError, match="less than or equal to 8"):
            GenerateParams.model_validate({"count": 9})
        assert GenerateParams.model_validate({"count": 8}).count == 8

    def test_provided_records_explicit_keys_only(self) -> None:
        # provided 是"多传即错"的判定依据：显式提供的键集合，与值是否等于默认值无关
        p = GenerateParams.model_validate({"prompt": {"positive": ""}, "count": 1})
        assert p.provided == {"count", "prompt"}
        assert p.prompt.provided == {"positive"}

    def test_strict_types_no_lax_coercion(self) -> None:
        # 类型严格：字符串数字/布尔不做 lax 强转，直接点名拒绝
        with pytest.raises(ValidationError) as ei:
            GenerateParams.model_validate({"advance": {"steps": "30"}})
        assert first_error(ei.value)[0] == "int_type"
        with pytest.raises(ValidationError) as ei:
            GenerateParams.model_validate({"advance": {"teacache": "true"}})
        assert first_error(ei.value)[0] == "bool_type"
        with pytest.raises(ValidationError) as ei:
            GenerateParams.model_validate({"seed": 7.5})
        assert str(first_error(ei.value)[0]).startswith("int")

    def test_seed_bounds_match_comfyui(self) -> None:
        # 与 ComfyUI KSampler 同域 [0, 2^64-1]：越界即时 400
        with pytest.raises(ValidationError) as ei:
            GenerateParams.model_validate({"seed": -1})
        assert first_error(ei.value)[0] == "greater_than_equal"
        with pytest.raises(ValidationError) as ei:
            GenerateParams.model_validate({"seed": 2**64})
        assert first_error(ei.value)[0] == "less_than_equal"
        assert GenerateParams.model_validate({"seed": 2**64 - 1}).seed == 2**64 - 1


class TestEnumValidation:
    def test_invalid_size_rejected(self) -> None:
        # size 是顶层参数；advance 组内出现即为未知字段（forbid 点名拒绝）
        with pytest.raises(ValidationError) as ei:
            GenerateParams.model_validate({"size": "huge"})
        assert first_error(ei.value)[0] == "enum"
        with pytest.raises(ValidationError) as ei:
            GenerateParams.model_validate({"advance": {"size": "huge"}})
        typ, loc, _ = first_error(ei.value)
        assert typ == "extra_forbidden" and loc == ("advance", "size")

    def test_invalid_reference_mode_rejected(self) -> None:
        with pytest.raises(ValidationError) as ei:
            GenerateParams.model_validate({"reference": {"image": "x", "mode": "segment"}})
        assert first_error(ei.value)[0] == "enum"

    def test_upscale_group_removed_from_generate(self) -> None:
        # 超分走 POST /upscale：generate 不认识 upscale 参数，整组出现即未知字段
        with pytest.raises(ValidationError) as ei:
            GenerateParams.model_validate({"upscale": {"enable": True}})
        typ, loc, _ = first_error(ei.value)
        assert typ == "extra_forbidden" and loc == ("upscale",)

    def test_denoise_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError) as ei:
            GenerateParams.model_validate({"base": {"image": "x", "denoise": 0}})
        typ, loc, _ = first_error(ei.value)
        assert typ == "greater_than" and loc == ("base", "denoise")
        with pytest.raises(ValidationError) as ei:
            GenerateParams.model_validate({"base": {"image": "x", "denoise": 1.5}})
        typ, loc, _ = first_error(ei.value)
        assert typ == "less_than_equal" and loc == ("base", "denoise")

    def test_reference_strength_out_of_range_rejected(self) -> None:
        # Field 声明的 (0,1] 域：越界即时 400，而非提交后才被节点拒绝
        for bad in (0, 1.5):
            with pytest.raises(ValidationError) as ei:
                GenerateParams.model_validate({"reference": {"image": "x", "strength": bad}})
            typ, loc, _ = first_error(ei.value)
            assert loc == ("reference", "strength") and typ in ("greater_than", "less_than_equal")


class TestUnknownAndShapes:
    def test_unknown_field_named_with_path(self) -> None:
        with pytest.raises(ValidationError) as ei:
            GenerateParams.model_validate({"reference": {"extra": 1}})
        typ, loc, _ = first_error(ei.value)
        assert typ == "extra_forbidden" and loc == ("reference", "extra")

    def test_bare_loras_string_rejected(self) -> None:
        # 裸字符串不是列表，点名拒绝而非逐字符误解析
        with pytest.raises(ValidationError) as ei:
            GenerateParams.model_validate({"loras": "a.safetensors"})
        assert first_error(ei.value)[0] == "list_type"

    def test_bad_lora_name_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"plain '<name>\.safetensors'"):
            GenerateParams.model_validate({"loras": [{"name": "../evil.safetensors"}]})

    def test_bad_download_filename_rejected(self) -> None:
        # 与 loras.name 同一规矩：纯文件名 + .safetensors，路径穿越在校验层即拒
        with pytest.raises(ValidationError, match=r"plain '<name>\.safetensors'"):
            DownloadParams.model_validate({"version_id": 1, "filename": "../evil.safetensors", "trigger": "t"})

    def test_non_object_body_rejected(self) -> None:
        with pytest.raises(ValidationError) as ei:
            GenerateParams.model_validate("garbage")
        assert first_error(ei.value)[0] == "model_type"


class TestStrengthSemantics:
    """strength 的缺省收敛移到了 generation.service（执行时解析，与 denoise 同款）；
    schema 层只保证：显式值域校验、None 保持 None、provided 记账不含收敛污染。"""

    def test_explicit_strength_kept(self) -> None:
        p = GenerateParams.model_validate({"reference": {"image": "x", "strength": 0.9}})
        assert p.reference.strength == 0.9
        assert p.reference.provided == {"image", "strength"}

    def test_default_strength_stays_none(self) -> None:
        p = GenerateParams.model_validate({"reference": {"image": "x", "mode": "pose"}})
        assert p.reference.strength is None
        assert p.reference.provided == {"image", "mode"}
        # service 侧解析口径：pose 的默认强度来自 REFERENCE_STRENGTH_DEFAULTS
        assert REFERENCE_STRENGTH_DEFAULTS.pose_strength == 1.0


class TestUpscaleAndDownload:
    def test_upscale_minimum(self) -> None:
        p = UpscaleParams.model_validate({"image": PNG_B64})
        assert p.image == PNG_B64 and p.timeout == GENERATE_DEFAULTS.timeout

    def test_upscale_requires_image(self) -> None:
        # 必填缺失是 ValidationError（经 handler 转 400），不是 TypeError（会变 500）
        with pytest.raises(ValidationError) as ei:
            UpscaleParams.model_validate({})
        assert first_error(ei.value)[0] == "missing"

    def test_download_minimum(self) -> None:
        p = DownloadParams.model_validate({"version_id": 1, "filename": "a.safetensors", "trigger": "t"})
        assert p.version_id == 1 and p.trigger == "t"

    def test_download_empty_trigger_rejected(self) -> None:
        with pytest.raises(ValidationError) as ei:
            DownloadParams.model_validate({"version_id": 1, "filename": "a.safetensors", "trigger": ""})
        assert first_error(ei.value)[0] == "string_too_short"
