"""
请求契约：pydantic BaseModel，三条硬规则（面向 Agent 调用方，语义必须无歧义）全部
集中在 RequestModel 基类：
1. 未知字段一律拒绝（extra="forbid"，报错点名到字段路径）——拼错键名直接暴露；
2. 显式 null 一律等于未设置（before validator 剔除 null 键，等价省略该键，也不计入 provided）；
3. 类型严格（strict：不做 "30"->30、"true"->true 之类的 lax 强转，类型不对即 400）。
字段约束声明在 Field 上，语义说明随 description 进入 /docs 与 /openapi.json。
provided（显式提供的键集合，与值无关）由 model_fields_set 承载——service 据此拒绝
当前模式下无意义的字段（多传即错）。reference.strength 的"缺省按模式收敛"不在 schema
做（会污染 provided 记账），由 generation.service 在执行时解析，与 base.denoise 同款。
"""

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from comfyui_anima_api.config import GENERATE_DEFAULTS, SIZE_PRESETS, ControlNetConfig

# 落盘/引用文件名仅允许无路径分隔符的纯文件名（loras/download 与 loras.name 共用同一规矩）
LORA_FILENAME_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9 _()\[\].-]*\.safetensors")


class SizeMode(StrEnum):
    normal_portrait = "normal_portrait"
    normal_square = "normal_square"
    normal_landscape = "normal_landscape"
    large_portrait = "large_portrait"
    large_square = "large_square"
    large_landscape = "large_landscape"


class ReferenceMode(StrEnum):
    mix = "mix"
    inpaint = "inpaint"
    depth = "depth"
    pose = "pose"
    lineart = "lineart"


# 枚举与 config 的取值集合锁定：改 SIZE_PRESETS / ControlNet 权重时两边一起改，漂移在导入期即报错
# （显式 raise 而非 assert：python -O 会剥掉 assert，校验就静默失效了）
if {m.value for m in SizeMode} != set(SIZE_PRESETS):
    raise RuntimeError("SizeMode 与 config.SIZE_PRESETS 取值集漂移，需两边同步修改")
if {m.value for m in ReferenceMode} != set(ControlNetConfig.model_fields):
    raise RuntimeError("ReferenceMode 与 config.ControlNetConfig 取值集漂移，需两边同步修改")


def _plain_safetensors(v: str) -> str:
    """与 /lora/download 的落盘文件名同规矩：纯文件名、.safetensors 结尾，杜绝路径拼接。"""
    if not LORA_FILENAME_RE.fullmatch(v):
        raise ValueError("must be a plain '<name>.safetensors' filename without path separators")
    return v


class RequestModel(BaseModel):
    """全部请求模型的基类：三件硬规则（见模块 docstring）。provided 不做成字段，
    以 property 暴露：不参与校验输入，也不会混进 model_dump。"""

    model_config = ConfigDict(strict=True, extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _drop_nulls(cls, data: object) -> object:
        # 显式 null = 未设置：丢键回落默认值，也不计入 provided
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if v is not None}
        return data

    @property
    def provided(self) -> frozenset[str]:
        """显式提供且非 null 的字段名集合（"多传即错"的判定依据，与值是否等于默认无关）。"""
        return frozenset(self.model_fields_set)


class LoraSpec(RequestModel):
    name: str = Field(description="LoRA 文件名（<name>.safetensors，纯文件名，无路径分隔符）")
    strength: float = Field(default=1.0, description="LoRA 强度")

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        return _plain_safetensors(v)


class PromptGroup(RequestModel):
    positive: str = Field(default="", description="正面提示词，必填（空白串会被点名拒绝）")
    negative: str = Field(default="", description="负面提示词，默认为空")


class BaseGroup(RequestModel):
    """底图组：提供 latent 的"画布上已有什么"。image 存在即图生图；mask 为硬锁定重绘蒙版。"""

    image: str | None = Field(
        default=None, description="底图（PNG/JPG/WebP 的 base64 或 data: URL）；提供即图生图"
    )
    denoise: float | None = Field(
        default=None, gt=0, le=1, description="重绘幅度 (0,1]；仅随 image 有效，缺省取服务端默认"
    )
    mask: str | None = Field(
        default=None, description="局部重绘蒙版（灰度 PNG base64，白=重绘）；仅随 image 有效"
    )


class ReferenceGroup(RequestModel):
    """参考组：LLLite ControlNet 引导。与 base 独立可叠加；组内任何字段没有 image 都无意义
    （service 拒绝）。strength 缺省值由 service 按 mode 解析（不在此收敛，保持 provided 纯净）。"""

    image: str | None = Field(default=None, description="参考图（base64 或 data: URL）")
    # 枚举字段 strict=False：strict 模式下 StrEnum 只认枚举实例（JSON 字符串会 is_instance_of 拒绝）；
    # lax 枚举仍按值集合封闭校验（"segment" 照样 enum 报错点名合法值），语义不失严
    mode: ReferenceMode = Field(default=ReferenceMode.mix, strict=False, description="参考模式（LLLite 引导类型）")
    strength: float | None = Field(
        default=None, gt=0, le=1, description="参考强度 (0,1]；缺省按模式取服务端默认"
    )
    mask: str | None = Field(
        default=None, description="CN 侧蒙版；仅 inpaint 模式有效，缺省回退 base.mask"
    )


class AdvanceGroup(RequestModel):
    """高级生图配置，一般无需改动；默认值来自服务端配置（.env 可调）。"""

    steps: int = Field(default=GENERATE_DEFAULTS.steps, ge=1, le=1000, description="采样步数（1~1000）")
    cfg: float = Field(default=GENERATE_DEFAULTS.cfg, gt=0, description="CFG 引导系数（正数）")
    sampler: str = Field(default=GENERATE_DEFAULTS.sampler, min_length=1, description="ComfyUI 采样器")
    scheduler: str = Field(default=GENERATE_DEFAULTS.scheduler, min_length=1, description="ComfyUI 调度器")
    teacache: bool = Field(default=GENERATE_DEFAULTS.teacache, description="TeaCache 缓存加速开关")
    sage_attention: bool = Field(default=GENERATE_DEFAULTS.sage_attention, description="SageAttention 加速开关")


class GenerateParams(RequestModel):
    prompt: PromptGroup = Field(default_factory=PromptGroup, description="提示词组")
    # seed 与 ComfyUI KSampler 同域：[0, 2^64-1]；显式越界即时 400（而非提交后才被节点拒绝）
    seed: int | None = Field(
        default=None, ge=0, le=2**64 - 1, description="自定义种子 [0, 2^64-1]；不填随机；count>1 时禁止提供"
    )
    # 批量张数：逐张顺序采样；>1 时不可指定 seed（400），每张独立随机。
    # 上限 8：对单任务的 ComfyUI 排队时长封顶（images.zip 为流式，内存不受批量影响）
    count: int = Field(
        default=1, ge=1, le=8, description="批量出图张数（1~8）；>1 时不可指定 seed，每张独立随机种子"
    )
    # 同 ReferenceGroup.mode：枚举按值精确匹配（大小写敏感、集合封闭），字符串输入照收
    size: SizeMode = Field(default=SizeMode(GENERATE_DEFAULTS.size), strict=False, description="生图尺寸预设")
    loras: list[LoraSpec] = Field(default_factory=list, description="LoRA 链（按序应用）")
    base: BaseGroup = Field(default_factory=BaseGroup, description="底图组（图生图/局部重绘）")
    reference: ReferenceGroup = Field(default_factory=ReferenceGroup, description="参考组（ControlNet 引导）")
    advance: AdvanceGroup = Field(default_factory=AdvanceGroup, description="高级生图配置")
    timeout: float = Field(
        default=GENERATE_DEFAULTS.timeout, gt=0, description="单张图执行的秒数上限；超时任务转 error(504)"
    )


class UpscaleParams(RequestModel):
    """POST /upscale 请求体：底图经 4x-AnimeSharp 放大 2x（纯图像链，无扩散重绘）。"""

    image: str = Field(description="待超分底图（PNG/JPG/WebP 的 base64 或 data: URL）")
    timeout: float = Field(
        default=GENERATE_DEFAULTS.timeout, gt=0, description="执行的秒数上限；超时任务转 error(504)"
    )


class DownloadParams(RequestModel):
    """POST /lora/download 请求体。鉴权凭据只认服务端环境变量 CIVITAI_API_KEY。"""

    version_id: int = Field(description="Civitai 版本 ID")
    filename: str = Field(description="落盘文件名（<name>.safetensors，纯文件名）；同名覆盖")
    trigger: str = Field(min_length=1, description="触发词；随 loras.json 登记，/lora/list 可查")

    @field_validator("filename")
    @classmethod
    def _check_filename(cls, v: str) -> str:
        return _plain_safetensors(v)
