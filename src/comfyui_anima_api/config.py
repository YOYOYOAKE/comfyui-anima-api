"""
配置层：pydantic-settings 声明式分节，与请求契约（schemas.py）同一严格纪律（strict=True，
跨类型 lax 强转不存在）。env 变量名与旧版 100% 一致，且由唯一一条规则推导：

    env 变量名 = 节前缀（env_prefix） + 字段名（大小写不敏感）

全文件零 validation_alias——分节结构就是为了让这条推导对全部旧名成立；增改配置时
保持这条规则，报错（_boot 映射回 env 名）与 .env.example 都不需要额外记忆。
合法性约束全部声明在字段上（Field ge/le/pattern/min_length）。所有节在 import 期经
_boot() 实例化：非法配置立即 SystemExit（fail-fast 与旧版一致）。
模块级单例的导出名与属性形状和旧版逐一相同，消费方 import 形状不变。
"""

import os
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

SIZE_PRESETS: dict[str, tuple[int, int]] = {
    "normal_portrait": (832, 1216),
    "normal_square": (1024, 1024),
    "normal_landscape": (1216, 832),
    "large_portrait": (1024, 1536),
    "large_square": (1472, 1472),
    "large_landscape": (1536, 1024),
}

# 模型文件名约束：非空且 .safetensors 结尾（首尾锚定，杜绝 "x.safetensors.txt" 这类尾巴）
_SAFETENSORS = r"^.+\.safetensors$"

# 首尾空白剥离的字符串（env 值常有尾随空白；剥离后其余约束照常生效）
Stripped = Annotated[str, StringConstraints(strip_whitespace=True)]


def _env_name(section: type[BaseSettings], field: str) -> str:
    """env 变量名 = 节前缀 + 字段名；零别名的全文件唯一真相，报错与文档共用。"""
    return f"{section.model_config.get('env_prefix', '')}{field}".upper()


def _boot[T: BaseSettings](section: type[T]) -> T:
    """
    实例化一个配置节；非法即 SystemExit，报错点名 env 变量名。SettingsError（复杂类型
    env 值 JSON 解析失败等）同为 ValueError 子类，一并收口成同一形态。
    """
    try:
        return section()
    except ValidationError as e:
        parts = []
        for err in e.errors():
            first = str(err["loc"][0])
            name = _env_name(section, first) if first in section.model_fields else first.upper()
            where = ".".join(str(loc) for loc in err["loc"])
            parts.append(f"invalid {name}: {err['msg']} ({where})")
        raise SystemExit("; ".join(parts)) from e
    except ValueError as e:
        raise SystemExit(f"invalid config: {e}") from e


class Section(BaseSettings):
    """
    全部配置节的公共基类：从工作目录 .env 加载（真实环境变量优先），忽略无关 env。
    strict=True：env 字符串在 source 层解码为字段声明的类型，之后按严格模式校验，
    不做任何跨类型强转（与 schemas.RequestModel 同一纪律）。
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", strict=True)


# ---- API 服务（HOST / PORT）----
class _ApiService(Section):
    host: str = Field("127.0.0.1", min_length=1)
    port: int = Field(8189, ge=1, le=65535)


API_SERVICE_CONFIG = _boot(_ApiService)

# ---- 日志（LOG_*，统一由 utils/logger.py 消费）----
class _Log(Section):
    model_config = SettingsConfigDict(env_prefix="LOG_")
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    # 置空 LOG_FILE 关闭文件日志；console 输出始终保留（Docker stdout 依赖它）
    file: Stripped = "logs/comfyui-anima-api.log"
    max_bytes: int = Field(10 * 1024 * 1024, ge=1024)
    backup_count: int = Field(5, ge=0)


LOG_CONFIG = _boot(_Log)

# ---- ComfyUI（COMFYUI_*）----
class _ComfyUI(Section):
    model_config = SettingsConfigDict(env_prefix="COMFYUI_")
    health_timeout: float = Field(5.0, gt=0)
    # 必填：ComfyUI 项目根目录，LoRA 下载目录默认从它派生（纯空白 strip 后过 min_length 即拒）
    path: Stripped = Field(min_length=1)
    base_url: str = Field("http://127.0.0.1:8188", min_length=1)
    client_id: str = Field("comfyui-anima-api", min_length=1)
    poll_interval: float = Field(1.0, gt=0)
    connect_timeout: float = Field(60, gt=0)


COMFYUI_CONFIG = _boot(_ComfyUI)

# ---- Civitai（CIVITAI_*）----
class _Civitai(Section):
    model_config = SettingsConfigDict(env_prefix="CIVITAI_")
    loras_download_dir: str = ""  # 空 -> 模块级派生 ${COMFYUI_PATH}/models/loras
    base_url: str = Field("https://civitai.com", min_length=1)
    extra_headers: dict[str, str] = Field(default_factory=dict)  # JSON object 字符串
    api_key: str | None = None  # 下载受限资源时需填写
    proxy: str | None = None
    connect_timeout: float = Field(30, gt=0)
    read_timeout: float = Field(60, gt=0)
    ua: str = Field("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36", min_length=1)
    download_timeout: float = Field(1800, gt=0)


CIVITAI_CONFIG = _boot(_Civitai)
# 派生规则保持在模块级（跨节组合）：未显式指定下载目录时挂到 ComfyUI 的 loras/
if not CIVITAI_CONFIG.loras_download_dir:
    CIVITAI_CONFIG.loras_download_dir = os.path.join(COMFYUI_CONFIG.path, "models", "loras")

# ---- aria2（ARIA2_*）----
class _Aria2(Section):
    model_config = SettingsConfigDict(env_prefix="ARIA2_")
    connections: int = Field(8, ge=1)
    connect_timeout: int = Field(60, ge=1)
    max_tries: int = Field(5, ge=0)
    retry_wait: int = Field(3, ge=0)
    lowest_speed_limit: str = Field("1K", min_length=1)


ARIA2_CONFIG = _boot(_Aria2)

# ---- Anima 基础模型（历史上无公共前缀，字段即全名）----
class _AnimBase(Section):
    anima_model: str = Field("anima-base-v1.0.safetensors", pattern=_SAFETENSORS)
    text_encoder: str = Field("qwen_3_06b_base.safetensors", pattern=_SAFETENSORS)
    vae: str = Field("qwen_image_vae.safetensors", pattern=_SAFETENSORS)


# ---- ControlNet 模型（CONTROLNET_{MODE}_MODEL）----
class _ControlNetModels(Section):
    model_config = SettingsConfigDict(env_prefix="CONTROLNET_")
    mix_model: str = Field("anima-lllite-any-test-like-v2.safetensors", pattern=_SAFETENSORS)
    inpaint_model: str = Field("anima-lllite-inpainting-v1.safetensors", pattern=_SAFETENSORS)
    depth_model: str = Field("anima-lllite-depth-1.safetensors", pattern=_SAFETENSORS)
    pose_model: str = Field("anima-lllite-pose-1.safetensors", pattern=_SAFETENSORS)
    lineart_model: str = Field("anima-lllite-lineart-1.safetensors", pattern=_SAFETENSORS)


# ---- Sharp 超分（SHARP_UPSCALE_MODEL 模型 + SHARP_UPSCALE_BY 放大倍率，同节）----
class _Sharp(Section):
    model_config = SettingsConfigDict(env_prefix="SHARP_")
    upscale_model: str = Field("4x-AnimeSharp.safetensors", pattern=_SAFETENSORS)
    upscale_by: float = Field(2.0, gt=0)


# env 读取节与导出结构分离（同 MODEL_CONFIG 模式）：导出形状与旧版逐一相同
class AnimBaseConfig(BaseModel):
    anima: str
    text_encoder: str
    vae: str


class ControlNetConfig(BaseModel):
    mix: str
    inpaint: str
    depth: str
    pose: str
    lineart: str


class ModelConfig(BaseModel):
    anima_base: AnimBaseConfig
    sharp: str
    control_net: ControlNetConfig


class SharpParams(BaseModel):
    upscale_by: float


class UpscaleConfig(BaseModel):
    sharp: SharpParams


_animb = _boot(_AnimBase)
_cn = _boot(_ControlNetModels)
_sharp = _boot(_Sharp)
MODEL_CONFIG = ModelConfig(
    anima_base=AnimBaseConfig(anima=_animb.anima_model, text_encoder=_animb.text_encoder, vae=_animb.vae),
    sharp=_sharp.upscale_model,
    control_net=ControlNetConfig(
        mix=_cn.mix_model,
        inpaint=_cn.inpaint_model,
        depth=_cn.depth_model,
        pose=_cn.pose_model,
        lineart=_cn.lineart_model,
    ),
)
UPSCALE_CONFIG = UpscaleConfig(sharp=SharpParams(upscale_by=_sharp.upscale_by))

# ---- 默认绘图参数（DEFAULT_GENERATE_*）----
class _GenerateDefaults(Section):
    model_config = SettingsConfigDict(env_prefix="DEFAULT_GENERATE_")
    sampler: str = Field("euler", min_length=1)
    scheduler: str = Field("simple", min_length=1)
    steps: int = Field(30, ge=1)
    cfg: float = Field(4.5, gt=0)
    img2img_denoise: float = Field(0.6, gt=0, le=1)
    size: str = "normal_portrait"
    timeout: float = Field(600, gt=0)

    @field_validator("size")
    @classmethod
    def _size_preset(cls, v: str) -> str:
        if v not in SIZE_PRESETS:
            raise ValueError(f"must be one of: {', '.join(SIZE_PRESETS)}")
        return v


# ---- 加速开关（DEFAULT_TEACACHE_ENABLED / DEFAULT_ANIMA_BOOSTER_SAGE_ATTENTION_ENABLED）----
# 旧名不合 DEFAULT_GENERATE_ 前缀，独立成节让"前缀+字段"推导成立，再组装回 GENERATE_DEFAULTS
class _AcceleratorSwitches(Section):
    model_config = SettingsConfigDict(env_prefix="DEFAULT_")
    teacache_enabled: bool = True
    anima_booster_sage_attention_enabled: bool = True


class GenerateDefaults(BaseModel):
    """导出形状与旧版一致（teacache / sage_attention 短名），供 schemas 消费。"""

    sampler: str
    scheduler: str
    steps: int
    cfg: float
    img2img_denoise: float
    size: str
    teacache: bool
    sage_attention: bool
    timeout: float


_gd = _boot(_GenerateDefaults)
_sw = _boot(_AcceleratorSwitches)
GENERATE_DEFAULTS = GenerateDefaults(
    sampler=_gd.sampler,
    scheduler=_gd.scheduler,
    steps=_gd.steps,
    cfg=_gd.cfg,
    img2img_denoise=_gd.img2img_denoise,
    size=_gd.size,
    teacache=_sw.teacache_enabled,
    sage_attention=_sw.anima_booster_sage_attention_enabled,
    timeout=_gd.timeout,
)

# ---- 任务（DATA_DIR，SQLite 持久化 + SSE 进度推送）----
class _Tasks(Section):
    # 置空 DATA_DIR 回落默认值（与 LOG_FILE 的置空语义不同：任务库没有"关闭"一说）
    data_dir: str = "data"

    @field_validator("data_dir", mode="before")
    @classmethod
    def _fallback_default(cls, v: object) -> object:
        return v.strip() or "data" if isinstance(v, str) else v


TASKS_CONFIG = _boot(_Tasks)  # 任务库 tasks.db 落这里（WAL 模式，连同 -wal/-shm 一起备份/清理）

# ---- Anima Booster（ANIMA_BOOSTER_*）----
class _AnimaBooster(Section):
    model_config = SettingsConfigDict(env_prefix="ANIMA_BOOSTER_")
    torch_compile: bool = False


ANIMA_BOOSTER_CONFIG = _boot(_AnimaBooster)

# ---- 默认 ControlNet 参考强度（DEFAULT_REFERENCE_{MODE}_STRENGTH，字段名含后缀）----
class _ReferenceStrength(Section):
    model_config = SettingsConfigDict(env_prefix="DEFAULT_REFERENCE_")
    mix_strength: float = Field(0.68, ge=0)
    inpaint_strength: float = Field(1.0, ge=0)
    depth_strength: float = Field(0.68, ge=0)
    pose_strength: float = Field(1.0, ge=0)
    lineart_strength: float = Field(0.68, ge=0)


REFERENCE_STRENGTH_DEFAULTS = _boot(_ReferenceStrength)

# ---- TeaCache（TEACACHE_*；字段 version -> env TEACACHE_VERSION，节点输入名映射在 workflows）----
class _TeaCache(Section):
    model_config = SettingsConfigDict(env_prefix="TEACACHE_")
    min_steps: int = Field(12, ge=0)
    threshold: float = Field(0.15, ge=0)
    version: str = Field("v2 (Standard Precise)", min_length=1)
    adaptive_mode: bool = True
    early_steps_factor: float = Field(0.4, ge=0)
    late_steps_factor: float = Field(1.8, ge=0)
    start_percent: float = Field(0.0, ge=0, le=1)
    end_percent: float = Field(1.0, ge=0, le=1)
    cache_device: str = Field("cuda", min_length=1)


TEACACHE_CONFIG = _boot(_TeaCache)
