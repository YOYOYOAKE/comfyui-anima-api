"""config.py 统一校验层测试：非法配置必须拒绝启动（SystemExit），合法配置正常加载。
与请求契约（schemas.py）同一严格纪律：strict=True，跨类型 lax 强转不存在。"""

import importlib
from collections.abc import Iterator
from types import ModuleType

import pytest


def _reload_config(monkeypatch: pytest.MonkeyPatch, **env: str) -> ModuleType:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import comfyui_anima_api.config as config

    importlib.reload(config)
    return config


@pytest.fixture(autouse=True)
def restore_config() -> Iterator[None]:
    yield
    import comfyui_anima_api.config as config

    importlib.reload(config)  # 用当前环境重新加载，恢复被测试改写的模块级配置


def test_bad_bool_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    # 旧版变量名经 alias 锚定（报错也点名旧名）
    with pytest.raises(SystemExit, match="DEFAULT_TEACACHE_ENABLED"):
        _reload_config(monkeypatch, DEFAULT_TEACACHE_ENABLED="ture")


def test_non_integer_port_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    # strict：字符串数字之外的任何形态（含 "8189.5"）都不强转
    with pytest.raises(SystemExit, match="PORT"):
        _reload_config(monkeypatch, PORT="8189.5")


def test_port_out_of_range_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit, match="PORT"):
        _reload_config(monkeypatch, PORT="99999")


def test_bad_model_filename_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit, match="ANIMA_MODEL"):
        _reload_config(monkeypatch, ANIMA_MODEL="model.ckpt")


def test_bad_default_size_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit, match="DEFAULT_GENERATE_SIZE"):
        _reload_config(monkeypatch, DEFAULT_GENERATE_SIZE="huge")


def test_non_numeric_timeout_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit, match="DEFAULT_GENERATE_TIMEOUT"):
        _reload_config(monkeypatch, DEFAULT_GENERATE_TIMEOUT="soon")


def test_comfyui_path_required(monkeypatch: pytest.MonkeyPatch) -> None:
    # 置为纯空白即视为未设置（strip 后过 min_length）
    with pytest.raises(SystemExit, match="COMFYUI_PATH"):
        _reload_config(monkeypatch, COMFYUI_PATH="  ")


def test_valid_overrides_load(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _reload_config(
        monkeypatch,
        COMFYUI_PATH="/comfy",
        PORT="8080",
        DEFAULT_REFERENCE_POSE_STRENGTH="0.9",
        DEFAULT_ANIMA_BOOSTER_SAGE_ATTENTION_ENABLED="false",
    )
    assert config.API_SERVICE_CONFIG.port == 8080
    assert config.CIVITAI_CONFIG.loras_download_dir == "/comfy/models/loras"
    assert config.REFERENCE_STRENGTH_DEFAULTS.pose_strength == 0.9
    assert config.GENERATE_DEFAULTS.sage_attention is False


def test_teacache_enabled_override_flows_to_generate_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    # DEFAULT_TEACACHE_ENABLED（旧版变量名，alias 锚定）决定 generate 请求 teacache 的默认值
    config = _reload_config(monkeypatch, COMFYUI_PATH="/comfy", DEFAULT_TEACACHE_ENABLED="false")
    assert config.GENERATE_DEFAULTS.teacache is False
