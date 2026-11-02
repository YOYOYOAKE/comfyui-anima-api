"""utils/logger.py 统一日志配置测试：console+文件双通道、uvicorn 收编、LOG_LEVEL 生效。"""

import importlib
import logging
import os
import re
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest


def _reload(**env: str) -> ModuleType:
    import comfyui_anima_api.config as config
    import comfyui_anima_api.utils.logger as logger_mod

    for key, value in env.items():
        os.environ[key] = value
    importlib.reload(config)
    importlib.reload(logger_mod)  # 重新绑定 reload 后的 LOG_CONFIG
    return logger_mod


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    saved = {k: os.environ.get(k) for k in ("LOG_FILE", "LOG_LEVEL")}
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    _reload().setup_logging()


def test_file_handler_writes_formatted_lines(tmp_path: Path) -> None:
    log_file = tmp_path / "sub" / "api.log"
    logger_mod = _reload(LOG_FILE=str(log_file))
    logger_mod.setup_logging()
    logging.getLogger("test").info("hello %s", "file")
    for h in logging.getLogger().handlers:
        h.flush()
    line = log_file.read_text(encoding="utf-8").strip().splitlines()[-1]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} INFO hello file", line)


def test_empty_log_file_disables_file_handler() -> None:
    logger_mod = _reload(LOG_FILE="")
    logger_mod.setup_logging()
    assert not any(hasattr(h, "baseFilename") for h in logging.getLogger().handlers)


def test_uvicorn_loggers_adopt_config() -> None:
    logger_mod = _reload(LOG_FILE="")
    logger_mod.setup_logging()
    uvicorn = logging.getLogger("uvicorn")
    assert uvicorn.level == logging.INFO
    assert uvicorn.propagate is False
    assert [type(h) for h in uvicorn.handlers] == [type(h) for h in logging.getLogger().handlers]
    uvicorn_error = logging.getLogger("uvicorn.error")
    assert not uvicorn_error.handlers
    assert uvicorn_error.propagate is True


def test_log_level_applies_to_root_and_uvicorn() -> None:
    logger_mod = _reload(LOG_LEVEL="WARNING", LOG_FILE="")
    logger_mod.setup_logging()
    assert logging.getLogger().level == logging.WARNING
    assert logging.getLogger("uvicorn").level == logging.WARNING
