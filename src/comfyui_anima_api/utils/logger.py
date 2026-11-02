"""
统一日志初始化：格式、级别、console + 轮转文件双通道，并把 uvicorn 的 logger 收编进来。
全项目只有这里配置 logging；应用与 uvicorn 的消息走同一格式、同一 LOG_LEVEL
（__main__.py 把 build_logging_config() 传给 uvicorn 的 log_config，勿设 use_colors，
否则 uvicorn 会向 formatters 里塞只有自家彩色 formatter 才认的 use_colors 键）。
"""

import os
from logging.config import dictConfig

from comfyui_anima_api.config import LOG_CONFIG

FORMAT = "%(asctime)s %(levelname)s %(message)s"


def _handlers() -> dict[str, dict]:
    handlers: dict[str, dict] = {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "plain",
            "stream": "ext://sys.stdout",
        },
    }
    if LOG_CONFIG.file:
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "plain",
            "filename": LOG_CONFIG.file,
            "maxBytes": LOG_CONFIG.max_bytes,
            "backupCount": LOG_CONFIG.backup_count,
            "encoding": "utf-8",
            "delay": True,  # 首条日志才建文件
        }
    return handlers


def build_logging_config() -> dict:
    """产出 dictConfig 字典；同一字典也交给 uvicorn.run(log_config=...) 收编其日志。"""
    names = list(_handlers())
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"plain": {"format": FORMAT}},
        "handlers": _handlers(),
        "root": {"level": LOG_CONFIG.level, "handlers": names},
        "loggers": {
            # uvicorn 用同一组 handler/级别，propagate=False 防止经 root 二次输出；
            # uvicorn.error 不挂 handler，沿 propagate 汇入 uvicorn
            "uvicorn": {"level": LOG_CONFIG.level, "handlers": names, "propagate": False},
            "uvicorn.error": {"level": LOG_CONFIG.level},
        },
    }


def setup_logging() -> None:
    """应用统一日志配置（幂等：重复调用只是重放同一配置）。"""
    if LOG_CONFIG.file:
        os.makedirs(os.path.dirname(LOG_CONFIG.file) or ".", exist_ok=True)
    dictConfig(build_logging_config())
