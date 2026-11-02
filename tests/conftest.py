"""
测试全局环境：conftest 先于测试模块 import 生效，置空 LOG_FILE 避免测试在项目根写日志文件。
任务库一律指向 tmp_path 下的独立目录（get_store 按 path 换连接），绝不落在项目根 data/。
注意：这里故意不用 monkeypatch——它会被本 autouse fixture 拉前实例化，颠倒
test_config_validation 里 restore_config 的拆卸顺序（reload 先于环境还原 -> SystemExit）。
"""

import os
from collections.abc import Iterator
from pathlib import Path

os.environ.setdefault("LOG_FILE", "")

import pytest

from comfyui_anima_api import config


@pytest.fixture(autouse=True)
def isolated_task_store(tmp_path: Path) -> Iterator[None]:
    original = config.TASKS_CONFIG.data_dir
    config.TASKS_CONFIG.data_dir = str(tmp_path / "tasks-data")
    yield
    config.TASKS_CONFIG.data_dir = original
