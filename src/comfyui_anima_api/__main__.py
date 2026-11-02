import uvicorn

from comfyui_anima_api.app import app
from comfyui_anima_api.config import API_SERVICE_CONFIG
from comfyui_anima_api.utils import build_logging_config


def main() -> None:
    uvicorn.run(
        app,
        host=API_SERVICE_CONFIG.host,
        port=API_SERVICE_CONFIG.port,
        access_log=False,
        # 同一份 dictConfig 交给 uvicorn（勿设 use_colors）：其启动/停机日志与应用同格式、同 LOG_LEVEL
        log_config=build_logging_config(),
    )


if __name__ == "__main__":
    main()
