import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from comfyui_anima_api.comfyui import comfyui_progress
from comfyui_anima_api.config import API_SERVICE_CONFIG, CIVITAI_CONFIG, COMFYUI_CONFIG
from comfyui_anima_api.loras import aria2c_available
from comfyui_anima_api.routes import ROUTERS
from comfyui_anima_api.tasks import get_store, reconcile
from comfyui_anima_api.utils import setup_logging

logger = logging.getLogger(__name__)


def _version() -> str:
    try:
        return version("comfyui-anima-api")
    except PackageNotFoundError:
        return "dev"


# info.description：全局契约规则与最小调用链（/docs 首页渲染；语义维护在各路由 docstring 与 Field.description）
_API_DESCRIPTION = """Agent 友好的 Anima 工作流 API：文生图 / 图生图 / 局部重绘 / 参考引导 / 超分 / LoRA 下载。

### 全局规则

- 不支持未知字段：拼错键名点名报错（400），不会被静默吞掉
- JSON 严格类型检查：不做 `"30" -> 30` 之类的 lax 强转，类型不对即 400
- 显式 `null` 等价未设置：等价省略该键，回落默认值
- 底图 / 参考图 / 超分原图用 PNG / JPG / WebP 的 base64 或 `data:` URL

### 快速开始

```bash
TASK=$(curl -sS -X POST http://127.0.0.1:8189/anima/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt": {"positive": "masterpiece, best quality, 1girl, solo"}}' | cut -d'"' -f4)
curl -N  http://127.0.0.1:8189/tasks/$TASK/events      # SSE 进度
curl -sS -o out.zip http://127.0.0.1:8189/tasks/$TASK/images.zip
curl -sS -X POST http://127.0.0.1:8189/tasks/$TASK/cancel   # 幂等取消
```

各端点的字段约束与语义见 `/docs`（Swagger UI）或 `/openapi.json`。"""


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    # 建目录建表（懒初始化的主动触发；裸 TestClient 不进 lifespan 时首次访问也会自建）
    get_store()
    await comfyui_progress.start()
    try:
        await reconcile()
    except Exception:
        logger.exception("startup reconcile failed; unfinished tasks left as-is")
    yield
    await comfyui_progress.stop()


def create_app() -> FastAPI:
    setup_logging()
    if API_SERVICE_CONFIG.host == "0.0.0.0":
        logger.warning(
            "HOST=0.0.0.0: listening on all interfaces without authentication; any device on the LAN can call "
            "this API, make sure the network is trusted"
        )
    if not aria2c_available():
        logger.warning("aria2c not found in PATH: /lora/download is disabled (returns 503 until aria2 is installed)")
    if not CIVITAI_CONFIG.api_key:
        logger.warning("CIVITAI_API_KEY not set: LoRAs that require authentication will fail to download")
    application = FastAPI(
        lifespan=lifespan, title="comfyui-anima-api", version=_version(), description=_API_DESCRIPTION
    )

    @application.middleware("http")
    async def access_log(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        started = time.monotonic()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("%s %s unhandled error", request.method, request.url.path)
            raise

        if request.url.path == "/health" and response.status_code < 400:
            return response
        logger.info(
            "%s %s -> %d (%.1fs) from %s",
            request.method,
            request.url.path,
            response.status_code,
            time.monotonic() - started,
            request.client.host if request.client else "-",
        )
        return response

    @application.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, dict) else {"error": exc.detail}
        return JSONResponse(detail, status_code=exc.status_code)

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # 全部错误逐条点名（loc 去掉首段 "body" 后是字段路径，如 base.image / loras[0].name；
        # 坏 JSON 的 loc 里带 None，跳过）
        parts: list[str] = []
        for err in exc.errors():
            where = ".".join(str(loc) for loc in err.get("loc", [])[1:] if loc is not None)
            parts.append(f"{where} {err.get('msg', 'validation failed')}".strip())
        return JSONResponse({"error": "invalid request: " + "; ".join(parts)}, status_code=400)

    for router in ROUTERS:
        application.include_router(router)

    logger.info(
        "config: COMFYUI_BASE_URL=%s CIVITAI_LORAS_DOWNLOAD_DIR=%s CIVITAI_BASE_URL=%s",
        COMFYUI_CONFIG.base_url,
        CIVITAI_CONFIG.loras_download_dir,
        CIVITAI_CONFIG.base_url,
    )
    logger.info(
        "config: CIVITAI_PROXY=%s civitai_extra_headers=%s",
        CIVITAI_CONFIG.proxy or "(direct)",
        list(CIVITAI_CONFIG.extra_headers) or "-",
    )
    return application


app = create_app()
