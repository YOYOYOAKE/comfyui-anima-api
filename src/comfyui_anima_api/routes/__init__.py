"""
路由层门面：ROUTERS 按挂载顺序汇总全部路由器，app 循环挂载即可。
业务路由直接以 schemas 的请求模型作 body 参数（FastAPI/pydantic 校验），
端点语义写在各路由 docstring、字段语义写在 Field(description)——二者随
/openapi.json 与 /docs 实时派生，不存在需要手工同步的第二份文档。
"""

from fastapi import APIRouter

from comfyui_anima_api.routes import generate, health, loras, tasks, upscale

ROUTERS: tuple[APIRouter, ...] = (
    health.router,
    generate.router,
    upscale.router,
    loras.router,
    tasks.router,
)

__all__ = ["ROUTERS"]
