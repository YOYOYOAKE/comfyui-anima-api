# ComfyUI Anima API

Agent 友好的、适用于 [Anima Base](https://huggingface.co/circlestone-labs/Anima) 的 ComfyUI HTTP API 封装，一个请求搞定**文生图**、**图生图**、**局部重绘**、**超分辨率**、**LLLite ControlNet 引导**与**LoRA 下载**。

## 前置准备

本仓库只含 API 服务，**不包含 ComfyUI 本体与任何模型文件**。

### ComfyUI

一个可用的 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 实例，**其 HTTP API 需要对本服务可达**。

### Anima Base 模型

前往 [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima) 下载以下文件并放入指定路径：

| 文件 | 放入 ComfyUI 路径 |
|---|---|
| `anima-base-v1.0.safetensors` | `models/diffusion_models/` |
| `qwen_3_06b_base.safetensors` | `models/text_encoders/` |
| `qwen_image_vae.safetensors` | `models/vae/` |

### 超分辨率模型

前往 [Kim2091/AnimeSharp](https://huggingface.co/Kim2091/AnimeSharp) 下载以下文件并放入指定路径：

| 文件 | 放入 ComfyUI 路径 |
|---|---|
| `4x-AnimeSharp.safetensors` | `models/upscale_models/` |

### ControlNet 模型

前往 [kohya-ss/Anima-LLLite](https://huggingface.co/kohya-ss/Anima-LLLite) 下载以下文件并放入指定路径：

| 文件 | 放入 ComfyUI 路径 |
|---|---|
| `anima-lllite-any-test-like-v2.safetensors` | `models/controlnet` |
| `anima-lllite-inpainting-v1.safetensors` | `models/controlnet` |
| `anima-lllite-depth-1.safetensors` | `models/controlnet` |
| `anima-lllite-lineart-1.safetensors` | `models/controlnet` |
| `anima-lllite-pose-1.safetensors` | `models/controlnet` |

### 自定义节点

前往各项目主页下载并放入 ComfyUI 的 `custom_nodes/` 目录：

| 节点 | 链接 |
|---|---|
| ANIMA_BOOSTER | [BlackSnowSkill/ANIMA_BOOSTER](https://github.com/BlackSnowSkill/ANIMA_BOOSTER) |
| Anima-Artist-Mixer | [An1X3R/Anima-Artist-Mixer](https://github.com/An1X3R/Anima-Artist-Mixer) |
| ComfyUI-Anima-LLLite  | [kohya-ss/ComfyUI-Anima-LLLite](https://github.com/kohya-ss/ComfyUI-Anima-LLLite) |
| ComfyUI-Anima-Remap  | [shin131002/ComfyUI-Anima-Remap](https://github.com/shin131002/ComfyUI-Anima-Remap) |
| ComfyUI-KJNodes | [kijai/ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) |

## 快速开始

### Docker Compose 运行（推荐）

准备配置文件：

```bash
cp .env.example .env
```

请注意配置 `.env` 中容器部署的三个关键项：

```bash
HOST=0.0.0.0                                       # 固定填写 0.0.0.0
COMFYUI_PATH=/app                                  # 固定填写 /app
COMFYUI_BASE_URL=http://host.docker.internal:8188  # 外部 ComfyUI 地址
```

修改 `docker-compose.yaml` 挂载卷，将占位路径换为 ComfyUI 真实路径：

```yaml
    volumes:
      - /path/to/comfyui/models:/app/models
```

启动：

```bash
docker compose up -d --build
```

> 说明：
>
> `host.docker.internal` 仅针对本服务与 ComfyUI 在同一台机器上部署的情况；若 ComfyUI 部署在其他机器上，`COMFYUI_BASE_URL` 直接填写地址即可
>
> 修改 `.env` 后执行 `docker compose restart` 生效

### 源码运行

要求：

- Python ≥ 3.12
- `aria2c` 可用

```bash
uv sync
uv run --no-sync app
```

### 验证

```bash
curl http://127.0.0.1:8189/health
# {"status":"ok","comfyui":"ok"}

# 提交生成任务（count=3 批量三张，每张独立随机种子），拿 task_id
curl -s -X POST http://127.0.0.1:8189/anima/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": {"positive": "masterpiece, best quality, 1girl, solo, long silver hair, miko, shrine, sunset"}, "count": 3}'
# {"task_id":"a1b2c3d4e5f6"}

# 查看任务进度
curl -N http://127.0.0.1:8189/tasks/a1b2c3d4e5f6/events

# 获取生成的图片
curl -s -o out.zip http://127.0.0.1:8189/tasks/a1b2c3d4e5f6/images.zip
```

API 的详细用法（含全部字段约束与语义）可访问 `/docs`（Swagger UI）或 `GET /openapi.json`。

## 开发

```bash
uv sync
uv run ruff check src tests
uv run mypy src tests
uv run pytest
uv run python -m compileall -q src
```

依赖变更后重新锁定：

```bash
uv lock
```

## License

[MIT](LICENSE)
