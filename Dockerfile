FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends aria2 curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv==0.13.5

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8189}/health" || exit 1

EXPOSE 8189

CMD ["uv", "run", "--no-sync", "app"]

