FROM python:3.12-slim
RUN useradd -m -u 1000 user
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
USER user
ENV HOME=/home/user \
    PATH=/home/user/.venv/bin:/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    UV_PROJECT_ENVIRONMENT=/home/user/.venv \
    UV_NO_CACHE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app
COPY --chown=user pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY --chown=user . .
RUN uv sync --frozen --no-dev && python -m scripts.predownload
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
