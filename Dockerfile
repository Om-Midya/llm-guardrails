FROM python:3.12-slim
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    UV_PROJECT_ENVIRONMENT=/home/user/.venv \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY --chown=user pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY --chown=user . .
RUN uv sync --frozen --no-dev && chown -R user:user /home/user
USER user
RUN /home/user/.venv/bin/python -m scripts.predownload
EXPOSE 7860
CMD ["/home/user/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
