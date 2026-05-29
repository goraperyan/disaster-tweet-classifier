FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY configs ./configs
COPY disaster_tweet_classifier ./disaster_tweet_classifier

RUN uv sync --frozen --no-dev

COPY artifacts/models/bertweet ./artifacts/models/bertweet

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "disaster_tweet_classifier.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
