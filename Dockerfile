FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/storage/model-cache \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /worker

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential espeak-ng ffmpeg git libsndfile1 tini \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY requirements.txt ./
COPY dubbing_pipeline.py ./
COPY worker ./worker

RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install .

RUN useradd --create-home --uid 10001 worker \
    && mkdir -p /app/storage/results /app/storage/worker-temp /app/storage/model-cache \
    && chown -R worker:worker /app/storage /worker

USER worker

ENTRYPOINT ["tini", "--"]
CMD ["sh", "-c", "celery -A worker.celery_app.celery_app worker --loglevel=${LOG_LEVEL:-INFO} --queues=${DUBBING_QUEUE:-dubbing} --concurrency=${WORKER_CONCURRENCY:-1} --prefetch-multiplier=1"]
