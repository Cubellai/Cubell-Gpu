FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /worker

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential espeak-ng ffmpeg git libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY requirements.txt ./
COPY dubbing_pipeline.py ./
COPY worker ./worker

RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install .

CMD ["celery", "-A", "worker.celery_app.celery_app", "worker", "--loglevel=INFO", "--queues=dubbing"]
