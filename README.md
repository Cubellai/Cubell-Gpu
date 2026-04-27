# Cubell GPU Worker

GPU-ready Celery worker for Cubell dubbing jobs.

## Pipeline

The worker runs four model stages:

- Whisper large-v3 for transcription.
- NLLB-200 1.3B for translation.
- StyleTTS 2 for voice generation.
- MuseTalk for lip synchronization.

The Celery task is:

```text
cubell.gpu_worker.process_dubbing_job
```

It accepts a single `job_id`, reads the job from PostgreSQL, updates progress through each stage, and writes the completed result path back to the job row.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```

Run the worker:

```bash
celery -A worker.celery_app.celery_app worker --loglevel=INFO --queues=dubbing
```

## Docker

```bash
docker build -t cubell-gpu-worker .
```

The image expects Redis, PostgreSQL, shared storage, and model script paths to be provided by the deployment environment.

## Model Script Configuration

Whisper and NLLB load directly from Hugging Face. StyleTTS 2 and MuseTalk are invoked through inference scripts because those projects are commonly maintained as separate model repos.

Set:

```bash
STYLE_TTS2_SCRIPT=/path/to/styletts2/inference.py
MUSETALK_SCRIPT=/path/to/musetalk/inference.py
```

Mount or bake the StyleTTS 2 and MuseTalk repos/checkpoints into the worker image before running production inference.
