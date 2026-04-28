# Cubell GPU Worker

GPU-ready Celery worker for Cubell dubbing jobs.

## Pipeline

The worker runs four model stages:

- Whisper large-v3 for transcription.
- NLLB-200 1.3B for translation.
- Fish Speech for voice generation.
- MuseTalk for lip synchronization.

The Celery task is:

```text
cubell.gpu_worker.process_dubbing_job
```

It accepts a single `job_id`, reads the job from PostgreSQL, updates progress through each stage, and writes the completed result path back to the job row.

## Runtime Requirements

- NVIDIA GPU host with the NVIDIA Container Toolkit installed.
- Redis and PostgreSQL reachable from the worker.
- Shared storage mounted at the same paths used by the API for uploaded videos and results.
- Fish Speech and MuseTalk repositories/checkpoints baked into the image or mounted into the container.

The worker fails jobs early when required GPU or model script configuration is missing. Set
`REQUIRE_CUDA=false` only for CPU development experiments; production should keep it enabled.

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
docker build --platform linux/amd64 -t cubell-gpu-worker .
docker run --rm --env-file .env --gpus all \
  -v "$PWD/../storage:/app/storage" \
  cubell-gpu-worker
```

The image expects Redis, PostgreSQL, shared storage, and model script paths to be provided by the deployment environment.
It runs as an unprivileged `worker` user and defaults to `WORKER_CONCURRENCY=1` to avoid loading multiple large model copies on a single GPU.
The `linux/amd64` platform is required because the pinned CUDA PyTorch wheels are published for NVIDIA Linux hosts.

## Model Script Configuration

Whisper and NLLB load directly from Hugging Face. Fish Speech is invoked through its Python package/CLI with local checkpoints, and MuseTalk is invoked through its inference script.

Set:

```bash
FISH_SPEECH_MODEL_PATH=/workspace/fish-speech/checkpoints/s2-pro
MUSETALK_SCRIPT=/workspace/musetalk/inference.sh
```

Mount or bake the Fish Speech and MuseTalk repos/checkpoints into the worker image before running production inference.

## Configuration

- `DATABASE_URL`: SQLAlchemy database URL for the backend `jobs` table.
- `REDIS_URL`: Celery broker/result backend URL.
- `DUBBING_QUEUE`: Celery queue name, defaults to `dubbing`.
- `RESULT_DIR`: Directory where completed videos are written.
- `WORKER_TEMP_DIR`: Directory for transcripts, translations, and intermediate audio.
- `HF_HOME`: Hugging Face model cache directory.
- `WHISPER_MODEL`: Whisper model ID, defaults to `openai/whisper-large-v3`.
- `NLLB_MODEL`: NLLB model ID, defaults to `facebook/nllb-200-1.3B`.
- `NLLB_SOURCE_LANGUAGE_CODE`: Source language code for NLLB, defaults to `eng_Latn`.
- `REQUIRE_CUDA`: Keep `true` in production so jobs do not silently run on CPU.
- `COMMAND_TIMEOUT_SECONDS`: Timeout for Fish Speech and MuseTalk subprocesses.
- `FISH_SPEECH_MODEL_PATH`: Path inside the container to the Fish Speech checkpoint directory.
- `MUSETALK_SCRIPT`: Path inside the container to the MuseTalk inference script.
