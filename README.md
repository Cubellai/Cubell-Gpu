# Cubell GPU Worker

GPU-ready Celery worker for Cubell dubbing jobs.

## Pipeline

The worker runs four model stages:

- Whisper large-v3 for transcription.
- NLLB-200 1.3B for translation.
- ElevenLabs for voice generation.
- Sync Labs for lip synchronization.

The Celery task is:

```text
cubell.gpu_worker.process_dubbing_job
```

It accepts a single `job_id`, reads the job from PostgreSQL, updates progress through each stage, and writes the completed result path back to the job row.

## Runtime Requirements

- NVIDIA GPU host with the NVIDIA Container Toolkit installed.
- Redis and PostgreSQL reachable from the worker.
- Shared storage mounted at the same paths used by the API for uploaded videos and results, or Cloudflare R2 credentials for object-backed uploads.
- ElevenLabs API key for text-to-speech voice generation.
- Sync Labs API key for lip synchronization.

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

The image expects Redis, PostgreSQL, storage configuration, and API credentials to be provided by the deployment environment.
It runs as an unprivileged `worker` user and defaults to `WORKER_CONCURRENCY=1` to avoid loading multiple large model copies on a single GPU.
The `linux/amd64` platform is required because the pinned CUDA PyTorch wheels are published for NVIDIA Linux hosts.

## Backend Integration

The backend must enqueue this task in production:

```bash
DUBBING_TASK_NAME=cubell.gpu_worker.process_dubbing_job
DUBBING_QUEUE=dubbing
```

The task message contains only the job UUID. The worker reads `original_video_path`, `language`, and progress fields from the shared PostgreSQL `jobs` table. `original_video_path` may be either:

- A local file path, such as `/app/storage/uploads/<file>.mp4`, when the API and worker share a mounted storage volume.
- An R2 object key, such as `uploads/<file>.mp4`, when the API stores uploads in Cloudflare R2.

For R2-backed jobs, the worker downloads `uploads/...` into `WORKER_TEMP_DIR`, writes the local Sync Labs result under `RESULT_DIR`, uploads it back to R2 as `results/<original>-<language>-dubbed.mp4`, and stores that `results/...` key in `job.result_path`.

## Model Configuration

Whisper and NLLB load directly from Hugging Face. ElevenLabs is used for hosted text-to-speech voice generation, and Sync Labs is used for hosted lip synchronization.

Set:

```bash
ELEVENLABS_API_KEY=...
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM
SYNCLABS_API_KEY=...
```

For ElevenLabs voice cloning, set `ELEVENLABS_REFERENCE_AUDIO_PATH` to a local reference audio file. Leave it unset to use `ELEVENLABS_VOICE_ID`.

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
- `COMMAND_TIMEOUT_SECONDS`: Timeout for Sync Labs polling.
- `ELEVENLABS_API_KEY`: ElevenLabs API key used for `/v1/text-to-speech`.
- `ELEVENLABS_VOICE_ID`: Default ElevenLabs voice ID, defaults to `21m00Tcm4TlvDq8ikWAM`.
- `ELEVENLABS_MODEL_ID`: ElevenLabs model ID, defaults to `eleven_multilingual_v2`.
- `ELEVENLABS_REFERENCE_AUDIO_PATH`: Optional local reference audio path for ElevenLabs voice cloning.
- `ELEVENLABS_CLONED_VOICE_NAME`: Optional name for cloned ElevenLabs voices.
- `SYNCLABS_API_KEY`: Sync Labs API key used for `/v1/lip-sync`.
- `R2_BUCKET_NAME`: Cloudflare R2 bucket name. Required for R2-backed jobs.
- `R2_ACCESS_KEY_ID`: Cloudflare R2 access key ID. Required for R2-backed jobs.
- `R2_SECRET_ACCESS_KEY`: Cloudflare R2 secret access key. Required for R2-backed jobs.
- `R2_ENDPOINT_URL`: Cloudflare R2 S3-compatible endpoint URL. Required for R2-backed jobs.
- `R2_ACCOUNT_ID`: Cloudflare account ID. Kept for parity with the backend deployment env.
- `R2_PUBLIC_URL`: Public R2 URL. The worker does not generate frontend URLs, but accepts the env var for parity with the backend.
- `R2_PRESIGNED_URL_EXPIRATION_SECONDS`: Presigned URL lifetime used by the backend. The worker does not currently generate presigned URLs.
