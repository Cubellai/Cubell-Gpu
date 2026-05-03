# Cubell GPU Worker

GPU-ready Celery worker for Cubell dubbing jobs.

## Pipeline

The worker runs four model stages:

- Whisper large-v3 for transcription.
- NLLB-200 1.3B for translation.
- ElevenLabs for voice generation.
- Sync.so for lip synchronization.

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
- Sync.so API key for lip synchronization.

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
./start_worker.sh
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

For R2-backed jobs, the worker downloads `uploads/...` into `WORKER_TEMP_DIR`, writes the local Sync.so result under `RESULT_DIR`, uploads it back to R2 as `results/<original>-<language>-dubbed.mp4`, and stores that `results/...` key in `job.result_path`.

## Model Configuration

Whisper and NLLB load directly from Hugging Face. ElevenLabs is used for hosted text-to-speech voice generation with per-upload voice cloning, and Sync.so is used for hosted lip synchronization.

Set:

```bash
ELEVENLABS_API_KEY=...
ELEVENLABS_MODEL_ID=eleven_multilingual_v2
ELEVENLABS_REFERENCE_AUDIO_SECONDS=60
SYNC_API_KEY=...
SYNC_MODEL=lipsync-2
```

For production voice cloning, leave `ELEVENLABS_REFERENCE_AUDIO_PATH` unset. The worker extracts a short WAV sample from each uploaded video, creates a temporary ElevenLabs cloned voice for that job, uses it for the translated speech, and then attempts to delete the temporary voice. Set `ELEVENLABS_REFERENCE_AUDIO_PATH` only when you want every job to use the same local test reference audio file.

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
- `COMMAND_TIMEOUT_SECONDS`: General command timeout for long-running stages.
- `ELEVENLABS_API_KEY`: ElevenLabs API key used for voice cloning and `/v1/text-to-speech`.
- `ELEVENLABS_VOICE_ID`: Fallback ElevenLabs voice ID, used only when no reference video/audio is available.
- `ELEVENLABS_MODEL_ID`: ElevenLabs model ID, defaults to `eleven_multilingual_v2`.
- `ELEVENLABS_REFERENCE_AUDIO_PATH`: Optional static local reference audio path for testing. Leave unset to clone from each uploaded video.
- `ELEVENLABS_CLONED_VOICE_NAME`: Optional name for cloned ElevenLabs voices.
- `ELEVENLABS_REFERENCE_AUDIO_SECONDS`: Number of seconds extracted from the uploaded video for voice cloning, defaults to `60`.
- `SYNC_API_KEY`: Sync.so API key used for `/v2/generate`. `SYNCLABS_API_KEY` is also accepted as a backwards-compatible fallback.
- `SYNC_MODEL`: Sync.so generation model, defaults to `lipsync-2`.
- `SYNC_POLL_INTERVAL_SECONDS`: Sync.so polling interval, defaults to `5`.
- `SYNC_POLL_TIMEOUT_SECONDS`: Sync.so polling timeout, defaults to `900`.
- `R2_BUCKET_NAME`: Cloudflare R2 bucket name. Required for R2-backed jobs.
- `R2_ACCESS_KEY_ID`: Cloudflare R2 access key ID. Required for R2-backed jobs.
- `R2_SECRET_ACCESS_KEY`: Cloudflare R2 secret access key. Required for R2-backed jobs.
- `R2_ENDPOINT_URL`: Cloudflare R2 S3-compatible endpoint URL. Required for R2-backed jobs.
- `R2_ACCOUNT_ID`: Cloudflare account ID. Kept for parity with the backend deployment env.
- `R2_PUBLIC_URL`: Public R2 URL. Used to pass temporary video/audio URLs to Sync.so. If unset, the worker falls back to presigned URLs.
- `R2_PRESIGNED_URL_EXPIRATION_SECONDS`: Presigned URL lifetime used when `R2_PUBLIC_URL` is not set.
