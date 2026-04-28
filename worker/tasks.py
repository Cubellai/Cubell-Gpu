import json
import logging
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

from dubbing_pipeline import DubbingPipeline
from worker.celery_app import celery_app
from worker.config import get_settings
from worker.db import Job, JobStatus, SessionLocal

logger = logging.getLogger(__name__)

TEST_VIDEO_ENV_VARS = (
    "CUBELL_TEST_VIDEO_PATH",
    "TEST_VIDEO_PATH",
    "INPUT_VIDEO_PATH",
    "ORIGINAL_VIDEO_PATH",
)
SUPPORTED_TEST_VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".wav", ".mp3", ".m4a")
COMMON_TEST_VIDEO_NAMES = (
    "input.mp4",
    "test.mp4",
    "sample.mp4",
    "video.mp4",
    "upload.mp4",
)
TEST_VIDEO_SEARCH_ROOTS = (
    Path("/workspace/storage/uploads"),
    Path("/workspace/storage"),
    Path("/workspace"),
    Path("/app/storage/uploads"),
    Path("/app/storage"),
)


def update_progress(
    job: Job,
    message: str,
    percent: int,
    *,
    status: JobStatus = JobStatus.processing,
) -> None:
    job.status = status
    job.progress_message = message
    job.progress_percent = percent
    job.error_message = None
    job.updated_at = datetime.now(UTC)


def create_pipeline(settings) -> DubbingPipeline:
    return DubbingPipeline(
        work_dir=settings.worker_temp_dir,
        result_dir=settings.result_dir,
        style_tts2_script=settings.style_tts2_script,
        musetalk_script=settings.musetalk_script,
        whisper_model=settings.whisper_model,
        nllb_model=settings.nllb_model,
        source_language_code=settings.nllb_source_language_code,
        require_cuda=settings.require_cuda,
        command_timeout_seconds=settings.command_timeout_seconds,
    )


def run_pipeline_steps(
    *,
    pipeline: DubbingPipeline,
    original_video_path: Path,
    job_work_dir: Path,
    target_language: str,
    result_path: Path,
    progress_callback,
) -> None:
    job_work_dir.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline.set_job_id(result_path.stem)

    progress_callback("Transcribing", 10)
    transcription = pipeline.transcribe(original_video_path)
    if transcription is None:
        raise RuntimeError(f"Transcription failed for {original_video_path}: pipeline returned None.")
    if not getattr(transcription, "text", None):
        raise RuntimeError(f"Transcription failed for {original_video_path}: no transcript text returned.")

    transcript_path = job_work_dir / "transcript.json"
    transcript_path.write_text(
        json.dumps(
            {
                "text": transcription.text,
                "chunks": transcription.chunks,
                "detected_language": transcription.detected_language,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    progress_callback("Translating", 35)
    translated_text = pipeline.translate(transcription.text, target_language)
    translation_path = job_work_dir / "translation.txt"
    translation_path.write_text(translated_text, encoding="utf-8")

    progress_callback("Generating voice", 60)
    voice_path = pipeline.generate_voice(
        text=translated_text,
        target_language=target_language,
    )

    progress_callback("Lip sync", 85)
    final_video_path = pipeline.lip_sync(
        original_video_path=original_video_path,
        generated_audio_path=voice_path,
    )
    if final_video_path != result_path:
        raise RuntimeError(f"Lip sync wrote unexpected output path: {final_video_path}")


def resolve_test_video_path(job_id: str) -> Path:
    direct_path = Path(job_id)
    if direct_path.is_file():
        return direct_path

    for env_name in TEST_VIDEO_ENV_VARS:
        value = os.getenv(env_name)
        if value:
            path = Path(value)
            if path.is_file():
                return path
            raise FileNotFoundError(f"{env_name} points to a missing video file: {path}")

    candidates = [
        Path(f"/workspace/storage/uploads/{job_id}.mp4"),
        Path(f"/workspace/storage/{job_id}.mp4"),
        Path(f"/app/storage/uploads/{job_id}.mp4"),
        Path(f"/app/storage/{job_id}.mp4"),
    ]
    candidates.extend(Path.cwd() / name for name in COMMON_TEST_VIDEO_NAMES)
    for path in candidates:
        if path.is_file():
            return path

    discovered_video = find_first_test_video()
    if discovered_video is not None:
        logger.warning("Using discovered test media file for non-UUID job %s: %s", job_id, discovered_video)
        return discovered_video

    candidate_list = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Non-UUID test jobs must provide a real input video for Whisper. "
        f"Pass the video path as job_id, set one of {', '.join(TEST_VIDEO_ENV_VARS)}, "
        f"or place a file at one of: {candidate_list}"
    )


def find_first_test_video() -> Path | None:
    for root in TEST_VIDEO_SEARCH_ROOTS:
        if not root.is_dir():
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED_TEST_VIDEO_SUFFIXES:
                continue
            if should_skip_test_video_path(path):
                continue
            return path

    return None


def should_skip_test_video_path(path: Path) -> bool:
    parts = set(path.parts)
    return bool({".git", "__pycache__", "results", "worker-temp"} & parts)


def run_non_database_job(job_id: str, settings) -> None:
    logger.warning("Running real dubbing pipeline without database row: %s", job_id)
    target_language = os.getenv("CUBELL_TEST_TARGET_LANGUAGE", "Spanish")
    original_video_path = resolve_test_video_path(job_id)
    pipeline = DubbingPipeline(
        work_dir=settings.worker_temp_dir,
        result_dir=settings.result_dir,
        style_tts2_script=settings.style_tts2_script,
        musetalk_script=settings.musetalk_script,
        whisper_model=settings.whisper_model,
        nllb_model=settings.nllb_model,
        source_language_code=settings.nllb_source_language_code,
        require_cuda=settings.require_cuda,
        command_timeout_seconds=settings.command_timeout_seconds,
    )
    safe_job_id = Path(job_id).stem if Path(job_id).suffix else job_id.replace("/", "_")
    job_work_dir = settings.worker_temp_dir / safe_job_id
    result_path = settings.result_dir / f"{safe_job_id}.mp4"
    job_work_dir.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline.set_job_id(result_path.stem)

    def log_progress(message: str, percent: int) -> None:
        logger.info("%s (%d%%) job_id=%s", message, percent, job_id)
        print(f"{message}...")

    log_progress("Transcribing", 10)
    transcription = pipeline.transcribe(original_video_path)
    if transcription is None:
        raise RuntimeError(f"Transcription failed for {original_video_path}: pipeline returned None.")
    if not getattr(transcription, "text", None):
        raise RuntimeError(f"Transcription failed for {original_video_path}: no transcript text returned.")

    transcript_path = job_work_dir / "transcript.json"
    transcript_path.write_text(
        json.dumps(
            {
                "text": transcription.text,
                "chunks": transcription.chunks,
                "detected_language": transcription.detected_language,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    log_progress("Translating", 35)
    translated_text = pipeline.translate(transcription.text, target_language)
    translation_path = job_work_dir / "translation.txt"
    translation_path.write_text(translated_text, encoding="utf-8")

    log_progress("Generating voice", 60)
    voice_path = pipeline.generate_voice(
        text=translated_text,
        target_language=target_language,
    )

    log_progress("Lip sync", 85)
    final_video_path = pipeline.lip_sync(
        original_video_path=original_video_path,
        generated_audio_path=voice_path,
    )
    if final_video_path != result_path:
        raise RuntimeError(f"Lip sync wrote unexpected output path: {final_video_path}")

    print("Completed")
    logger.info("Non-database dubbing job %s completed: %s", job_id, result_path)


@celery_app.task(name="cubell.gpu_worker.process_dubbing_job", bind=True, max_retries=0)
def process_dubbing_job(self, job_id: str) -> None:
    settings = get_settings()

    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        run_non_database_job(job_id, settings)
        return

    with SessionLocal() as db:
        job = db.get(Job, job_uuid)
        if job is None:
            logger.warning("Skipping missing dubbing job %s", job_id)
            return

        update_progress(job, "Preparing pipeline", 5)
        db.add(job)
        db.commit()

        try:
            pipeline = create_pipeline(settings)
            original_video_path = Path(job.original_video_path)
            job_work_dir = settings.worker_temp_dir / str(job.id)
            result_path = settings.result_dir / (
                f"{original_video_path.stem}-{job.language.lower().replace(' ', '-')}-dubbed"
                f"{original_video_path.suffix}"
            )

            def persist_progress(message: str, percent: int) -> None:
                update_progress(job, message, percent)
                db.add(job)
                db.commit()

            run_pipeline_steps(
                pipeline=pipeline,
                original_video_path=original_video_path,
                job_work_dir=job_work_dir,
                target_language=job.language,
                result_path=result_path,
                progress_callback=persist_progress,
            )
        except Exception as exc:
            logger.exception("Dubbing job %s failed", job_id)
            job.status = JobStatus.failed
            job.error_message = str(exc)
            job.progress_message = "Failed"
            job.updated_at = datetime.now(UTC)
            db.add(job)
            db.commit()
            raise

        update_progress(job, "Done", 100, status=JobStatus.completed)
        job.result_path = str(result_path)
        db.add(job)
        db.commit()
        logger.info("Dubbing job %s completed", job_id)
