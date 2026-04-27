import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

from dubbing_pipeline import DubbingPipeline
from worker.celery_app import celery_app
from worker.config import get_settings
from worker.db import Job, JobStatus, SessionLocal

logger = logging.getLogger(__name__)


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


@celery_app.task(name="cubell.gpu_worker.process_dubbing_job", bind=True, max_retries=0)
def process_dubbing_job(self, job_id: str) -> None:
    settings = get_settings()

    with SessionLocal() as db:
        job = db.get(Job, uuid.UUID(job_id))
        if job is None:
            logger.warning("Skipping missing dubbing job %s", job_id)
            return

        update_progress(job, "Preparing pipeline", 5)
        db.add(job)
        db.commit()

        try:
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
            original_video_path = Path(job.original_video_path)
            job_work_dir = settings.worker_temp_dir / str(job.id)
            job_work_dir.mkdir(parents=True, exist_ok=True)

            update_progress(job, "Transcribing", 10)
            db.add(job)
            db.commit()
            transcription = pipeline.transcribe(original_video_path)
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

            update_progress(job, "Translating", 35)
            db.add(job)
            db.commit()
            translated_text = pipeline.translate(transcription.text, job.language)
            translation_path = job_work_dir / "translation.txt"
            translation_path.write_text(translated_text, encoding="utf-8")

            update_progress(job, "Generating voice", 60)
            db.add(job)
            db.commit()
            voice_path = job_work_dir / "dubbed_voice.wav"
            pipeline.generate_voice(
                text=translated_text,
                reference_video=original_video_path,
                target_language=job.language,
                output_audio=voice_path,
            )

            update_progress(job, "Lip sync", 85)
            db.add(job)
            db.commit()
            result_path = settings.result_dir / (
                f"{original_video_path.stem}-{job.language.lower().replace(' ', '-')}-dubbed"
                f"{original_video_path.suffix}"
            )
            pipeline.lip_sync(
                source_video=original_video_path,
                dubbed_audio=voice_path,
                output_video=result_path,
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
