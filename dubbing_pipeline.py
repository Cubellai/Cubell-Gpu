import json
import logging
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    chunks: list[dict]
    detected_language: str | None


@dataclass(frozen=True)
class DubbingResult:
    transcript_path: Path
    translation_path: Path
    voice_path: Path
    video_path: Path


class DubbingPipeline:
    """Placeholder dubbing pipeline wired to the worker task interface.

    The methods below keep the same shape as the future Whisper, NLLB, StyleTTS2,
    and MuseTalk integration points so real model calls can replace them later.
    """

    def __init__(
        self,
        work_dir,
        result_dir,
        style_tts2_script,
        musetalk_script,
        whisper_model,
        nllb_model,
        source_language_code,
        require_cuda,
        command_timeout_seconds,
    ):
        self.work_dir = Path(work_dir)
        self.result_dir = Path(result_dir)
        self.style_tts2_script = style_tts2_script
        self.musetalk_script = musetalk_script
        self.whisper_model = whisper_model
        self.nllb_model = nllb_model
        self.source_language_code = source_language_code
        self.require_cuda = require_cuda
        self.command_timeout_seconds = command_timeout_seconds

        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def run(self, input_video: Path, target_language: str) -> DubbingResult:
        """Run the placeholder pipeline end-to-end for direct callers."""
        input_video = Path(input_video)
        job_work_dir = self.work_dir / input_video.stem
        job_work_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)

        transcription = self.transcribe(input_video)
        transcript_path = job_work_dir / "transcript.json"
        transcript_path.write_text(
            json.dumps(
                {
                    "text": transcription.text,
                    "chunks": transcription.chunks,
                    "detected_language": transcription.detected_language,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        translated_text = self.translate(transcription.text, target_language)
        translation_path = job_work_dir / "translation.txt"
        translation_path.write_text(translated_text, encoding="utf-8")

        voice_path = job_work_dir / "dubbed_voice.wav"
        self.generate_voice(
            text=translated_text,
            reference_video=input_video,
            target_language=target_language,
            output_audio=voice_path,
        )

        output_video = self.result_dir / (
            f"{input_video.stem}-{target_language.lower().replace(' ', '-')}-dubbed"
            f"{input_video.suffix}"
        )
        self.lip_sync(
            source_video=input_video,
            dubbed_audio=voice_path,
            output_video=output_video,
        )

        return DubbingResult(
            transcript_path=transcript_path,
            translation_path=translation_path,
            voice_path=voice_path,
            video_path=output_video,
        )

    def transcribe(self, video_path):
        print(f"Transcribing {video_path}")
        return TranscriptionResult(
            text="This is a placeholder transcription.",
            chunks=[],
            detected_language="eng",
        )

    def translate(self, text, target_language):
        print(f"Translating to {target_language}")
        return "This is a placeholder translated text."

    def generate_voice(self, text, reference_video, target_language, output_audio):
        print(f"Generating voice for {target_language}")
        output_audio = Path(output_audio)
        output_audio.parent.mkdir(parents=True, exist_ok=True)

        # Placeholder for StyleTTS2: write a dummy audio artifact for the worker.
        with open(output_audio, "wb") as f:
            f.write(b"dummy audio content")

    def lip_sync(self, source_video, dubbed_audio, output_video):
        print(f"Performing lip sync: {source_video} -> {output_video}")
        output_video = Path(output_video)
        output_video.parent.mkdir(parents=True, exist_ok=True)

        # Placeholder for MuseTalk: copy the original video as the final result.
        shutil.copy2(source_video, output_video)


def process_dubbing_job(job_id: str) -> str:
    """Run a placeholder dubbing job and return the final output video path."""
    from worker.db import Job, JobStatus, SessionLocal

    output_dir = Path("/workspace/storage/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = output_dir / f"{job_id}.mp4"
    job = None
    db = None

    def log_progress(message: str) -> None:
        logger.info("%s job_id=%s", message, job_id)
        print(message)

    def save_progress(message: str, percent: int, *, status: JobStatus = JobStatus.processing) -> None:
        if job is None or db is None:
            return
        job.status = status
        job.progress_message = message
        job.progress_percent = percent
        job.error_message = None
        job.updated_at = datetime.now(UTC)
        db.add(job)
        db.commit()

    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        job_uuid = None
        logger.warning("Skipping database lookup for non-UUID dubbing job id: %s", job_id)

    if job_uuid is not None:
        db = SessionLocal()
        job = db.get(Job, job_uuid)
        if job is None:
            logger.warning("Dubbing job not found in database; running placeholders only: %s", job_id)
        else:
            save_progress("Starting", 0)

    try:
        steps = [
            ("Transcribing...", 20),
            ("Translating...", 40),
            ("Generating voice...", 65),
            ("Performing lip sync...", 90),
        ]

        for message, percent in steps:
            log_progress(message)
            save_progress(message, percent)

        log_progress("Completed")
        if job is not None and db is not None:
            job.result_path = str(output_video_path)
            save_progress("Completed", 100, status=JobStatus.completed)
    finally:
        if db is not None:
            db.close()

    return str(output_video_path)
