import json
import logging
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

NLLB_LANGUAGE_CODES = {
    "arabic": "arb_Arab",
    "english": "eng_Latn",
    "french": "fra_Latn",
    "german": "deu_Latn",
    "hindi": "hin_Deva",
    "italian": "ita_Latn",
    "japanese": "jpn_Jpan",
    "korean": "kor_Hang",
    "mandarin": "zho_Hans",
    "chinese": "zho_Hans",
    "portuguese": "por_Latn",
    "spanish": "spa_Latn",
}


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
    """Dubbing pipeline wired to the worker task interface.

    Transcription and translation use real Whisper and NLLB models. Voice
    generation and lip sync remain placeholders until StyleTTS2 and MuseTalk are
    wired into the runtime image.
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
        require_cuda=True,
        command_timeout_seconds=300,
    ):
        self.work_dir = Path(work_dir)
        self.result_dir = Path(result_dir)
        self.style_tts2_script = style_tts2_script
        self.musetalk_script = musetalk_script
        self.whisper_model = whisper_model or "openai/whisper-large-v3"
        self.nllb_model = nllb_model or "facebook/nllb-200-1.3B"
        self.source_language_code = source_language_code or "eng_Latn"
        self.require_cuda = require_cuda
        self.command_timeout_seconds = command_timeout_seconds
        self.device, self.torch_dtype = self._select_device()
        self._transcriber = None
        self._translator_tokenizer = None
        self._translator_model = None

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
        video_path = Path(video_path)
        logger.info("Transcribing %s with %s", video_path, self.whisper_model)
        print(f"Transcribing {video_path}")

        try:
            result = self._get_transcriber()(
                str(video_path),
                chunk_length_s=30,
                stride_length_s=(5, 5),
                return_timestamps=True,
                generate_kwargs={"task": "transcribe"},
            )
        except Exception as exc:
            logger.exception("Whisper transcription failed for %s", video_path)
            raise RuntimeError(f"Whisper transcription failed for {video_path}: {exc}") from exc

        text = str(result.get("text", "")).strip()
        if not text:
            raise RuntimeError(f"Whisper produced an empty transcript for {video_path}.")

        chunks = result.get("chunks") or []
        logger.info("Completed transcription for %s (%d characters)", video_path, len(text))
        return TranscriptionResult(
            text=text,
            chunks=list(chunks),
            detected_language=result.get("language"),
        )

    def translate(self, text, target_language):
        if not text or not str(text).strip():
            raise ValueError("Cannot translate empty transcription text.")

        target_code = self._resolve_target_language(target_language)
        logger.info("Translating transcript to %s with %s", target_language, self.nllb_model)
        print(f"Translating to {target_language}")

        try:
            tokenizer, model = self._get_translator()
            tokenizer.src_lang = self.source_language_code
            text_chunks = self._chunk_text_for_translation(str(text), tokenizer)

            import torch

            translated_chunks = []
            for text_chunk in text_chunks:
                inputs = tokenizer(
                    text_chunk,
                    return_tensors="pt",
                    truncation=True,
                    max_length=900,
                ).to(self.device)

                with torch.inference_mode():
                    generated_tokens = model.generate(
                        **inputs,
                        forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_code),
                        max_new_tokens=900,
                        num_beams=5,
                    )
                translated_chunks.append(
                    tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                )
            translated_text = "\n".join(chunk.strip() for chunk in translated_chunks if chunk.strip())
        except Exception as exc:
            logger.exception("NLLB translation failed for target language %s", target_language)
            raise RuntimeError(f"NLLB translation failed for {target_language}: {exc}") from exc

        translated_text = translated_text.strip()
        if not translated_text:
            raise RuntimeError(f"NLLB produced an empty translation for {target_language}.")

        logger.info("Completed translation to %s (%d characters)", target_language, len(translated_text))
        return translated_text

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

    def _select_device(self):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for Whisper/NLLB inference.") from exc

        if torch.cuda.is_available():
            logger.info("Using CUDA for dubbing model inference")
            return "cuda:0", torch.float16
        if self.require_cuda:
            raise RuntimeError("CUDA is required for dubbing inference but is not available.")

        logger.warning("CUDA is not available; running dubbing model inference on CPU.")
        return "cpu", torch.float32

    def _get_transcriber(self):
        if self._transcriber is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise RuntimeError("transformers is required for Whisper transcription.") from exc

            logger.info("Loading Whisper ASR model: %s", self.whisper_model)
            self._transcriber = pipeline(
                "automatic-speech-recognition",
                model=self.whisper_model,
                torch_dtype=self.torch_dtype,
                device=self.device,
            )
        return self._transcriber

    def _get_translator(self):
        if self._translator_model is None or self._translator_tokenizer is None:
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError("transformers is required for NLLB translation.") from exc

            logger.info("Loading NLLB translation model: %s", self.nllb_model)
            self._translator_tokenizer = AutoTokenizer.from_pretrained(self.nllb_model)
            self._translator_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.nllb_model,
                torch_dtype=self.torch_dtype,
            ).to(self.device)
            self._translator_model.eval()
        return self._translator_tokenizer, self._translator_model

    def _resolve_target_language(self, target_language: str) -> str:
        normalized = str(target_language).strip()
        if not normalized:
            raise ValueError("Target language is required for translation.")
        if "_" in normalized:
            return normalized

        target_code = NLLB_LANGUAGE_CODES.get(normalized.lower())
        if target_code is None:
            raise ValueError(f"No NLLB language code configured for target language: {target_language}")
        return target_code

    @staticmethod
    def _chunk_text_for_translation(text: str, tokenizer, max_tokens: int = 850) -> list[str]:
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
        chunks: list[str] = []
        current: list[str] = []

        def token_count(value: str) -> int:
            return len(tokenizer(value, add_special_tokens=False)["input_ids"])

        for sentence in sentences:
            candidate = " ".join([*current, sentence]).strip()
            if candidate and token_count(candidate) <= max_tokens:
                current.append(sentence)
                continue

            if current:
                chunks.append(" ".join(current))
                current = []

            if token_count(sentence) <= max_tokens:
                current.append(sentence)
                continue

            words = sentence.split()
            word_chunk: list[str] = []
            for word in words:
                candidate = " ".join([*word_chunk, word]).strip()
                if candidate and token_count(candidate) <= max_tokens:
                    word_chunk.append(word)
                    continue

                if word_chunk:
                    chunks.append(" ".join(word_chunk))
                word_chunk = [word]

            if word_chunk:
                chunks.append(" ".join(word_chunk))

        if current:
            chunks.append(" ".join(current))

        return chunks or [text]


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
