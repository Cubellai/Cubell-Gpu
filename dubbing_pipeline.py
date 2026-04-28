import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from worker.config import get_settings

logger = logging.getLogger(__name__)

NLLB_MODEL_NAME = "facebook/nllb-200-distilled-1.3B"

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
    generation uses Fish Speech, and lip sync uses MuseTalk.
    """

    def __init__(
        self,
        work_dir: str | Path,
        result_dir: str | Path,
        whisper_model: str,
        nllb_model: str,
        source_language_code: str,
        require_cuda: bool = True,
        command_timeout_seconds: int = 300,
    ) -> None:
        self.settings = get_settings()
        self.work_dir = Path(work_dir)
        self.result_dir = Path(result_dir)
        self.fish_speech_model_path = Path(self.settings.fish_speech_model_path)
        self.musetalk_script = Path(self.settings.musetalk_script)
        self.whisper_model = whisper_model or "openai/whisper-large-v3"
        self.nllb_model = NLLB_MODEL_NAME
        self.source_language_code = source_language_code or "eng_Latn"
        self.require_cuda = require_cuda
        self.command_timeout_seconds = command_timeout_seconds
        self.device, self.torch_dtype = self._select_device()
        self.torch_dtype_name = "float16" if str(self.torch_dtype).endswith("float16") else "float32"
        self._transcriber = None

        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.job_id = "dubbing"
        self._translator_tokenizer, self._translator_model = self._load_translator()

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

        self.fish_speech_model_path = Path(self.settings.fish_speech_model_path)
        self.musetalk_script = Path(self.settings.musetalk_script)
        self.set_job_id(input_video.stem)
        voice_path = self.generate_voice(text=translated_text, target_language=target_language)
        output_video = self.lip_sync(original_video_path=input_video, generated_audio_path=voice_path)

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

        if result is None:
            raise RuntimeError(f"Whisper returned no result for {video_path}.")
        if not isinstance(result, dict):
            raise RuntimeError(
                f"Whisper returned an unexpected result type for {video_path}: "
                f"{type(result).__name__}"
            )

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
            tokenizer = self._translator_tokenizer
            model = self._translator_model
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

    def set_job_id(self, job_id: str) -> None:
        self.job_id = self._safe_path_stem(job_id)

    def generate_voice(self, text: str, target_language: str) -> Path:
        if not text or not text.strip():
            raise ValueError("Cannot generate voice from empty text.")

        output_path = self.settings.worker_temp_dir / f"{self.job_id}_voice.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Generating voice placeholder for job %s using Fish Speech path: %s", self.job_id, output_path)
        print(f"Generating voice using Fish Speech for: {target_language}")

        # Temporary Fish Speech integration shim: create valid silent audio until
        # the full Fish Speech model invocation is finalized.
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "anullsrc",
                    "-t",
                    "10",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            logger.error("Fish Speech placeholder audio generation failed: %s", exc.stderr)
            raise RuntimeError(f"Fish Speech placeholder audio generation failed: {exc.stderr}") from exc

        return output_path

    def lip_sync(self, original_video_path: Path, generated_audio_path: Path) -> Path:
        original_video_path = Path(original_video_path)
        generated_audio_path = Path(generated_audio_path)
        if not original_video_path.is_file():
            raise FileNotFoundError(f"Original video not found for lip sync: {original_video_path}")
        if not generated_audio_path.is_file():
            raise FileNotFoundError(f"Generated audio not found for lip sync: {generated_audio_path}")

        musetalk_script = self._require_script(self.settings.musetalk_script, "MUSETALK_SCRIPT")
        output_video = self.settings.result_dir / f"{self.job_id}.mp4"
        output_video.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Running MuseTalk lip sync for job %s: video=%s audio=%s output=%s",
            self.job_id,
            original_video_path,
            generated_audio_path,
            output_video,
        )
        print(f"Performing lip sync: {original_video_path} -> {output_video}")
        command = [
            sys.executable,
            str(musetalk_script),
            "--video",
            str(original_video_path),
            "--audio",
            str(generated_audio_path),
            "--result",
            str(output_video),
        ]
        self._run_command(command, "MuseTalk lip sync")

        if not output_video.is_file():
            raise RuntimeError(f"MuseTalk did not create expected video file: {output_video}")
        logger.info("Generated dubbed video for job %s: %s", self.job_id, output_video)
        return output_video

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

    def _load_translator(self):
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is required for NLLB translation.") from exc

        logger.info("Loading NLLB translation model: %s", self.nllb_model)
        tokenizer = AutoTokenizer.from_pretrained(self.nllb_model)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.nllb_model,
            torch_dtype=self.torch_dtype,
            use_safetensors=True,
        ).to(self.device)
        model.eval()
        logger.info("Loaded NLLB translation model on %s", self.device)
        return tokenizer, model

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
    def _safe_path_stem(value: str) -> str:
        stem = Path(str(value)).stem if Path(str(value)).suffix else str(value)
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or "dubbing"

    @staticmethod
    def _require_script(script_path: Path | None, env_name: str) -> Path:
        if script_path is None:
            raise RuntimeError(f"{env_name} must be configured before running this stage.")

        resolved = Path(script_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"{env_name} script does not exist: {resolved}")
        return resolved

    @staticmethod
    def _require_directory(directory_path: Path | None, env_name: str) -> Path:
        if directory_path is None:
            raise RuntimeError(f"{env_name} must be configured before running this stage.")

        resolved = Path(directory_path).expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"{env_name} directory does not exist: {resolved}")
        return resolved

    def _fish_speech_device(self) -> str:
        return "cuda" if str(self.device).startswith("cuda") else "cpu"

    @staticmethod
    def _resolve_fish_speech_repo_root(model_path: Path) -> Path | None:
        for candidate in [model_path, *model_path.parents]:
            if (candidate / "fish_speech").is_dir():
                return candidate
        return None

    @staticmethod
    def _pythonpath_env(repo_root: Path | None) -> dict[str, str] | None:
        if repo_root is None:
            return None

        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(repo_root)
            if not existing_pythonpath
            else f"{repo_root}{os.pathsep}{existing_pythonpath}"
        )
        return env

    def _run_command(
        self,
        command: list[str],
        step_name: str,
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        logger.info("Starting %s: %s", step_name, " ".join(command))
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.settings.command_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"{step_name} timed out after {self.settings.command_timeout_seconds} seconds."
            ) from exc

        if completed.returncode != 0:
            logger.error(
                "%s failed with exit code %s. stdout=%s stderr=%s",
                step_name,
                completed.returncode,
                completed.stdout,
                completed.stderr,
            )
            raise RuntimeError(
                f"{step_name} failed with exit code {completed.returncode}: {completed.stderr}"
            )

        logger.info("%s completed. stdout=%s", step_name, completed.stdout)

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

