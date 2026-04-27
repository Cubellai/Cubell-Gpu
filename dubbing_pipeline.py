import json
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    pipeline,
)

logger = logging.getLogger(__name__)

NLLB_LANGUAGE_CODES = {
    "Arabic": "arb_Arab",
    "English": "eng_Latn",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Hindi": "hin_Deva",
    "Italian": "ita_Latn",
    "Japanese": "jpn_Jpan",
    "Korean": "kor_Hang",
    "Mandarin": "zho_Hans",
    "Portuguese": "por_Latn",
    "Spanish": "spa_Latn",
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


class PipelineConfigurationError(RuntimeError):
    pass


class DubbingPipeline:
    """High-quality GPU dubbing pipeline built around Whisper, NLLB, StyleTTS 2, and MuseTalk."""

    def __init__(
        self,
        *,
        work_dir: Path,
        result_dir: Path,
        style_tts2_script: Path | None,
        musetalk_script: Path | None,
        whisper_model: str = "openai/whisper-large-v3",
        nllb_model: str = "facebook/nllb-200-1.3B",
        source_language_code: str = "eng_Latn",
        require_cuda: bool = True,
        command_timeout_seconds: int = 3600,
    ) -> None:
        self.work_dir = work_dir
        self.result_dir = result_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.style_tts2_script = self._validate_script(style_tts2_script, "STYLE_TTS2_SCRIPT")
        self.musetalk_script = self._validate_script(musetalk_script, "MUSETALK_SCRIPT")
        self.whisper_model_name = whisper_model
        self.nllb_model_name = nllb_model
        self.source_language_code = source_language_code
        self.command_timeout_seconds = command_timeout_seconds
        self.device = self._select_device(require_cuda=require_cuda)
        self.torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self._transcriber = None
        self._translator_tokenizer = None
        self._translator_model = None

    @staticmethod
    def _validate_script(script_path: Path | None, env_name: str) -> Path:
        if script_path is None:
            raise PipelineConfigurationError(f"{env_name} must point to an inference script.")
        resolved = script_path.expanduser().resolve()
        if not resolved.is_file():
            raise PipelineConfigurationError(f"{env_name} does not exist or is not a file: {resolved}")
        return resolved

    @staticmethod
    def _select_device(*, require_cuda: bool) -> torch.device:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            return torch.device("cuda")
        if require_cuda:
            raise PipelineConfigurationError("CUDA is required but is not available to PyTorch.")
        logger.warning("CUDA is not available; running the dubbing pipeline on CPU.")
        return torch.device("cpu")

    def run(self, input_video: Path, target_language: str) -> DubbingResult:
        input_video = input_video.resolve()
        if not input_video.is_file():
            raise FileNotFoundError(f"Input video not found: {input_video}")

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
                ensure_ascii=False,
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
            f"{input_video.stem}-{target_language.lower().replace(' ', '-')}-dubbed{input_video.suffix}"
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

    def transcribe(self, input_video: Path) -> TranscriptionResult:
        logger.info("Loading Whisper model %s", self.whisper_model_name)
        transcriber = self._get_transcriber()
        logger.info("Transcribing %s with Whisper large-v3", input_video)
        result = transcriber(
            str(input_video),
            chunk_length_s=30,
            stride_length_s=(5, 5),
            batch_size=8 if self.device.type == "cuda" else 1,
            return_timestamps=True,
            generate_kwargs={
                "task": "transcribe",
                "num_beams": 5,
                "temperature": 0.0,
                "compression_ratio_threshold": 1.35,
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.6,
            },
        )
        text = str(result.get("text", "")).strip()
        if not text:
            raise RuntimeError("Whisper produced an empty transcript.")
        return TranscriptionResult(
            text=text,
            chunks=list(result.get("chunks") or []),
            detected_language=result.get("language"),
        )

    def translate(self, text: str, target_language: str) -> str:
        target_code = NLLB_LANGUAGE_CODES.get(target_language)
        if target_code is None:
            raise PipelineConfigurationError(f"No NLLB language code configured for {target_language}.")

        tokenizer, model = self._get_translator()
        tokenizer.src_lang = self.source_language_code
        chunks = self._chunk_text_for_translation(text, tokenizer)

        logger.info("Translating transcript to %s with NLLB-200 1.3B", target_language)
        translated_chunks = []
        for chunk in chunks:
            encoded = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=900).to(
                self.device
            )
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_code),
                    max_new_tokens=900,
                    num_beams=5,
                    length_penalty=1.0,
                    no_repeat_ngram_size=4,
                )
            translated_chunks.append(tokenizer.batch_decode(generated, skip_special_tokens=True)[0])

        translated_text = "\n".join(chunk.strip() for chunk in translated_chunks if chunk.strip())
        if not translated_text:
            raise RuntimeError("NLLB produced an empty translation.")
        return translated_text

    def generate_voice(
        self,
        *,
        text: str,
        reference_video: Path,
        target_language: str,
        output_audio: Path,
    ) -> None:
        command = [
            "python",
            str(self.style_tts2_script),
            "--text",
            text,
            "--reference_video",
            str(reference_video),
            "--language",
            target_language,
            "--output",
            str(output_audio),
            "--diffusion_steps",
            os.getenv("STYLE_TTS2_DIFFUSION_STEPS", "10"),
            "--alpha",
            os.getenv("STYLE_TTS2_ALPHA", "0.3"),
            "--beta",
            os.getenv("STYLE_TTS2_BETA", "0.7"),
            "--embedding_scale",
            os.getenv("STYLE_TTS2_EMBEDDING_SCALE", "1.0"),
        ]
        self._run_command(
            command,
            "StyleTTS 2 voice generation",
            timeout_seconds=self.command_timeout_seconds,
        )
        if not output_audio.is_file():
            raise RuntimeError(f"StyleTTS 2 did not create expected audio: {output_audio}")

    def lip_sync(self, *, source_video: Path, dubbed_audio: Path, output_video: Path) -> None:
        command = [
            "python",
            str(self.musetalk_script),
            "--video",
            str(source_video),
            "--audio",
            str(dubbed_audio),
            "--result",
            str(output_video),
            "--fps",
            os.getenv("MUSETALK_FPS", "25"),
            "--batch_size",
            os.getenv("MUSETALK_BATCH_SIZE", "8"),
        ]
        self._run_command(
            command,
            "MuseTalk lip synchronization",
            timeout_seconds=self.command_timeout_seconds,
        )
        if not output_video.is_file():
            raise RuntimeError(f"MuseTalk did not create expected video: {output_video}")

    def _get_transcriber(self):
        if self._transcriber is None:
            processor = WhisperProcessor.from_pretrained(self.whisper_model_name)
            model = WhisperForConditionalGeneration.from_pretrained(
                self.whisper_model_name,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            ).to(self.device)
            self._transcriber = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=self.torch_dtype,
                device=0 if self.device.type == "cuda" else -1,
            )
        return self._transcriber

    def _get_translator(self):
        if self._translator_model is None or self._translator_tokenizer is None:
            self._translator_tokenizer = AutoTokenizer.from_pretrained(self.nllb_model_name)
            self._translator_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.nllb_model_name,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            ).to(self.device)
            self._translator_model.eval()
        return self._translator_tokenizer, self._translator_model

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
                else:
                    if word_chunk:
                        chunks.append(" ".join(word_chunk))
                    word_chunk = [word]
            if word_chunk:
                chunks.append(" ".join(word_chunk))

        if current:
            chunks.append(" ".join(current))
        return chunks or [text]

    @staticmethod
    def _run_command(command: list[str], step_name: str, *, timeout_seconds: int) -> None:
        logger.info("Starting %s", step_name)
        with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as stderr_file:
            try:
                completed = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=stderr_file,
                    text=True,
                    check=False,
                    timeout=timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"{step_name} timed out after {timeout_seconds} seconds.") from exc
            stderr_file.seek(0)
            stderr = stderr_file.read()

        if completed.returncode != 0:
            logger.error("%s failed. stdout=%s stderr=%s", step_name, completed.stdout, stderr)
            raise RuntimeError(f"{step_name} failed with exit code {completed.returncode}: {stderr}")
        logger.info("%s completed", step_name)
