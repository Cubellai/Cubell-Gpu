import json
import logging
import os
import re
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from worker.config import get_settings

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("TRANSFORMERS_NO_VISION", "1")

logger = logging.getLogger(__name__)

NLLB_MODEL_NAME = "facebook/nllb-200-distilled-1.3B"
SYNCLABS_API_BASE_URL = "https://api.synclabs.so"
SYNCLABS_LIP_SYNC_ENDPOINT = f"{SYNCLABS_API_BASE_URL}/v1/lip-sync"
SYNCLABS_POLL_INTERVAL_SECONDS = 10
SYNCLABS_TERMINAL_STATUSES = {"COMPLETED", "FAILED", "REJECTED"}
ELEVENLABS_API_BASE_URL = "https://api.elevenlabs.io"

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
    generation uses ElevenLabs, and lip sync uses Sync Labs.
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
        """Run the dubbing pipeline end-to-end for direct callers."""
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

        self.set_job_id(input_video.stem)
        voice_path = Path(self.generate_voice(text=translated_text, language=target_language))
        output_video = Path(self.lip_sync(str(input_video), str(voice_path)))

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

        result = self._get_transcriber()(
            str(video_path),
            chunk_length_s=30,
            stride_length_s=(5, 5),
            return_timestamps=True,
            generate_kwargs={"task": "transcribe"},
        )

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

        translated_text = translated_text.strip()
        if not translated_text:
            raise RuntimeError(f"NLLB produced an empty translation for {target_language}.")

        logger.info("Completed translation to %s (%d characters)", target_language, len(translated_text))
        return translated_text

    def set_job_id(self, job_id: str) -> None:
        self.job_id = self._safe_path_stem(job_id)

    def generate_voice(self, text: str, language: str = "es") -> str:
        if not text or not text.strip():
            raise ValueError("Cannot generate voice from empty text.")

        output_path = self.settings.worker_temp_dir / f"{self.job_id}_voice.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Generating voice using ElevenLabs for job %s in %s", self.job_id, language)
        print(f"Generating voice using ElevenLabs for: {language}")

        voice_id = self._elevenlabs_voice_id()
        audio_bytes = self._generate_voice_with_elevenlabs(
            text=text,
            voice_id=voice_id,
        )
        self._write_elevenlabs_wav(audio_bytes=audio_bytes, output_path=output_path)

        logger.info("ElevenLabs voice generation completed: %s", output_path)
        print("Voice generation completed")

        return str(output_path)

    def _elevenlabs_voice_id(self) -> str:
        reference_audio_path = self.settings.elevenlabs_reference_audio_path
        if reference_audio_path is None:
            return self.settings.elevenlabs_voice_id

        reference_audio_path = Path(reference_audio_path).expanduser()
        if not reference_audio_path.is_file():
            raise FileNotFoundError(
                "ELEVENLABS_REFERENCE_AUDIO_PATH must point to a real audio file. "
                f"Missing: {reference_audio_path}"
            )

        logger.info("Creating ElevenLabs cloned voice from %s", reference_audio_path)
        return self._create_elevenlabs_voice_clone(reference_audio_path)

    def _create_elevenlabs_voice_clone(self, reference_audio_path: Path) -> str:
        import requests

        api_key = self._require_elevenlabs_api_key()
        headers = {"xi-api-key": api_key}
        data = {
            "name": self.settings.elevenlabs_cloned_voice_name,
            "description": "Voice cloned for Cubell dubbing",
        }

        with reference_audio_path.open("rb") as audio_file:
            files = {
                "files": (
                    reference_audio_path.name,
                    audio_file,
                    self._audio_content_type(reference_audio_path),
                )
            }
            response = requests.post(
                f"{ELEVENLABS_API_BASE_URL}/v1/voices/add",
                headers=headers,
                data=data,
                files=files,
                timeout=120,
            )

        if response.status_code >= 400:
            raise RuntimeError(
                "ElevenLabs voice cloning failed "
                f"({response.status_code}): {response.text}"
            )

        payload = self._parse_json_response(response, "ElevenLabs voice cloning")
        voice_id = self._extract_first_string(payload, ("voice_id", "voiceId", "id"))
        if not voice_id:
            raise RuntimeError(f"ElevenLabs voice cloning response did not include voice_id: {payload}")

        logger.info("Created ElevenLabs cloned voice: %s", voice_id)
        return voice_id

    def _generate_voice_with_elevenlabs(self, *, text: str, voice_id: str) -> bytes:
        import requests

        api_key = self._require_elevenlabs_api_key()
        response = requests.post(
            f"{ELEVENLABS_API_BASE_URL}/v1/text-to-speech/{voice_id}",
            headers={
                "xi-api-key": api_key,
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
            },
            json={
                "text": text,
                "model_id": self.settings.elevenlabs_model_id,
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                },
            },
            params={"output_format": "mp3_44100_128"},
            timeout=180,
        )

        if response.status_code >= 400:
            raise RuntimeError(
                "ElevenLabs text-to-speech failed "
                f"({response.status_code}): {response.text}"
            )
        if not response.content:
            raise RuntimeError("ElevenLabs returned an empty audio response.")

        return response.content

    @staticmethod
    def _write_elevenlabs_wav(*, audio_bytes: bytes, output_path: Path) -> None:
        try:
            from pydub import AudioSegment
        except ImportError as exc:
            raise RuntimeError("pydub is required to convert ElevenLabs audio to WAV.") from exc

        try:
            audio = AudioSegment.from_file(BytesIO(audio_bytes), format="mp3")
            audio.export(output_path, format="wav")
        except Exception as exc:
            raise RuntimeError(f"Failed to write ElevenLabs WAV output: {output_path}") from exc

        if not output_path.is_file():
            raise RuntimeError(f"ElevenLabs WAV output was not created: {output_path}")

    @staticmethod
    def _require_elevenlabs_api_key() -> str:
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise RuntimeError("ELEVENLABS_API_KEY must be set to generate voice.")
        return api_key

    @staticmethod
    def _audio_content_type(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".wav":
            return "audio/wav"
        if suffix == ".mp3":
            return "audio/mpeg"
        if suffix == ".m4a":
            return "audio/mp4"
        if suffix == ".ogg":
            return "audio/ogg"
        return "application/octet-stream"

    def lip_sync(self, video_path: str, audio_path: str) -> str:
        """Run Sync Labs lip sync and download the completed video locally."""
        video_file = Path(video_path)
        audio_file = Path(audio_path)
        if not video_file.is_file():
            raise FileNotFoundError(f"Original video not found for Sync Labs lip sync: {video_file}")
        if not audio_file.is_file():
            raise FileNotFoundError(f"Generated audio not found for Sync Labs lip sync: {audio_file}")

        api_key = os.getenv("SYNCLABS_API_KEY")
        if not api_key:
            raise RuntimeError("SYNCLABS_API_KEY must be set to run Sync Labs lip sync.")

        output_video = self.settings.result_dir / f"{self.job_id}.mp4"
        output_video.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting Sync Labs lip sync for job %s: video=%s audio=%s output=%s",
            self.job_id,
            video_file,
            audio_file,
            output_video,
        )
        print(f"Performing Sync Labs lip sync: {video_file} -> {output_video}")

        job_id = self._create_synclabs_lip_sync_job(
            api_key=api_key,
            video_path=video_file,
            audio_path=audio_file,
        )
        output_url = self._poll_synclabs_lip_sync_job(api_key=api_key, job_id=job_id)
        self._download_synclabs_output(output_url=output_url, output_path=output_video)

        if not output_video.is_file():
            raise RuntimeError(f"Sync Labs did not create expected video file: {output_video}")
        logger.info("Generated Sync Labs dubbed video for job %s: %s", self.job_id, output_video)
        return str(output_video)

    def _create_synclabs_lip_sync_job(
        self,
        *,
        api_key: str,
        video_path: Path,
        audio_path: Path,
    ) -> str:
        import requests

        headers = {"x-api-key": api_key}
        data = {
            "model": "lipsync-2",
            "sync_mode": "cut_off",
        }

        # Sync Labs accepts the source media as multipart files on /v1/lip-sync.
        # Keep the files open only for the request lifetime so large media is not
        # loaded fully into Python memory.
        with video_path.open("rb") as video_file, audio_path.open("rb") as audio_file:
            files = {
                "video": (video_path.name, video_file, "video/mp4"),
                "audio": (audio_path.name, audio_file, "audio/wav"),
            }
            response = requests.post(
                SYNCLABS_LIP_SYNC_ENDPOINT,
                headers=headers,
                data=data,
                files=files,
                timeout=120,
            )

        if response.status_code >= 400:
            raise RuntimeError(
                "Sync Labs lip sync request failed "
                f"({response.status_code}): {response.text}"
            )

        payload = self._parse_json_response(response, "Sync Labs lip sync request")
        job_id = self._extract_first_string(payload, ("id", "job_id", "jobId", "generation_id"))
        if not job_id:
            raise RuntimeError(f"Sync Labs response did not include a job id: {payload}")

        logger.info("Sync Labs lip sync job submitted: %s", job_id)
        return job_id

    def _poll_synclabs_lip_sync_job(self, *, api_key: str, job_id: str) -> str:
        import requests

        headers = {"x-api-key": api_key}
        status_url = f"{SYNCLABS_LIP_SYNC_ENDPOINT}/{job_id}"
        deadline = time.monotonic() + self.settings.command_timeout_seconds

        while True:
            response = requests.get(status_url, headers=headers, timeout=60)
            if response.status_code >= 400:
                raise RuntimeError(
                    "Sync Labs status request failed "
                    f"({response.status_code}): {response.text}"
                )

            payload = self._parse_json_response(response, "Sync Labs status request")
            status = (self._extract_first_string(payload, ("status", "state")) or "").upper()
            logger.info("Sync Labs lip sync job %s status: %s", job_id, status or "unknown")

            if status == "COMPLETED":
                output_url = self._extract_synclabs_output_url(payload)
                if not output_url:
                    raise RuntimeError(f"Sync Labs job completed without output URL: {payload}")
                return output_url

            if status in SYNCLABS_TERMINAL_STATUSES:
                error = payload.get("error") or payload.get("message") or payload
                raise RuntimeError(f"Sync Labs lip sync job {job_id} ended with {status}: {error}")

            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Sync Labs lip sync job {job_id} timed out after "
                    f"{self.settings.command_timeout_seconds} seconds."
                )

            time.sleep(SYNCLABS_POLL_INTERVAL_SECONDS)

    @staticmethod
    def _download_synclabs_output(*, output_url: str, output_path: Path) -> None:
        import requests

        with requests.get(output_url, stream=True, timeout=300) as response:
            if response.status_code >= 400:
                raise RuntimeError(
                    "Sync Labs output download failed "
                    f"({response.status_code}): {response.text}"
                )
            with output_path.open("wb") as output_file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output_file.write(chunk)

    @staticmethod
    def _parse_json_response(response, request_name: str) -> dict:
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(f"{request_name} returned non-JSON response: {response.text}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"{request_name} returned unexpected JSON payload: {payload}")
        return payload

    @staticmethod
    def _extract_first_string(payload: dict, keys: tuple[str, ...]) -> str | None:
        for key in keys:
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
        return None

    def _extract_synclabs_output_url(self, payload: dict) -> str | None:
        output_url = self._extract_first_string(
            payload,
            ("output_url", "outputUrl", "video_url", "videoUrl", "url"),
        )
        if output_url:
            return output_url

        output = payload.get("output") or payload.get("result")
        if isinstance(output, dict):
            return self._extract_first_string(
                output,
                ("url", "video_url", "videoUrl", "output_url", "outputUrl"),
            )
        return None

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
            self._disable_transformers_vision_backends()
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
        self._disable_transformers_vision_backends()
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

    @staticmethod
    def _disable_transformers_vision_backends() -> None:
        """Prevent text/audio model loading from importing a broken torchvision."""
        try:
            from transformers.utils import import_utils
        except Exception:
            return

        import_utils._torchvision_available = False
        import_utils._torchvision_version = "unavailable"

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

