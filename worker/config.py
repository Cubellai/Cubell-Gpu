from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://lipdub:lipdub@postgres:5432/lipdub"
    redis_url: str = "redis://localhost:6379/0"
    dubbing_queue: str = "dubbing"
    result_dir: Path = Path("/workspace/storage/results")
    worker_temp_dir: Path = Path("/workspace/storage/worker-temp")
    whisper_model: str = "openai/whisper-large-v3"
    nllb_model: str = "facebook/nllb-200-1.3B"
    nllb_source_language_code: str = "eng_Latn"
    fish_speech_model_path: Path = Path("/workspace/fish-speech/checkpoints/s2-pro")
    fish_speech_prompt_text: str | None = None
    fish_speech_prompt_tokens_path: Path | None = None
    musetalk_script: Path = Path("/workspace/musetalk/inference.sh")
    require_cuda: bool = True
    command_timeout_seconds: int = Field(default=3600, ge=1)
    r2_bucket_name: str | None = None
    r2_access_key_id: str | None = None
    r2_secret_access_key: str | None = None
    r2_account_id: str | None = None
    r2_endpoint_url: str | None = None
    r2_public_url: str | None = None
    r2_presigned_url_expiration_seconds: int = Field(default=3600, ge=1)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
