from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
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
    elevenlabs_api_key: str | None = None
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    elevenlabs_model_id: str = "eleven_multilingual_v2"
    elevenlabs_reference_audio_path: Path | None = None
    elevenlabs_cloned_voice_name: str = "Cubell cloned voice"
    elevenlabs_reference_audio_seconds: int = Field(default=60, ge=1)
    sync_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("SYNC_API_KEY", "SYNCLABS_API_KEY"),
    )
    sync_model: Literal["lipsync-2", "lipsync-2-pro", "lipsync-1.9.0-beta", "sync-3", "react-1"] = "lipsync-2"
    sync_poll_interval_seconds: int = Field(default=5, ge=1)
    sync_poll_timeout_seconds: int = Field(default=900, ge=1)
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

    @field_validator("elevenlabs_reference_audio_path", mode="before")
    @classmethod
    def empty_reference_audio_path_to_none(cls, value):
        if isinstance(value, str) and not value.strip():
            return None
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
