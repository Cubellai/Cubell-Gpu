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
    style_tts2_script: Path | None = None
    musetalk_script: Path | None = None
    require_cuda: bool = True
    command_timeout_seconds: int = Field(default=3600, ge=1)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
