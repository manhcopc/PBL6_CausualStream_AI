from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ai_host: str = Field(default="127.0.0.1", alias="AI_HOST")
    ai_port: int = Field(default=8001, alias="AI_PORT")
    inference_provider: str = Field(default="mock", alias="INFERENCE_PROVIDER")
    model_checkpoint_path: str | None = Field(default=None, alias="MODEL_CHECKPOINT_PATH")
    model_version: str = Field(default="mock-temporal-v1", alias="MODEL_VERSION")
    session_timeout_seconds: int = Field(default=1800, alias="SESSION_TIMEOUT_SECONDS")
    max_active_sessions: int = Field(default=32, alias="MAX_ACTIVE_SESSIONS")
    max_frame_size_mb: int = Field(default=5, alias="MAX_FRAME_SIZE_MB")
    memory_enabled: bool = Field(default=True, alias="MEMORY_ENABLED")
    memory_k: int = Field(default=2, ge=0, le=16, alias="MEMORY_K")
    sample_frames: int = Field(default=4, ge=1, le=32, alias="SAMPLE_FRAMES")
    mock_latency_ms: int = Field(default=0, ge=0, alias="MOCK_LATENCY_MS")
    mock_fail_after_frames: int | None = Field(default=None, ge=1, alias="MOCK_FAIL_AFTER_FRAMES")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


@lru_cache
def get_settings() -> Settings:
    return Settings()

