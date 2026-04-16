"""Runtime configuration from environment."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openrouter_api_key: str = Field(default="", validation_alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        validation_alias="OPENROUTER_BASE_URL",
    )
    openrouter_model_cheap: str = Field(
        default="openai/gpt-4o-mini",
        validation_alias="OPENROUTER_MODEL_CHEAP",
    )
    openrouter_model_strong: str = Field(
        default="openai/gpt-4o",
        validation_alias="OPENROUTER_MODEL_STRONG",
    )
    openrouter_model_transcription: str = Field(
        default="openai/gpt-4o-mini",
        validation_alias="OPENROUTER_MODEL_TRANSCRIPTION",
    )
    openrouter_temperature: float = Field(default=0.15, validation_alias="OPENROUTER_TEMPERATURE")
    openrouter_timeout_s: float = Field(default=90.0, validation_alias="OPENROUTER_TIMEOUT_S")
    openrouter_max_retries: int = Field(default=3, validation_alias="OPENROUTER_MAX_RETRIES")

    langfuse_public_key: str = Field(default="", validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", validation_alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(default="https://cloud.langfuse.com", validation_alias="LANGFUSE_HOST")
    team_name: str = Field(default="reply-agent", validation_alias="TEAM_NAME")
    langfuse_media_upload_enabled: bool = Field(
        default=False,
        validation_alias="LANGFUSE_MEDIA_UPLOAD_ENABLED",
    )

    enable_audio: bool = Field(default=False, validation_alias="ENABLE_AUDIO")
    enable_audio_transcription: bool = Field(default=False, validation_alias="ENABLE_AUDIO_TRANSCRIPTION")
    audio_transcription_max_bytes: int = Field(
        default=1_200_000,
        validation_alias="AUDIO_TRANSCRIPTION_MAX_BYTES",
    )
    audio_transcription_max_files: int = Field(
        default=2,
        validation_alias="AUDIO_TRANSCRIPTION_MAX_FILES",
    )
    llm_parallel_workers: int = Field(default=4, validation_alias="LLM_PARALLEL_WORKERS")

    fusion_weight_fraud: float = Field(default=0.55, validation_alias="FUSION_WEIGHT_FRAUD")
    fusion_weight_economic: float = Field(default=0.35, validation_alias="FUSION_WEIGHT_ECONOMIC")
    fusion_weight_cluster: float = Field(default=0.10, validation_alias="FUSION_WEIGHT_CLUSTER")

    data_cache_dir: Path = Field(default=Path("data_cache"), validation_alias="DATA_CACHE_DIR")


def load_settings() -> Settings:
    return Settings()


def is_llm_configured(settings: Settings) -> bool:
    return bool(settings.openrouter_api_key.strip())


def is_langfuse_configured(settings: Settings) -> bool:
    return bool(settings.langfuse_public_key.strip() and settings.langfuse_secret_key.strip())


ReviewPriority = Literal["low", "medium", "high"]
