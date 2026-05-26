from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

AllowedMimeType = Literal["image/jpeg", "image/png", "image/webp", "application/pdf"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PASSPORT_OCR_",
        env_file=".env",
        extra="ignore",
    )

    app_name: str = "passport-ocr-api"
    log_level: str = "INFO"
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1, le=25 * 1024 * 1024)
    max_pdf_pages: int = Field(default=2, ge=1, le=5)
    max_render_dpi: int = Field(default=220, ge=120, le=300)
    max_raw_snippets: int = Field(default=8, ge=0, le=20)
    max_raw_snippet_chars: int = Field(default=240, ge=40, le=1000)
    local_ocr_timeout_seconds: int = Field(default=30, ge=1, le=90)
    google_ocr_timeout_seconds: int = Field(default=15, ge=1, le=60)
    google_max_retries: int = Field(default=1, ge=0, le=2)
    google_fallback_enabled: bool = False
    low_confidence_threshold: float = Field(default=0.72, ge=0.0, le=1.0)
    circuit_breaker_failure_threshold: int = Field(default=3, ge=1, le=10)
    circuit_breaker_reset_seconds: int = Field(default=60, ge=1, le=600)
    allowed_mime_types: tuple[AllowedMimeType, ...] = (
        "image/jpeg",
        "image/png",
        "image/webp",
        "application/pdf",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        normalized = value.upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalized not in allowed:
            msg = f"log_level must be one of {sorted(allowed)}"
            raise ValueError(msg)
        return normalized


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
