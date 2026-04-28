from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "CardioSense Pro API"
    app_version: str = "1.0.0"
    api_v1_prefix: str = "/api/v1"

    secret_key: str = "change-this-in-production"
    access_token_expire_minutes: int = 60 * 24

    database_url: str = "sqlite:///./cardiosense.db"
    model_artifact_path: str = "./model_artifacts/heart_model.joblib"

    cors_origins: list[str] | str = [
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "null",
    ]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: Any) -> list[str]:
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return []

    @property
    def model_artifact_full_path(self) -> Path:
        return Path(self.model_artifact_path).expanduser().resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
