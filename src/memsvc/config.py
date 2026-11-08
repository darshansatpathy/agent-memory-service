"""Settings, read from MEMSVC_* environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MEMSVC_", env_file=".env", extra="ignore")

    database_url: str = "postgresql://postgres@localhost:5432/memory"
    embedder: str = "minilm"  # or "hash" for tests and air-gapped smoke runs
    episodic_ttl_days: float | None = 90
    semantic_ttl_days: float | None = None
    merge_similarity: float = 0.92
    recency_half_life_days: float = 30
    sweep_interval_seconds: float = 300
    max_content_chars: int = 4000
    pool_max: int = 10
