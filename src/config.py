"""
Configuration management for the arXiv Paper Curator.
=====================================================

Week 1: Foundation settings for PostgreSQL, OpenSearch, and Ollama.
New settings classes will be added in future weeks as services are integrated:
    - Week 2: ArxivSettings, PDFParserSettings
    - Week 4: ChunkingSettings, JinaSettings
    - Week 6: LangfuseSettings, RedisSettings
    - Week 7: TelegramSettings
"""

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"


class BaseConfigSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        extra="ignore",
        frozen=True,
        env_nested_delimiter="__",
        case_sensitive=False,
    )


class OpenSearchSettings(BaseConfigSettings):
    """OpenSearch connection and index configuration."""

    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="OPENSEARCH__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    host: str = "http://localhost:9200"
    index_name: str = "arxiv-papers"
    chunk_index_suffix: str = "chunks"  # Creates index: {index_name}-{suffix}
    max_text_size: int = 1000000


class Settings(BaseConfigSettings):
    """Main application settings.

    Reads configuration from environment variables and .env file.
    Week 1 includes only the core infrastructure settings.
    """

    app_version: str = "0.1.0"
    debug: bool = True
    environment: Literal["development", "staging", "production"] = "development"
    service_name: str = "rag-api"

    # PostgreSQL settings
    postgres_database_url: str = "postgresql://rag_user:rag_password@localhost:5432/rag_db"
    postgres_echo_sql: bool = False
    postgres_pool_size: int = 20
    postgres_max_overflow: int = 0

    # Ollama settings (used by health check in Week 1, full integration in Week 5)
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:1b"
    ollama_timeout: int = 300

    # Nested settings
    opensearch: OpenSearchSettings = Field(default_factory=OpenSearchSettings)

    @field_validator("postgres_database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        if not (v.startswith("postgresql://") or v.startswith("postgresql+psycopg2://")):
            raise ValueError("Database URL must start with 'postgresql://' or 'postgresql+psycopg2://'")
        return v


def get_settings() -> Settings:
    """Create and return application settings instance."""
    return Settings()
