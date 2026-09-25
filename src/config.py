"""
Configuration management for the arXiv Paper Curator.
=====================================================

Week 1: Foundation settings for PostgreSQL, OpenSearch, and Ollama.
Week 2: Added ArxivSettings and PDFParserSettings for data ingestion.
Week 3: Added ChunkingSettings, Jina API key, and vector search config.
New settings classes will be added in future weeks as services are integrated:
    - Week 5: LangfuseSettings, RedisSettings (DONE)
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


class ArxivSettings(BaseConfigSettings):
    """arXiv API client configuration.

    Controls rate limiting, search parameters, PDF download behavior,
    and concurrency limits for the data ingestion pipeline.

    Environment variables are prefixed with ARXIV__ (double underscore).
    Example: ARXIV__MAX_RESULTS=20
    """

    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="ARXIV__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    base_url: str = "https://export.arxiv.org/api/query"
    pdf_cache_dir: str = "./data/arxiv_pdfs"
    rate_limit_delay: float = 3.0       # Seconds between requests (arXiv requires >= 3s)
    timeout_seconds: int = 30           # HTTP request timeout
    max_results: int = 15               # Default papers per fetch
    search_category: str = "cs.AI"      # arXiv category to search
    download_max_retries: int = 3       # PDF download retry attempts
    download_retry_delay_base: float = 5.0  # Base delay for retry backoff (seconds)
    max_concurrent_downloads: int = 5   # Concurrent PDF downloads
    max_concurrent_parsing: int = 1     # Concurrent PDF parsing operations

    # XML namespaces used in arXiv API responses
    namespaces: dict = {
        "atom": "http://www.w3.org/2005/Atom",
        "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    @field_validator("pdf_cache_dir")
    @classmethod
    def validate_cache_dir(cls, v: str) -> str:
        os.makedirs(v, exist_ok=True)
        return v


class PDFParserSettings(BaseConfigSettings):
    """PDF parsing configuration for Docling.

    Controls page limits, file size limits, and parsing behavior.
    Environment variables are prefixed with PDF_PARSER__ (double underscore).
    Example: PDF_PARSER__MAX_PAGES=50
    """

    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="PDF_PARSER__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    max_pages: int = 30                 # Maximum pages to process per PDF
    max_file_size_mb: int = 20          # Maximum PDF file size in MB
    do_ocr: bool = False                # Enable OCR for scanned PDFs (slow)
    do_table_structure: bool = True     # Extract table structures


class ChunkingSettings(BaseConfigSettings):
    """Text chunking configuration for the indexing pipeline.

    Controls how paper text is split into overlapping chunks for
    vector search. Uses word-based chunking with configurable sizes.

    Environment variables are prefixed with CHUNKING__ (double underscore).
    Example: CHUNKING__CHUNK_SIZE=600
    """

    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="CHUNKING__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    chunk_size: int = 600               # Target words per chunk
    overlap_size: int = 100             # Words to overlap between consecutive chunks
    min_chunk_size: int = 100           # Minimum words for a chunk to be valid
    section_based: bool = True          # Use section-based chunking when sections available


class OpenSearchSettings(BaseConfigSettings):
    """OpenSearch connection and index configuration.

    Week 3: Added vector search and hybrid search settings.
    """

    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="OPENSEARCH__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    host: str = "http://localhost:9200"
    index_name: str = "arxiv-papers"
    chunk_index_suffix: str = "chunks"  # Creates hybrid index: {index_name}-{suffix}
    max_text_size: int = 1000000

    # Vector search settings (Week 3)
    vector_dimension: int = 1024        # Jina embeddings v3 dimension
    vector_space_type: str = "cosinesimil"  # cosinesimil, l2, innerproduct

    # Hybrid search settings (Week 3)
    rrf_pipeline_name: str = "hybrid-rrf-pipeline"
    hybrid_search_size_multiplier: int = 2  # Get k*multiplier candidates for better recall

class LangfuseSettings(BaseConfigSettings):
    """Langfuse observability settings.

    Langfuse provides tracing, scoring, and debugging for RAG pipelines.
    Set LANGFUSE__PUBLIC_KEY, LANGFUSE__SECRET_KEY, LANGFUSE__HOST in .env.
    """

    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="LANGFUSE__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    public_key: str = ""
    secret_key: str = ""
    host: str = "http://localhost:3000"  # Self-hosted Langfuse URL
    enabled: bool = True
    flush_at: int = 15  # Number of events before flushing
    flush_interval: float = 1.0  # Seconds between flushes
    max_retries: int = 3
    timeout: int = 30
    debug: bool = False


class RedisSettings(BaseConfigSettings):
    """Redis caching settings.

    Used for exact-match query caching to avoid redundant LLM calls.
    Set REDIS__HOST, REDIS__PORT, REDIS__PASSWORD in .env.
    """

    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="REDIS__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    host: str = "localhost"
    port: int = 6379
    password: str = ""
    db: int = 0
    decode_responses: bool = True
    socket_timeout: int = 30
    socket_connect_timeout: int = 30

    # Cache settings
    ttl_hours: int = 6  # Cache TTL in hours



class Settings(BaseConfigSettings):
    """Main application settings.

    Reads configuration from environment variables and .env file.
    Week 2 adds ArxivSettings and PDFParserSettings for data ingestion.
    Week 3 adds ChunkingSettings and Jina API key for hybrid search.
    Week 5 adds LangfuseSettings and RedisSettings for observability and caching.
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

    # Jina AI embeddings configuration (Week 3)
    jina_api_key: str = ""

    # Nested settings
    arxiv: ArxivSettings = Field(default_factory=ArxivSettings)
    pdf_parser: PDFParserSettings = Field(default_factory=PDFParserSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    opensearch: OpenSearchSettings = Field(default_factory=OpenSearchSettings)
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)

    @field_validator("postgres_database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        if not (v.startswith("postgresql://") or v.startswith("postgresql+psycopg2://")):
            raise ValueError("Database URL must start with 'postgresql://' or 'postgresql+psycopg2://'")
        return v


def get_settings() -> Settings:
    """Create and return application settings instance."""
    return Settings()

