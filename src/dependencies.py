"""
FastAPI dependency injection for the arXiv Paper Curator.
=========================================================

Week 1: Dependencies for Settings, Database, Session, and OpenSearch.
Week 2: Added ArxivDep and PDFParserDep.
New dependencies will be added as services are integrated:
    - Week 4: EmbeddingsDep
    - Week 5: OllamaDep
    - Week 6: LangfuseDep, CacheDep
    - Week 7: TelegramDep, AgenticRAGDep
"""

from functools import lru_cache
from typing import Annotated, Generator

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from src.config import Settings
from src.db.interfaces.base import BaseDatabase
from src.services.arxiv.client import ArxivClient
from src.services.opensearch.client import OpenSearchClient
from src.services.pdf_parser.parser import PDFParserService
from src.services.embeddings.jina_client import JinaEmbeddingsClient


@lru_cache
def get_settings() -> Settings:
    """Get application settings (cached singleton)."""
    return Settings()


def get_request_settings(request: Request) -> Settings:
    """Get settings from the request state."""
    return request.app.state.settings


def get_database(request: Request) -> BaseDatabase:
    """Get database from the request state."""
    return request.app.state.database


def get_db_session(database: Annotated[BaseDatabase, Depends(get_database)]) -> Generator[Session, None, None]:
    """Get database session dependency."""
    with database.get_session() as session:
        yield session


def get_opensearch_client(request: Request) -> OpenSearchClient:
    """Get OpenSearch client from the request state."""
    return request.app.state.opensearch_client


def get_arxiv_client(request: Request) -> ArxivClient:
    """Get arXiv client from the request state."""
    return request.app.state.arxiv_client


def get_pdf_parser(request: Request) -> PDFParserService:
    """Get PDF parser service from the request state."""
    return request.app.state.pdf_parser



def get_embeddings_client(request: Request) -> JinaEmbeddingsClient:
    "`""Get Jina embeddings client from the request state."`""
    return request.app.state.embeddings_client

# Dependency annotations - use these in router function signatures
SettingsDep = Annotated[Settings, Depends(get_settings)]
DatabaseDep = Annotated[BaseDatabase, Depends(get_database)]
SessionDep = Annotated[Session, Depends(get_db_session)]
OpenSearchDep = Annotated[OpenSearchClient, Depends(get_opensearch_client)]
ArxivDep = Annotated[ArxivClient, Depends(get_arxiv_client)]
PDFParserDep = Annotated[PDFParserService, Depends(get_pdf_parser)]
EmbeddingsDep = Annotated[JinaEmbeddingsClient, Depends(get_embeddings_client)]

