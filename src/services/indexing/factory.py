"""Factory for creating hybrid indexing service instances."""

import logging
from functools import lru_cache

from src.config import get_settings
from src.services.embeddings.factory import make_jina_embeddings_client
from src.services.opensearch.factory import make_opensearch_client

from .hybrid_indexer import HybridIndexingService
from .text_chunker import TextChunker

logger = logging.getLogger(__name__)


def make_hybrid_indexing_service() -> HybridIndexingService:
    """Create a HybridIndexingService with all dependencies.

    Note: NOT cached with lru_cache because the underlying services
    (embeddings, opensearch) are already cached individually.

    :returns: Configured HybridIndexingService instance
    """
    settings = get_settings()

    # Create text chunker with configured settings
    chunker = TextChunker(
        chunk_size=settings.chunking.chunk_size,
        overlap_size=settings.chunking.overlap_size,
        min_chunk_size=settings.chunking.min_chunk_size,
    )

    # Get cached service instances
    embeddings_client = make_jina_embeddings_client()
    opensearch_client = make_opensearch_client()

    service = HybridIndexingService(
        chunker=chunker,
        embeddings_client=embeddings_client,
        opensearch_client=opensearch_client,
    )

    logger.info("Hybrid indexing service created")
    return service
