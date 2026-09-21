"""Factory for creating Jina embeddings client instances."""

import logging
from functools import lru_cache

from src.config import get_settings

from .jina_client import JinaEmbeddingsClient

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def make_jina_embeddings_client() -> JinaEmbeddingsClient:
    """Create and cache a JinaEmbeddingsClient singleton.

    :returns: Configured JinaEmbeddingsClient instance
    :raises ValueError: If JINA_API_KEY is not configured
    """
    settings = get_settings()

    if not settings.jina_api_key:
        raise ValueError(
            "JINA_API_KEY is required for embeddings. "
            "Get a free API key from https://jina.ai and set it in your .env file."
        )

    client = JinaEmbeddingsClient(api_key=settings.jina_api_key)
    logger.info("Jina embeddings client created (cached)")
    return client
