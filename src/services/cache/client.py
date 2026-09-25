"""Redis-based exact match cache for RAG queries.

Caches complete AskResponse objects keyed by a SHA-256 hash of the
request parameters. Avoids redundant LLM calls for repeated queries.
"""

import hashlib
import json
import logging
from datetime import timedelta
from typing import Optional

import redis
from src.config import RedisSettings
from src.schemas.api.ask import AskRequest, AskResponse

logger = logging.getLogger(__name__)


class CacheClient:
    """Redis-based exact match cache for RAG queries.

    Cache keys are SHA-256 hashes of the full request parameters
    (query + model + top_k + use_hybrid + categories), ensuring
    that only identical requests return cached responses.
    """

    def __init__(self, redis_client: redis.Redis, settings: RedisSettings):
        """Initialize cache client.

        :param redis_client: Connected Redis client
        :param settings: Redis settings with TTL config
        """
        self.redis = redis_client
        self.settings = settings
        self.ttl = timedelta(hours=settings.ttl_hours)

    def _generate_cache_key(self, request: AskRequest) -> str:
        """Generate exact cache key based on request parameters.

        :param request: The ask request to hash
        :returns: Cache key string like 'exact_cache:a1b2c3d4e5f6g7h8'
        """
        key_data = {
            "query": request.query,
            "model": request.model,
            "top_k": request.top_k,
            "use_hybrid": request.use_hybrid,
            "categories": sorted(request.categories) if request.categories else [],
        }
        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
        return f"exact_cache:{key_hash}"

    async def find_cached_response(self, request: AskRequest) -> Optional[AskResponse]:
        """Find cached response for exact query match.

        :param request: The ask request to look up
        :returns: Cached AskResponse or None if not found
        """
        try:
            cache_key = self._generate_cache_key(request)

            # Simple Redis GET operation - O(1)
            cached_response = self.redis.get(cache_key)

            if cached_response:
                try:
                    response_data = json.loads(cached_response)
                    logger.info("Cache hit for exact query match")
                    return AskResponse(**response_data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to deserialize cached response: {e}")
                    return None

            return None

        except Exception as e:
            logger.error(f"Error checking cache: {e}")
            return None

    async def store_response(self, request: AskRequest, response: AskResponse) -> bool:
        """Store response for exact query matching.

        :param request: The original ask request
        :param response: The response to cache
        :returns: True if stored successfully
        """
        try:
            cache_key = self._generate_cache_key(request)

            # Simple Redis SET operation with TTL
            success = self.redis.set(cache_key, response.model_dump_json(), ex=self.ttl)

            if success:
                logger.info(f"Stored response in exact cache with key {cache_key[:20]}...")
                return True
            else:
                logger.warning("Failed to store response in cache")
                return False

        except Exception as e:
            logger.error(f"Error storing in cache: {e}")
            return False
