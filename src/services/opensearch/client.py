"""
Week 1: Minimal OpenSearch client — health check and index management only.

Search functionality (BM25, hybrid, vector) will be added in Weeks 3-4.
Full client methods:
    - Week 3: BM25 keyword search, document indexing
    - Week 4: Hybrid search with RRF, vector search, query builder
"""

import logging
from typing import Any, Dict

from opensearchpy import OpenSearch

from src.config import Settings

logger = logging.getLogger(__name__)


class OpenSearchClient:
    """OpenSearch client — Week 1 provides health check and index setup only.

    Search capabilities (BM25, hybrid, vector) will be added in Weeks 3-4.
    """

    def __init__(self, host: str, settings: Settings):
        self.host = host
        self.settings = settings
        self.index_name = f"{settings.opensearch.index_name}-{settings.opensearch.chunk_index_suffix}"

        self.client = OpenSearch(
            hosts=[host],
            use_ssl=False,
            verify_certs=False,
            ssl_show_warn=False,
        )

        logger.info(f"OpenSearch client initialized with host: {host}")

    def health_check(self) -> bool:
        """Check if OpenSearch cluster is healthy."""
        try:
            health = self.client.cluster.health()
            return health["status"] in ["green", "yellow"]
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics for the index."""
        try:
            if not self.client.indices.exists(index=self.index_name):
                return {"index_name": self.index_name, "exists": False, "document_count": 0}

            stats_response = self.client.indices.stats(index=self.index_name)
            index_stats = stats_response["indices"][self.index_name]["total"]

            return {
                "index_name": self.index_name,
                "exists": True,
                "document_count": index_stats["docs"]["count"],
                "deleted_count": index_stats["docs"]["deleted"],
                "size_in_bytes": index_stats["store"]["size_in_bytes"],
            }

        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {"index_name": self.index_name, "exists": False, "document_count": 0, "error": str(e)}

    def setup_indices(self, force: bool = False) -> Dict[str, bool]:
        """Setup a basic index for storing paper data.

        Week 1: Creates a simple index with basic mappings.
        Week 3-4: This will be enhanced with hybrid mappings (BM25 + vector).
        """
        results = {}
        results["hybrid_index"] = self._create_basic_index(force)
        return results

    def _create_basic_index(self, force: bool = False) -> bool:
        """Create a basic index for paper chunks.

        :param force: If True, recreate index even if it exists
        :returns: True if created, False if already exists
        """
        try:
            if force and self.client.indices.exists(index=self.index_name):
                self.client.indices.delete(index=self.index_name)
                logger.info(f"Deleted existing index: {self.index_name}")

            if not self.client.indices.exists(index=self.index_name):
                # Week 1: Basic mapping — will be enhanced with vector fields in Week 4
                basic_mapping = {
                    "settings": {
                        "index": {
                            "number_of_shards": 1,
                            "number_of_replicas": 0,
                        }
                    },
                    "mappings": {
                        "properties": {
                            "arxiv_id": {"type": "keyword"},
                            "title": {"type": "text", "analyzer": "standard"},
                            "content": {"type": "text", "analyzer": "standard"},
                            "section_title": {"type": "text"},
                            "chunk_index": {"type": "integer"},
                            "created_at": {"type": "date"},
                        }
                    },
                }

                self.client.indices.create(index=self.index_name, body=basic_mapping)
                logger.info(f"Created basic index: {self.index_name}")
                return True

            logger.info(f"Index already exists: {self.index_name}")
            return False

        except Exception as e:
            # Handle race condition when multiple workers start simultaneously
            if "resource_already_exists_exception" in str(e):
                logger.info(f"Index already exists (created by another worker): {self.index_name}")
                return False
            logger.error(f"Error creating index: {e}")
            raise
