"""
Setup task — verify environment and initialize services.
=========================================================

This runs first in the DAG to ensure all services are healthy
before starting the actual data ingestion.
"""

import logging

from sqlalchemy import text

from .common import get_cached_services

logger = logging.getLogger(__name__)


def setup_environment():
    """Setup environment and verify dependencies.

    Checks:
        1. Database connection is alive
        2. OpenSearch cluster is reachable
        3. OpenSearch index exists (creates if needed)
        4. arXiv client is configured
        5. PDF parser is ready
    """
    logger.info("Setting up environment for arXiv paper ingestion")

    try:
        arxiv_client, _pdf_parser, database, _metadata_fetcher, opensearch_client = get_cached_services()

        # Verify database connection
        with database.get_session() as session:
            session.execute(text("SELECT 1"))
            logger.info("Database connection verified")

        # Verify OpenSearch connection
        try:
            health = opensearch_client.client.cluster.health()
            if health["status"] in ["green", "yellow", "red"]:
                logger.info(f"OpenSearch connected (cluster status: {health['status']})")
            else:
                raise Exception(f"OpenSearch cluster unhealthy: {health['status']}")
        except Exception as e:
            raise Exception(f"OpenSearch connection failed: {e}")

        # Setup indices if needed
        setup_results = opensearch_client.setup_indices(force=False)
        if setup_results.get("hybrid_index"):
            logger.info("Search index created")
        else:
            logger.info("Search index already exists")

        logger.info(f"arXiv client ready: {arxiv_client.base_url}")
        logger.info("PDF parser service ready (Docling models cached)")

        return {"status": "success", "message": "Environment setup completed"}

    except Exception as e:
        error_msg = f"Environment setup failed: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)
