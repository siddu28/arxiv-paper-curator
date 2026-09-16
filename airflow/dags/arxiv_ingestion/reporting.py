"""
Reporting task — generate a daily summary of the ingestion pipeline.
=====================================================================

Collects statistics from previous tasks via XCom and generates
a summary report with database and OpenSearch statistics.
"""

import json
import logging
from datetime import datetime

from .common import get_cached_services

logger = logging.getLogger(__name__)


def generate_daily_report(**context):
    """Generate a daily report of the ingestion pipeline results.

    Collects statistics from all previous tasks and generates a summary report.
    """
    logger.info("Generating daily ingestion report")

    ti = context.get("ti")
    if not ti:
        logger.warning("No task instance available, generating basic report")
        return {"status": "basic_report", "message": "No task instance for XCom data"}

    # Pull results from previous tasks via XCom
    fetch_stats = ti.xcom_pull(task_ids="fetch_daily_papers", key="fetch_results") or {}

    report = {
        "execution_date": context.get("execution_date", datetime.now()).isoformat(),
        "fetch_statistics": {
            "papers_fetched": fetch_stats.get("papers_fetched", 0),
            "papers_stored": fetch_stats.get("papers_stored", 0),
            "target_date": fetch_stats.get("date", "unknown"),
        },
        "pipeline_status": "success" if fetch_stats else "partial",
    }

    # Get database statistics
    try:
        _arxiv_client, _pdf_parser, database, _metadata_fetcher, opensearch_client = get_cached_services()

        with database.get_session() as session:
            from sqlalchemy import func
            from src.models.paper import Paper

            total_papers = session.query(func.count(Paper.id)).scalar()
            report["database_statistics"] = {"total_papers": total_papers}

        # Get OpenSearch statistics
        if opensearch_client.health_check():
            try:
                stats_response = opensearch_client.client.indices.stats(index=opensearch_client.index_name)
                count_response = opensearch_client.client.count(index=opensearch_client.index_name)
                index_stats = stats_response["indices"][opensearch_client.index_name]["total"]

                report["opensearch_statistics"] = {
                    "index_name": opensearch_client.index_name,
                    "document_count": count_response["count"],
                    "index_size_mb": round(index_stats["store"]["size_in_bytes"] / (1024 * 1024), 2),
                }
            except Exception as stats_error:
                logger.error(f"Failed to get OpenSearch statistics: {stats_error}")
                report["opensearch_statistics"] = {"index_name": opensearch_client.index_name, "error": str(stats_error)}
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        report["error"] = str(e)

    # Log the report
    logger.info("Daily Ingestion Report:")
    logger.info(json.dumps(report, indent=2))

    ti.xcom_push(key="daily_report", value=report)

    return report
