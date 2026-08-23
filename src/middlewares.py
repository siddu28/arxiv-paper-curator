"""
Simple request/error logging middleware stubs for Week 1.

Note: These functions are NOT integrated into FastAPI middleware system yet.
In production, these would be used via FastAPI middleware decorators
or integrated into a proper BaseHTTPMiddleware class for automatic
request logging, performance monitoring, and error tracking.
"""

import logging

logger = logging.getLogger(__name__)


def log_request(method: str, path: str) -> None:
    """Simple request logging for Week 1."""
    logger.info(f"{method} {path}")


def log_error(error: str, method: str, path: str) -> None:
    """Simple error logging for Week 1."""
    logger.error(f"Error in {method} {path}: {error}")
