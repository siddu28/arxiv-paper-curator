"""Router modules for the RAG API."""

# Week 1: Health check router
from . import ping

# Week 3: BM25 search router
from . import search

__all__ = ["ping", "search"]
