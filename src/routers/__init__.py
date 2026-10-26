"""Router modules for the RAG API."""

# Week 1: Health check router
from . import ping

# Week 3: BM25 search router
from . import search

# Week 4: Hybrid search + RAG ask routers
from . import hybrid_search
from . import ask

# Week 6: Agentic RAG router
from . import agentic_ask

__all__ = ["ping", "search", "hybrid_search", "ask", "agentic_ask"]
