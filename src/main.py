"""
FastAPI application entry point for the arXiv Paper Curator.
=============================================================

Week 1: Initializes database + OpenSearch, includes health check router only.
Week 2: Added arXiv client and PDF parser to lifespan initialization.
Future weeks will add more services to the lifespan and include more routers:
    - Week 3: Search router (DONE)
    - Week 4: Hybrid search router, embeddings service
    - Week 5: Ollama LLM, RAG ask/stream routers, Gradio app
    - Week 6: Langfuse tracer, Redis cache
    - Week 7: Telegram bot, Agentic RAG router
"""

import logging
import os
import gradio as gr
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from src.config import get_settings
from src.db.factory import make_database
from src.routers import ping, search, hybrid_search, ask
from src.services.arxiv.factory import make_arxiv_client
from src.services.opensearch.factory import make_opensearch_client
from src.services.pdf_parser.factory import make_pdf_parser_service
from src.services.embeddings.factory import make_jina_embeddings_client
from src.services.ollama.factory import make_ollama_client
from src.gradio_app import create_gradio_interface
from src.services.langfuse.factory import make_langfuse_tracer
from src.services.cache.factory import make_cache_client

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan - runs on startup and shutdown.

    Startup: Initialize database, OpenSearch, arXiv client, PDF parser
    Shutdown: Clean up database connections
    """
    logger.info("Starting RAG API...")

    # Load settings
    settings = get_settings()
    app.state.settings = settings

    # Initialize database (creates engine, tables, etc.)
    database = make_database()
    app.state.database = database
    logger.info("Database connected")

    # Initialize OpenSearch client
    opensearch_client = make_opensearch_client()
    app.state.opensearch_client = opensearch_client

    # Verify OpenSearch connectivity and create index if needed
    if opensearch_client.health_check():
        logger.info("OpenSearch connected successfully")

        # Setup index (creates if doesn't exist)
        setup_results = opensearch_client.setup_indices(force=False)
        if setup_results.get("hybrid_index"):
            logger.info("Index created")
        else:
            logger.info("Index already exists")

        # Get simple statistics
        try:
            stats = opensearch_client.client.count(index=opensearch_client.index_name)
            logger.info(f"OpenSearch ready: {stats['count']} documents indexed")
        except Exception:
            logger.info("OpenSearch index ready (stats unavailable)")
    else:
        logger.warning("OpenSearch connection failed - search features will be limited")

    # Week 2: Initialize arXiv client
    arxiv_client = make_arxiv_client()
    app.state.arxiv_client = arxiv_client
    logger.info(f"arXiv client ready: category={arxiv_client.search_category}, max_results={arxiv_client.max_results}")

    # Week 2: Initialize PDF parser service
    pdf_parser = make_pdf_parser_service()
    app.state.pdf_parser = pdf_parser
    logger.info("PDF parser service ready (Docling)")


    # Week 3: Initialize Jina embeddings client
    try:
        jina_client = make_jina_embeddings_client()
        app.state.embeddings_client = jina_client
        logger.info("Jina embeddings client ready (1024-dim vectors)")
    except ValueError as e:
        logger.warning(f"Jina embeddings not available: {e}")
        app.state.embeddings_client = None


    # Week 4: Initialize Ollama LLM client
    ollama_client = make_ollama_client()
    app.state.ollama_client = ollama_client
    try:
        ollama_health = await ollama_client.health_check()
        logger.info(f"Ollama LLM ready: version={ollama_health.get('version', 'unknown')}")
    except Exception as e:
        logger.warning(f"Ollama not available (LLM features limited): {e}")

    # Week 5: Initialize Langfuse tracing
    langfuse_tracer = make_langfuse_tracer(settings)
    app.state.langfuse_tracer = langfuse_tracer

    # Week 5: Initialize Redis cache (optional - app works without it)
    try:
        cache_client = make_cache_client(settings)
        app.state.cache_client = cache_client
        logger.info("Redis cache initialized")
    except Exception as e:
        logger.warning(f"Redis not available (caching disabled): {e}")
        app.state.cache_client = None
    logger.info("Services initialized: Database, OpenSearch, arXiv, PDF Parser, Jina Embeddings, Ollama LLM, Langfuse, Redis")
    logger.info("API ready")

    yield

    # Shutdown: clean up resources
    database.teardown()
    logger.info("API shutdown complete")


# Create the FastAPI application
app = FastAPI(
    title="arXiv Paper Curator API",
    description="Personal arXiv CS.AI paper curator with RAG capabilities",
    version=os.getenv("APP_VERSION", "0.1.0"),
    lifespan=lifespan,
)

# Include routers
app.include_router(ping.router, prefix="/api/v1")  # Health check endpoint
app.include_router(search.router, prefix="/api/v1")  # Week 3: BM25 search
app.include_router(hybrid_search.router, prefix="/api/v1")  # Week 4: Hybrid search
app.include_router(ask.ask_router, prefix="/api/v1")  # Week 4: RAG ask endpoint
app.include_router(ask.stream_router, prefix="/api/v1")  # Week 4: Streaming RAG endpoint



# Week 4: Mount Gradio chat UI at /chat
gradio_interface = create_gradio_interface()
app = gr.mount_gradio_app(app, gradio_interface, path="/chat")
if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")








