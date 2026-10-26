"""Factory for creating AgenticRAGService instances."""

from typing import Optional

from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.langfuse.client import LangfuseTracer
from src.services.ollama.client import OllamaClient
from src.services.opensearch.client import OpenSearchClient

from .agentic_rag import AgenticRAGService
from .config import GraphConfig


def make_agentic_rag_service(
    opensearch_client: OpenSearchClient,
    ollama_client: OllamaClient,
    embeddings_client: JinaEmbeddingsClient,
    langfuse_tracer: Optional[LangfuseTracer] = None,
    top_k: int = 3,
    use_hybrid: bool = True,
) -> AgenticRAGService:
    """Create AgenticRAGService with dependency injection.

    :param opensearch_client: Client for document search
    :param ollama_client: Client for LLM generation
    :param embeddings_client: Client for embeddings
    :param langfuse_tracer: Optional Langfuse tracer for observability
    :param top_k: Number of documents to retrieve (default: 3)
    :param use_hybrid: Use hybrid search (default: True)
    :returns: Configured AgenticRAGService instance
    """
    graph_config = GraphConfig(
        top_k=top_k,
        use_hybrid=use_hybrid,
    )

    return AgenticRAGService(
        opensearch_client=opensearch_client,
        ollama_client=ollama_client,
        embeddings_client=embeddings_client,
        langfuse_tracer=langfuse_tracer,
        graph_config=graph_config,
    )
