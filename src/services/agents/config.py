"""Configuration for the Agentic RAG graph execution.

Defines tunable parameters for the graph: retry limits, guardrail
threshold, model selection, and search settings.
"""

from typing import Any, Dict

from pydantic import BaseModel, Field

from src.config import Settings, get_settings


class GraphConfig(BaseModel):
    """Configuration for the entire graph execution.

    :param max_retrieval_attempts: Maximum retrieval retries before fallback
    :param guardrail_threshold: Score threshold for on-topic validation (0-100)
    :param model: Default Ollama model for LLM calls
    :param temperature: LLM temperature (0.0 = deterministic)
    :param top_k: Number of documents to retrieve
    :param use_hybrid: Whether to use hybrid search (BM25 + vector)
    :param enable_tracing: Whether to enable Langfuse tracing
    :param metadata: Additional runtime metadata
    :param settings: Application settings instance
    """

    max_retrieval_attempts: int = 2
    guardrail_threshold: int = 60
    model: str = "llama3.2:1b"
    temperature: float = 0.0
    top_k: int = 3
    use_hybrid: bool = True
    enable_tracing: bool = True
    metadata: Dict[str, Any] = {}
    settings: Settings = Field(default_factory=get_settings)
