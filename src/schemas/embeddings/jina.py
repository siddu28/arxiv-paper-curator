from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class JinaEmbeddingRequest(BaseModel):
    """Request model for Jina AI embeddings API."""

    model: str = "jina-embeddings-v3"
    task: str = "retrieval.passage"  # "retrieval.passage" for indexing, "retrieval.query" for search
    dimensions: int = 1024
    input: List[str]


class JinaEmbeddingResponse(BaseModel):
    """Response model from Jina AI embeddings API."""

    model: str
    data: List[Dict[str, Any]]
    usage: Optional[Dict[str, int]] = None
