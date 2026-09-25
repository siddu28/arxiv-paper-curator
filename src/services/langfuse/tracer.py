"""Simple, efficient Langfuse tracing utility for RAG pipeline.

Provides context-manager-based tracing for each RAG step:
- trace_request: Top-level request trace
- trace_embedding: Query embedding step
- trace_search: OpenSearch retrieval step
- trace_prompt_construction: Prompt building step
- trace_generation: LLM generation step

All context managers are no-ops if Langfuse is disabled.
"""

import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from .client import LangfuseTracer


class RAGTracer:
    """Clean, purpose-built tracer for RAG operations.

    Wraps LangfuseTracer with RAG-specific context managers that
    instrument each step of the retrieve-augment-generate pipeline.
    """

    def __init__(self, tracer: LangfuseTracer):
        """Initialize with a LangfuseTracer instance.

        :param tracer: The underlying Langfuse tracer client
        """
        self.tracer = tracer

    @contextmanager
    def trace_request(self, user_id: str, query: str):
        """Main request trace context manager.

        Creates a top-level span that wraps the entire RAG request.

        :param user_id: User identifier
        :param query: The user's question
        :yields: Trace span object (or None if disabled)
        """
        with self.tracer.start_span(
            name="rag_request",
            input_data={"query": query, "user_id": user_id},
            metadata={"session_id": f"session_{user_id}"},
        ) as trace:
            try:
                yield trace
            finally:
                if trace:
                    self.tracer.flush()

    @contextmanager
    def trace_embedding(self, trace, query: str):
        """Query embedding operation with timing.

        :param trace: Parent trace span
        :param query: The query being embedded
        :yields: Embedding span object
        """
        start_time = time.time()
        with self.tracer.start_span(
            name="query_embedding",
            input_data={"query": query, "query_length": len(query)},
        ) as span:
            try:
                yield span
            finally:
                duration = time.time() - start_time
                if span:
                    self.tracer.update_span(
                        span=span,
                        output={"embedding_duration_ms": round(duration * 1000, 2), "success": True},
                    )

    @contextmanager
    def trace_search(self, trace, query: str, top_k: int):
        """Search operation with timing.

        :param trace: Parent trace span
        :param query: Search query
        :param top_k: Number of results requested
        :yields: Search span object
        """
        with self.tracer.start_span(
            name="search_retrieval",
            input_data={"query": query, "top_k": top_k},
        ) as span:
            yield span

    def end_search(self, span, chunks: List[Dict], arxiv_ids: List[str], total_hits: int):
        """End search span with essential results.

        :param span: Search span to end
        :param chunks: Retrieved chunks
        :param arxiv_ids: arXiv IDs found
        :param total_hits: Total search hits
        """
        if not span:
            return

        self.tracer.update_span(
            span=span,
            output={
                "chunks_returned": len(chunks),
                "unique_papers": len(set(arxiv_ids)),
                "total_hits": total_hits,
                "arxiv_ids": list(set(arxiv_ids)),
            },
        )

    @contextmanager
    def trace_prompt_construction(self, trace, chunks: List[Dict]):
        """Prompt building with timing.

        :param trace: Parent trace span
        :param chunks: Chunks used in prompt
        :yields: Prompt construction span
        """
        with self.tracer.start_span(
            name="prompt_construction",
            input_data={"chunk_count": len(chunks)},
        ) as span:
            yield span

    def end_prompt(self, span, prompt: str):
        """End prompt span with final prompt info.

        :param span: Prompt span to end
        :param prompt: The constructed prompt
        """
        if not span:
            return

        self.tracer.update_span(
            span=span,
            output={
                "prompt_length": len(prompt),
                "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            },
        )

    @contextmanager
    def trace_generation(self, trace, model: str, prompt: str):
        """LLM generation with timing.

        :param trace: Parent trace span
        :param model: Model name
        :param prompt: Full prompt sent to LLM
        :yields: Generation span
        """
        with self.tracer.start_span(
            name="llm_generation",
            input_data={"model": model, "prompt_length": len(prompt), "prompt": prompt},
        ) as span:
            yield span

    def end_generation(self, span, response: str, model: str):
        """End generation span with response.

        :param span: Generation span to end
        :param response: LLM response text
        :param model: Model that was used
        """
        if not span:
            return

        self.tracer.update_span(
            span=span,
            output={"response": response, "response_length": len(response), "model_used": model},
        )

    def end_request(self, trace, response: str, total_duration: float):
        """End main request trace with summary.

        :param trace: Top-level trace span
        :param response: Final answer
        :param total_duration: Total request time in seconds
        """
        if not trace:
            return

        try:
            self.tracer.update_span(
                span=trace,
                output={
                    "answer": response,
                    "total_duration_seconds": round(total_duration, 3),
                    "response_length": len(response),
                },
            )
        except Exception:
            # Silently fail - don't break the request for tracing issues
            pass
