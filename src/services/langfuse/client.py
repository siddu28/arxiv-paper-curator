"""Langfuse tracing client for RAG pipeline observability.

Wraps the Langfuse SDK to provide:
- Trace creation for end-to-end request tracking
- Span creation for individual pipeline steps
- Generation tracking for LLM calls with token usage
- Feedback collection linked to trace IDs
- LangGraph callback handler for agentic RAG (Week 7)
"""

import logging
from contextlib import contextmanager
from typing import Any, Dict, Optional

from langfuse import Langfuse
from src.config import Settings

logger = logging.getLogger(__name__)


class LangfuseTracer:
    """Wrapper for Langfuse v3 tracing client with CallbackHandler support."""

    def __init__(self, settings: Settings):
        """Initialize Langfuse tracer.

        :param settings: Application settings with langfuse config
        """
        self.settings = settings.langfuse
        self.client: Optional[Langfuse] = None

        if self.settings.enabled and self.settings.public_key and self.settings.secret_key:
            try:
                self.client = Langfuse(
                    public_key=self.settings.public_key,
                    secret_key=self.settings.secret_key,
                    host=self.settings.host,
                    flush_at=self.settings.flush_at,
                    flush_interval=self.settings.flush_interval,
                    debug=self.settings.debug,
                )
                logger.info(f"Langfuse v3 tracing initialized (host: {self.settings.host})")
            except Exception as e:
                logger.error(f"Failed to initialize Langfuse: {e}")
                self.client = None
        else:
            logger.info("Langfuse tracing disabled or missing credentials")

    def get_callback_handler(
        self,
        trace_name: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
    ):
        """Get a CallbackHandler for LangChain/LangGraph integration.

        :param trace_name: Optional name for the trace
        :param user_id: Optional user identifier
        :param session_id: Optional session identifier
        :param metadata: Additional metadata to attach
        :param tags: Optional tags for the trace
        :returns: CallbackHandler instance if enabled, None otherwise
        """
        if not self.client:
            return None

        try:
            from langfuse.langchain import CallbackHandler

            handler = CallbackHandler(
                trace_name=trace_name,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata,
                tags=tags,
            )
            return handler
        except Exception as e:
            logger.error(f"Error creating CallbackHandler: {e}")
            return None

    @contextmanager
    def trace_langgraph_agent(
        self,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
    ):
        """Context manager to wrap LangGraph agent execution with tracing.

        Usage:
            with tracer.trace_langgraph_agent(name="agentic_rag") as (trace_ctx, handler):
                result = graph.invoke(input, config={"callbacks": [handler]})

        :param name: Name for the trace span
        :param user_id: Optional user identifier
        :param session_id: Optional session identifier
        :param metadata: Additional metadata
        :param tags: Optional tags
        :yields: Tuple of (trace_context, callback_handler)
        """
        if not self.client:
            yield (None, None)
            return

        handler = self.get_callback_handler(
            trace_name=name,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
            tags=tags,
        )

        yield (None, handler)

    def get_trace_id(self, trace=None) -> Optional[str]:
        """Get the current trace ID from Langfuse context.

        :returns: Trace ID string or None if tracing disabled
        """
        if not self.client:
            return None

        try:
            trace_id = self.client.get_current_trace_id()
            return trace_id
        except Exception as e:
            logger.error(f"Error getting trace ID: {e}")
            return None

    def submit_feedback(
        self,
        trace_id: str,
        score: float,
        name: str = "user-feedback",
        comment: Optional[str] = None,
    ) -> bool:
        """Submit user feedback for a trace.

        :param trace_id: Trace ID from get_trace_id()
        :param score: Feedback score (-1 to 1)
        :param name: Name of the score
        :param comment: Optional feedback comment
        :returns: True if submitted successfully
        """
        if not self.client:
            logger.warning("Cannot submit feedback: Langfuse is disabled")
            return False

        try:
            self.client.score(
                trace_id=trace_id,
                name=name,
                value=score,
                comment=comment,
            )
            logger.info(f"Submitted feedback for trace {trace_id}: score={score}")
            return True
        except Exception as e:
            logger.error(f"Error submitting feedback: {e}")
            return False

    def flush(self):
        """Flush any pending traces."""
        if self.client:
            try:
                self.client.flush()
            except Exception as e:
                logger.error(f"Error flushing Langfuse: {e}")

    def shutdown(self):
        """Shutdown the Langfuse client."""
        if self.client:
            try:
                self.client.flush()
                self.client.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down Langfuse: {e}")

    @contextmanager
    def start_generation(
        self,
        name: str,
        model: str,
        input_data: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Start a generation span for LLM calls.

        Usage:
            with tracer.start_generation("rag_llm", "llama3.2", prompt) as gen:
                response = await llm.generate(...)
                gen.update(output=response, usage_metadata={...})

        :param name: Name for this generation
        :param model: Model identifier
        :param input_data: Input to the LLM
        :param metadata: Additional metadata
        :yields: Generation context object
        """
        if not self.client:
            yield None
            return

        try:
            generation = self.client.generation(
                name=name,
                model=model,
                input=input_data,
                metadata=metadata or {},
            )
            yield generation
        except Exception as e:
            logger.error(f"Error creating generation span: {e}")
            yield None

    @contextmanager
    def start_span(
        self,
        name: str,
        input_data: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Start a generic span for non-LLM operations.

        Usage:
            with tracer.start_span("retrieve_papers", {"query": q}) as span:
                docs = retrieve(...)
                span.update(output={"docs_count": len(docs)})

        :param name: Name for this span
        :param input_data: Input to this operation
        :param metadata: Additional metadata
        :yields: Span context object
        """
        if not self.client:
            yield None
            return

        try:
            span = self.client.span(
                name=name,
                input=input_data,
                metadata=metadata or {},
            )
            yield span
        except Exception as e:
            logger.error(f"Error creating span: {e}")
            yield None

    def update_generation(
        self,
        generation,
        output: Any,
        usage_metadata: Optional[Dict[str, Any]] = None,
        completion_start_time: Optional[float] = None,
    ):
        """Update a generation span with output and usage metrics.

        :param generation: Generation object from start_generation()
        :param output: LLM output/response
        :param usage_metadata: Token usage and timing info
        """
        if not generation:
            return

        try:
            update_data = {"output": output}

            if usage_metadata:
                if "prompt_tokens" in usage_metadata:
                    update_data["usage"] = {
                        "input": usage_metadata.get("prompt_tokens", 0),
                        "output": usage_metadata.get("completion_tokens", 0),
                        "total": usage_metadata.get("total_tokens", 0),
                    }

                if "latency_ms" in usage_metadata:
                    update_data["metadata"] = update_data.get("metadata", {})
                    update_data["metadata"]["latency_ms"] = usage_metadata["latency_ms"]

            generation.update(**update_data)
            generation.end()
        except Exception as e:
            logger.error(f"Error updating generation: {e}")

    def update_span(
        self,
        span,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        level: Optional[str] = None,
        status_message: Optional[str] = None,
    ):
        """Update a span with output and metadata.

        :param span: Span object from start_span()
        :param output: Operation output
        :param metadata: Additional metadata
        :param level: Log level for error tracking
        :param status_message: Status or error message
        """
        if not span:
            return

        try:
            update_data = {}
            if output is not None:
                update_data["output"] = output
            if metadata:
                update_data["metadata"] = metadata
            if level:
                update_data["level"] = level
            if status_message:
                update_data["status_message"] = status_message

            if update_data:
                span.update(**update_data)
            span.end()
        except Exception as e:
            logger.error(f"Error updating span: {e}")
