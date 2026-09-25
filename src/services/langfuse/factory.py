"""Factory for creating Langfuse tracer instances."""

from src.config import Settings
from src.services.langfuse.client import LangfuseTracer


def make_langfuse_tracer(settings: Settings | None = None) -> LangfuseTracer:
    """Create a Langfuse tracer instance.

    :param settings: Application settings (uses defaults if None)
    :returns: Configured LangfuseTracer instance
    """
    if settings is None:
        settings = Settings()
    return LangfuseTracer(settings)
