"""
Custom exception hierarchy for the arXiv Paper Curator.
=======================================================

Week 1: Foundation exceptions for repository and service errors.
New exception classes will be added as services are integrated:
    - Week 2: ArxivAPIException, PDFParsingException, etc.
    - Week 5: LLMException, OllamaException, etc.
"""


# Repository exceptions (database layer)
class RepositoryException(Exception):
    """Base exception for repository-related errors."""


class PaperNotFound(RepositoryException):
    """Exception raised when paper data is not found."""


class PaperNotSaved(RepositoryException):
    """Exception raised when paper data is not saved."""


# OpenSearch exceptions
class OpenSearchException(Exception):
    """Base exception for OpenSearch-related errors."""


# LLM exceptions (stubs for Week 1 health check, full implementation in Week 5)
class LLMException(Exception):
    """Base exception for LLM-related errors."""


class OllamaException(LLMException):
    """Exception raised for Ollama service errors."""


class OllamaConnectionError(OllamaException):
    """Exception raised when cannot connect to Ollama service."""


class OllamaTimeoutError(OllamaException):
    """Exception raised when Ollama service times out."""


# General application exceptions
class ConfigurationError(Exception):
    """Exception raised when configuration is invalid."""
