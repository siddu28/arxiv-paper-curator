"""
Custom exception hierarchy for the arXiv Paper Curator.
=======================================================

Week 1: Foundation exceptions for repository and service errors.
Week 2: Added arXiv API, PDF parsing, PDF download, and pipeline exceptions.
"""


# Repository exceptions (database layer)
class RepositoryException(Exception):
    """Base exception for repository-related errors."""


class PaperNotFound(RepositoryException):
    """Exception raised when paper data is not found."""


class PaperNotSaved(RepositoryException):
    """Exception raised when paper data is not saved."""


# Parsing exceptions (base for all parsers)
class ParsingException(Exception):
    """Base exception for parsing-related errors."""


# Week 2: PDF parsing exceptions
class PDFParsingException(ParsingException):
    """Base exception for PDF parsing-related errors."""


class PDFValidationError(PDFParsingException):
    """Exception raised when PDF file validation fails (size, page count, corrupt header)."""


# Week 2: PDF download exceptions
class PDFDownloadException(Exception):
    """Base exception for PDF download-related errors."""


class PDFDownloadTimeoutError(PDFDownloadException):
    """Exception raised when PDF download times out after all retries."""


class PDFCacheException(Exception):
    """Exception raised for PDF cache-related errors."""


# OpenSearch exceptions
class OpenSearchException(Exception):
    """Base exception for OpenSearch-related errors."""


# Week 2: arXiv API exceptions
class ArxivAPIException(Exception):
    """Base exception for arXiv API-related errors."""


class ArxivAPITimeoutError(ArxivAPIException):
    """Exception raised when arXiv API request times out."""


class ArxivAPIRateLimitError(ArxivAPIException):
    """Exception raised when arXiv API rate limit is exceeded."""


class ArxivParseError(ArxivAPIException):
    """Exception raised when arXiv API XML response parsing fails."""


# Week 2: Metadata fetching / pipeline exceptions
class MetadataFetchingException(Exception):
    """Base exception for metadata fetching pipeline errors."""


class PipelineException(MetadataFetchingException):
    """Exception raised during pipeline execution (fetch -> download -> parse -> store)."""


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
