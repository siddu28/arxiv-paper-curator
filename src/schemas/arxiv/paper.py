"""
ArXiv paper Pydantic schemas for data validation.
==================================================

Schema hierarchy:
    ArxivPaper     - Raw data from arXiv API (XML parsed)
    PaperBase      - Shared fields for database operations
    PaperCreate    - Creating/updating a paper (includes parsed PDF content)
    PaperResponse  - API response with all fields + timestamps
    PaperSearchResponse - Paginated list of papers

Usage flow:
    arXiv XML -> ArxivPaper -> PaperCreate -> Database -> PaperResponse -> API
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ArxivPaper(BaseModel):
    """Schema for arXiv API response data.

    This is the raw data parsed from arXiv's XML API response.
    All fields are strings/lists — no database IDs or timestamps.
    """

    arxiv_id: str = Field(..., description="arXiv paper ID")
    title: str = Field(..., description="Paper title")
    authors: List[str] = Field(..., description="List of author names")
    abstract: str = Field(..., description="Paper abstract")
    categories: List[str] = Field(..., description="Paper categories")
    published_date: str = Field(..., description="Date published on arXiv (ISO format)")
    pdf_url: str = Field(..., description="URL to PDF")


class PaperBase(BaseModel):
    """Base schema with shared fields for all paper database operations.

    Note: published_date is datetime here (vs string in ArxivPaper)
    because the database stores proper datetime objects.
    """

    # Core arXiv metadata
    arxiv_id: str = Field(..., description="arXiv paper ID")
    title: str = Field(..., description="Paper title")
    authors: List[str] = Field(..., description="List of author names")
    abstract: str = Field(..., description="Paper abstract")
    categories: List[str] = Field(..., description="Paper categories")
    published_date: datetime = Field(..., description="Date published on arXiv")
    pdf_url: str = Field(..., description="URL to PDF")


class PaperCreate(PaperBase):
    """Schema for creating a paper with optional parsed content.

    Extends PaperBase with PDF parsing results. These fields are
    None when the paper is first fetched (metadata only) and populated
    after PDF processing.
    """

    # Parsed PDF content (optional - added when PDF is processed)
    raw_text: Optional[str] = Field(None, description="Full raw text extracted from PDF")
    sections: Optional[List[Dict[str, Any]]] = Field(None, description="List of sections with titles and content")
    references: Optional[List[Dict[str, Any]]] = Field(None, description="List of references if extracted")

    # PDF processing metadata (optional)
    parser_used: Optional[str] = Field(None, description="Which parser was used (DOCLING, etc.)")
    parser_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional parser metadata")
    pdf_processed: Optional[bool] = Field(False, description="Whether PDF was successfully processed")
    pdf_processing_date: Optional[datetime] = Field(None, description="When PDF was processed")


class PaperResponse(PaperBase):
    """Schema for paper API responses with all content.

    Includes the database-generated UUID and timestamps.
    from_attributes = True enables creating this from SQLAlchemy ORM objects.
    """

    id: UUID

    # Parsed PDF content (optional fields)
    raw_text: Optional[str] = Field(None, description="Full raw text extracted from PDF")
    sections: Optional[List[Dict[str, Any]]] = Field(None, description="List of sections with titles and content")
    references: Optional[List[Dict[str, Any]]] = Field(None, description="List of references if extracted")

    # PDF processing metadata
    parser_used: Optional[str] = Field(None, description="Which parser was used")
    parser_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional parser metadata")
    pdf_processed: bool = Field(False, description="Whether PDF was successfully processed")
    pdf_processing_date: Optional[datetime] = Field(None, description="When PDF was processed")

    # Timestamps
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PaperSearchResponse(BaseModel):
    """Paginated response for paper search/list endpoints."""

    papers: List[PaperResponse]
    total: int
