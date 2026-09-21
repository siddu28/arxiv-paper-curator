"""
BM25 search router — keyword-based search over indexed paper chunks.

Week 3: BM25 search only (no embeddings needed).
Week 6: This is complemented by the hybrid search router which adds vector search.
"""

import logging

from fastapi import APIRouter, HTTPException
from src.dependencies import OpenSearchDep
from src.schemas.api.search import SearchRequest, SearchHit, SearchResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


@router.post("/", response_model=SearchResponse)
async def search_papers(request: SearchRequest, opensearch_client: OpenSearchDep) -> SearchResponse:
    """
    Search indexed paper chunks using BM25 keyword matching.

    Searches across chunk_text, title, and abstract fields with field boosting.
    Supports pagination, category filtering, and date-based sorting.

    Returns matching chunks with highlighted snippets showing where the query matched.
    """
    try:
        if not opensearch_client.health_check():
            raise HTTPException(status_code=503, detail="Search service is currently unavailable")

        logger.info(f"BM25 search: '{request.query}' (size={request.size}, from={request.from_})")

        results = opensearch_client.search_papers(
            query=request.query,
            size=request.size,
            from_=request.from_,
            categories=request.categories,
            latest=request.latest_papers,
        )

        hits = []
        for hit in results.get("hits", []):
            hits.append(
                SearchHit(
                    arxiv_id=hit.get("arxiv_id", ""),
                    title=hit.get("title", ""),
                    authors=hit.get("authors"),
                    abstract=hit.get("abstract"),
                    published_date=hit.get("published_date"),
                    pdf_url=hit.get("pdf_url"),
                    score=hit.get("score", 0.0),
                    highlights=hit.get("highlights"),
                    chunk_text=hit.get("chunk_text"),
                    chunk_id=hit.get("chunk_id"),
                    section_name=hit.get("section_name"),
                )
            )

        search_response = SearchResponse(
            query=request.query,
            total=results.get("total", 0),
            hits=hits,
            size=request.size,
            **{"from": request.from_},
            search_mode="bm25",
        )

        logger.info(f"Search completed: {search_response.total} results returned")
        return search_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
