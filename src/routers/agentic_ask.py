"""Agentic RAG Router — intelligent question answering with LangGraph.

Week 6: Agentic RAG endpoint with guardrail validation, retrieval,
document grading, query rewriting, and answer generation.
"""

import logging

from fastapi import APIRouter, HTTPException
from src.dependencies import AgenticRAGDep
from src.schemas.api.ask import AgenticAskResponse, AskRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["agentic-rag"])


@router.post("/ask-agentic", response_model=AgenticAskResponse)
async def ask_agentic(
    request: AskRequest,
    agentic_rag: AgenticRAGDep,
) -> AgenticAskResponse:
    """Agentic RAG endpoint with intelligent retrieval and query refinement.

    The agent automatically:
    1. Validates query scope via guardrail (score 0-100)
    2. Retrieves relevant papers from OpenSearch
    3. Grades document relevance with LLM
    4. Rewrites query if documents aren't relevant (up to 2 retries)
    5. Generates answer with citations from verified context

    :param request: Question and search parameters
    :param agentic_rag: Injected AgenticRAGService
    :returns: Answer with sources, reasoning steps, and metadata
    """
    try:
        result = await agentic_rag.ask(
            query=request.query,
        )

        return AgenticAskResponse(
            query=result["query"],
            answer=result["answer"],
            sources=result.get("sources", []),
            chunks_used=request.top_k,
            search_mode="hybrid" if request.use_hybrid else "bm25",
            reasoning_steps=result.get("reasoning_steps", []),
            retrieval_attempts=result.get("retrieval_attempts", 0),
            trace_id=result.get("trace_id"),
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error in agentic RAG: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")
