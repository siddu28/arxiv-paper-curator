"""
RAG Ask Router — Question answering over academic papers.

Week 4 Part 5: Non-streaming RAG endpoint (search -> retrieve -> prompt -> generate).
Week 4 Part 6: Streaming RAG endpoint added.
Week 6: Langfuse tracing and Redis caching will be layered on top.
"""

import json
import logging
import time
from typing import Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from src.dependencies import EmbeddingsDep, OllamaDep, OpenSearchDep
from src.schemas.api.ask import AskRequest, AskResponse

logger = logging.getLogger(__name__)

# Two separate routers - one for regular ask, one for streaming (Part 6)
ask_router = APIRouter(tags=["ask"])
stream_router = APIRouter(tags=["stream"])


async def _retrieve_chunks(
    request: AskRequest,
    opensearch_client,
    embeddings_service,
) -> tuple[List[Dict], List[str]]:
    """Retrieve and prepare chunks for RAG.

    :param request: Ask request with query and search parameters
    :param opensearch_client: OpenSearch client for search
    :param embeddings_service: Jina embeddings for query embedding
    :returns: Tuple of (chunks for LLM, source PDF URLs)
    """
    # Generate query embedding for hybrid search if requested
    query_embedding = None
    if request.use_hybrid:
        try:
            query_embedding = await embeddings_service.embed_query(request.query)
            logger.info("Generated query embedding for hybrid search")
        except Exception as e:
            logger.warning(f"Failed to generate embeddings, falling back to BM25: {e}")

    # Search for relevant chunks
    search_results = opensearch_client.search_unified(
        query=request.query,
        query_embedding=query_embedding,
        size=request.top_k,
        from_=0,
        categories=request.categories,
        use_hybrid=request.use_hybrid and query_embedding is not None,
        min_score=0.0,
    )

    # Extract chunk data for the LLM and source URLs for the response
    chunks = []
    sources_set = set()

    for hit in search_results.get("hits", []):
        arxiv_id = hit.get("arxiv_id", "")

        # Minimal chunk data — only what the LLM needs
        chunks.append(
            {
                "arxiv_id": arxiv_id,
                "chunk_text": hit.get("chunk_text", hit.get("abstract", "")),
            }
        )

        # Build PDF URL for source attribution
        if arxiv_id:
            arxiv_id_clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
            sources_set.add(f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf")

    return chunks, list(sources_set)


@ask_router.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    opensearch_client: OpenSearchDep,
    embeddings_service: EmbeddingsDep,
    ollama_client: OllamaDep,
) -> AskResponse:
    """RAG endpoint: search for relevant paper chunks, then generate an answer.

    Pipeline:
    1. Generate query embedding via Jina (if hybrid search enabled)
    2. Search OpenSearch for relevant paper chunks
    3. Build prompt with system instructions + retrieved context
    4. Generate answer via Ollama LLM
    5. Return answer with source PDFs

    :param request: AskRequest with query, top_k, model, etc.
    :param opensearch_client: OpenSearch search client
    :param embeddings_service: Jina embedding client
    :param ollama_client: Ollama LLM client
    :returns: AskResponse with answer, sources, and metadata
    """
    start_time = time.time()

    try:
        # Step 1-2: Retrieve relevant chunks
        chunks, sources = await _retrieve_chunks(
            request, opensearch_client, embeddings_service
        )

        if not chunks:
            return AskResponse(
                query=request.query,
                answer="I couldn't find any relevant information in the papers to answer your question.",
                sources=[],
                chunks_used=0,
                search_mode="bm25" if not request.use_hybrid else "hybrid",
            )

        logger.info(f"Retrieved {len(chunks)} chunks for query: '{request.query[:80]}...'")

        # Step 3-4: Generate answer via Ollama
        rag_response = await ollama_client.generate_rag_answer(
            query=request.query,
            chunks=chunks,
            model=request.model,
        )

        answer = rag_response.get("answer", "Unable to generate answer")

        elapsed = time.time() - start_time
        logger.info(f"RAG answer generated in {elapsed:.2f}s using {len(chunks)} chunks")

        # Step 5: Return structured response
        return AskResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            chunks_used=len(chunks),
            search_mode="bm25" if not request.use_hybrid else "hybrid",
        )

    except Exception as e:
        logger.error(f"Error processing RAG request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@stream_router.post("/stream")
async def ask_question_stream(
    request: AskRequest,
    opensearch_client: OpenSearchDep,
    embeddings_service: EmbeddingsDep,
    ollama_client: OllamaDep,
) -> StreamingResponse:
    """Streaming RAG endpoint — tokens arrive as they're generated.

    Uses Server-Sent Events (SSE) to stream the response:
    1. First event: metadata (sources, chunks_used, search_mode)
    2. Subsequent events: text chunks as they're generated
    3. Final event: complete answer with done=True

    :param request: AskRequest with query, top_k, model, etc.
    :param opensearch_client: OpenSearch search client
    :param embeddings_service: Jina embedding client
    :param ollama_client: Ollama LLM client
    :returns: StreamingResponse with SSE events
    """

    async def generate_stream():
        try:
            # Step 1-2: Retrieve relevant chunks
            chunks, sources = await _retrieve_chunks(
                request, opensearch_client, embeddings_service
            )

            if not chunks:
                yield f"data: {json.dumps({'answer': 'No relevant information found.', 'sources': [], 'done': True})}\n\n"
                return

            # Step 3: Send metadata first (before LLM starts generating)
            search_mode = "bm25" if not request.use_hybrid else "hybrid"
            metadata_response = {
                "sources": sources,
                "chunks_used": len(chunks),
                "search_mode": search_mode,
            }
            yield f"data: {json.dumps(metadata_response)}\n\n"

            # Step 4: Stream LLM generation token by token
            full_response = ""
            async for chunk in ollama_client.generate_rag_answer_stream(
                query=request.query,
                chunks=chunks,
                model=request.model,
            ):
                if chunk.get("response"):
                    text_chunk = chunk["response"]
                    full_response += text_chunk
                    yield f"data: {json.dumps({'chunk': text_chunk})}\n\n"

                if chunk.get("done", False):
                    # Step 5: Send completion signal with full answer
                    yield f"data: {json.dumps({'answer': full_response, 'done': True})}\n\n"
                    break

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
