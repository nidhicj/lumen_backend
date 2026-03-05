import os
from fastapi import APIRouter, HTTPException
from app.models.schemas import ChatRequest, ChatResponse, ChatMessage, CitedSource
from app.services.store import get_chunks, get_history, append_history, get_sources
from app.services.chunker import retrieve
from app.services.llm import call_openrouter

router = APIRouter()


@router.post("/", response_model=ChatResponse)
async def chat(req: ChatRequest):
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not set.")

    chunks = get_chunks(req.session_id)
    if not chunks:
        raise HTTPException(status_code=400, detail="No documents ingested for this session.")

    history  = get_history(req.session_id)
    top      = retrieve(req.question, chunks, top_k=6)
    top_chunks = [c for c, _ in top]

    answer = await call_openrouter(
        question=req.question,
        context_chunks=top_chunks,
        history=history,
        api_key=api_key,
        model=req.model,
    )

    # Persist conversation turn
    append_history(req.session_id, ChatMessage(role="user",      content=req.question))
    append_history(req.session_id, ChatMessage(role="assistant", content=answer))

    sources = [
        CitedSource(
            index=i + 1,
            source_name=c.source_name,
            snippet=c.text[:140] + "…",
        )
        for i, c in enumerate(top_chunks)
    ]

    return ChatResponse(answer=answer, sources=sources)


@router.get("/{session_id}/history")
def get_chat_history(session_id: str):
    return {"history": get_history(session_id)}
