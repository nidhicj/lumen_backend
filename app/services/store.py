"""
Simple in-memory store.
For production: swap chunk_store → Qdrant/Pinecone, session_store → Redis/Postgres.
"""
from typing import Dict, List
from app.models.schemas import Chunk, ChatMessage

# session_id → list of chunks
chunk_store: Dict[str, List[Chunk]] = {}

# session_id → conversation history
session_history: Dict[str, List[ChatMessage]] = {}

# session_id → list of source names
session_sources: Dict[str, List[str]] = {}


def get_chunks(session_id: str) -> List[Chunk]:
    return chunk_store.get(session_id, [])

def add_chunks(session_id: str, chunks: List[Chunk]):
    if session_id not in chunk_store:
        chunk_store[session_id] = []
    chunk_store[session_id].extend(chunks)

def get_history(session_id: str) -> List[ChatMessage]:
    return session_history.get(session_id, [])

def append_history(session_id: str, msg: ChatMessage):
    if session_id not in session_history:
        session_history[session_id] = []
    session_history[session_id].append(msg)

def get_sources(session_id: str) -> List[str]:
    return session_sources.get(session_id, [])

def add_source(session_id: str, name: str):
    if session_id not in session_sources:
        session_sources[session_id] = []
    if name not in session_sources[session_id]:
        session_sources[session_id].append(name)

def clear_session(session_id: str):
    chunk_store.pop(session_id, None)
    session_history.pop(session_id, None)
    session_sources.pop(session_id, None)

def list_sessions():
    all_ids = set(chunk_store) | set(session_history)
    return [
        {
            "session_id": sid,
            "sources": session_sources.get(sid, []),
            "message_count": len(session_history.get(sid, []))
        }
        for sid in all_ids
    ]
