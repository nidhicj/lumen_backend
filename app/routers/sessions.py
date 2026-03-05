from fastapi import APIRouter
from app.services.store import list_sessions, clear_session, get_sources, get_history

router = APIRouter()

@router.get("/")
def sessions():
    return {"sessions": list_sessions()}

@router.get("/{session_id}")
def session_detail(session_id: str):
    return {
        "session_id": session_id,
        "sources": get_sources(session_id),
        "message_count": len(get_history(session_id)),
        "history": get_history(session_id),
    }

@router.delete("/{session_id}")
def delete_session(session_id: str):
    clear_session(session_id)
    return {"status": "deleted", "session_id": session_id}
