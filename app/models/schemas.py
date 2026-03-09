from pydantic import BaseModel
from typing import Optional, List
from enum import Enum

class SourceType(str, Enum):
    pdf  = "pdf"
    url  = "url"
    text = "text"

class IngestRequest(BaseModel):
    session_id: str
    source_type: SourceType
    url: Optional[str] = None
    content: Optional[str] = None
    filename: Optional[str] = None

class Chunk(BaseModel):
    index: int
    text: str
    source_name: str
    source_type: SourceType
    source_url: Optional[str] = None   # ← clickable link back to origin

class ChatMessage(BaseModel):
    role: str   # "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    session_id: str
    question: str
    model: Optional[str] = "meta-llama/llama-3.2-3b-instruct:free"

class CitedSource(BaseModel):
    index: int
    source_name: str
    snippet: str
    source_url: Optional[str] = None   # ← passed to frontend for link rendering

class ChatResponse(BaseModel):
    answer: str
    sources: List[CitedSource]
    model_used: str