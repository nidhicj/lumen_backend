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
    url: Optional[str] = None          # for URL ingestion
    content: Optional[str] = None      # for raw text / base64 pdf
    filename: Optional[str] = None

class Chunk(BaseModel):
    index: int
    text: str
    source_name: str
    source_type: SourceType

class ChatMessage(BaseModel):
    role: str   # "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    session_id: str
    question: str
    model: Optional[str] = "google/gemma-3-12b-it:free"

class CitedSource(BaseModel):
    index: int
    source_name: str
    snippet: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[CitedSource]

class Session(BaseModel):
    session_id: str
    sources: List[str] = []
    message_count: int = 0
