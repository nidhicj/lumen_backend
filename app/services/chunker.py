import re
from typing import List, Tuple
from app.models.schemas import Chunk, SourceType

CHUNK_SIZE    = 800   # characters
CHUNK_OVERLAP = 150


def chunk_text(text: str, source_name: str, source_type: SourceType) -> List[Chunk]:
    """Split text into overlapping chunks, preserving paragraph boundaries where possible."""
    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text.strip())

    chunks: List[Chunk] = []
    start = 0
    idx   = 0

    while start < len(text):
        end = start + CHUNK_SIZE

        if end < len(text):
            # Try to break on paragraph, then sentence, then word
            para_break = text.rfind('\n\n', start, end)
            sent_break = max(text.rfind('. ', start, end),
                             text.rfind('! ', start, end),
                             text.rfind('? ', start, end))
            word_break = text.rfind(' ', start, end)

            if para_break > start + CHUNK_SIZE // 2:
                end = para_break
            elif sent_break > start + CHUNK_SIZE // 2:
                end = sent_break + 1
            elif word_break > start:
                end = word_break

        chunk_text = text[start:end].strip()
        if len(chunk_text) > 60:          # skip tiny fragments
            chunks.append(Chunk(
                index=idx,
                text=chunk_text,
                source_name=source_name,
                source_type=source_type,
            ))
            idx += 1

        start = end - CHUNK_OVERLAP       # overlap window

    return chunks


# ── Retrieval (TF-IDF keyword, swap for vector search in v2) ──────────────

def _tokenize(text: str) -> List[str]:
    return re.findall(r'\b[a-z]{3,}\b', text.lower())

def _score(query_tokens: List[str], chunk: Chunk) -> float:
    chunk_tokens = _tokenize(chunk.text)
    if not chunk_tokens:
        return 0.0
    freq = {}
    for t in chunk_tokens:
        freq[t] = freq.get(t, 0) + 1
    hits = sum(freq.get(t, 0) for t in query_tokens)
    return hits / len(chunk_tokens)

def retrieve(query: str, chunks: List[Chunk], top_k: int = 6) -> List[Tuple[Chunk, float]]:
    """Return top_k chunks ranked by keyword overlap with query."""
    q_tokens = _tokenize(query)
    if not q_tokens:
        return [(c, 0.0) for c in chunks[:top_k]]
    scored = [(c, _score(q_tokens, c)) for c in chunks]
    scored.sort(key=lambda x: x[1], reverse=True)
    # Always return at least top_k even if scores are 0
    return scored[:top_k]
