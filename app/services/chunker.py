import re
import math
from typing import List, Tuple
from app.models.schemas import Chunk, SourceType

CHUNK_SIZE    = 600   # smaller chunks = more precise retrieval
CHUNK_OVERLAP = 200   # bigger overlap = less chance of splitting answers


def chunk_text(text: str, source_name: str, source_type: SourceType, source_url: str = None) -> List[Chunk]:
    """Split text into overlapping chunks, preserving paragraph boundaries."""
    # Normalize whitespace but preserve paragraph structure
    text = re.sub(r'\n{3,}', '\n\n', text.strip())
    text = re.sub(r'[ \t]+', ' ', text)

    chunks: List[Chunk] = []
    start = 0
    idx   = 0

    while start < len(text):
        end = start + CHUNK_SIZE

        if end < len(text):
            para_break = text.rfind('\n\n', start, end)
            sent_break = max(
                text.rfind('. ', start, end),
                text.rfind('! ', start, end),
                text.rfind('? ', start, end),
                text.rfind('.\n', start, end),
            )
            word_break = text.rfind(' ', start, end)

            if para_break > start + CHUNK_SIZE // 2:
                end = para_break
            elif sent_break > start + CHUNK_SIZE // 2:
                end = sent_break + 1
            elif word_break > start:
                end = word_break

        chunk = text[start:end].strip()
        if len(chunk) > 40:
            chunks.append(Chunk(
                index=idx,
                text=chunk,
                source_name=source_name,
                source_type=source_type,
                source_url=source_url,
            ))
            idx += 1

        start = end - CHUNK_OVERLAP

    return chunks


# ── Retrieval ────────────────────────────────────────────────────────────────

# Common words that add no retrieval signal — skip them
STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "is","was","are","were","be","been","has","have","had","that","this",
    "it","its","they","their","them","we","our","you","your","from","by",
    "not","also","can","will","would","could","should","may","might","does",
    "did","do","said","says","about","which","who","when","where","how",
}

def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r'\b[a-z]{2,}\b', text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def _tfidf_score(query_tokens: List[str], chunk: Chunk, all_chunks: List[Chunk]) -> float:
    """
    Proper TF-IDF score:
    - TF: how often query term appears in this chunk
    - IDF: penalises terms that appear in every chunk (not discriminative)
    """
    chunk_tokens = _tokenize(chunk.text)
    if not chunk_tokens:
        return 0.0

    # Term frequency in this chunk
    tf: dict = {}
    for t in chunk_tokens:
        tf[t] = tf.get(t, 0) + 1

    # Document frequency across all chunks
    N = len(all_chunks)
    score = 0.0
    for qt in query_tokens:
        if qt not in tf:
            continue
        term_tf = tf[qt] / len(chunk_tokens)
        # Count how many chunks contain this term
        df = sum(1 for c in all_chunks if qt in _tokenize(c.text))
        idf = math.log((N + 1) / (df + 1)) + 1
        score += term_tf * idf

    return score


def _context_stitch(
    chunk: Chunk,
    all_chunks: List[Chunk],
    window: int = 1
) -> Chunk:
    """
    Returns a new chunk that includes neighbouring chunks for context.
    This ensures answers that span chunk boundaries are captured.
    """
    same_source = [c for c in all_chunks if c.source_name == chunk.source_name]
    same_source.sort(key=lambda c: c.index)

    idx_in_source = next((i for i, c in enumerate(same_source) if c.index == chunk.index), None)
    if idx_in_source is None:
        return chunk

    start = max(0, idx_in_source - window)
    end   = min(len(same_source), idx_in_source + window + 1)
    neighbours = same_source[start:end]

    stitched_text = "\n\n".join(c.text for c in neighbours)

    return Chunk(
        index=chunk.index,
        text=stitched_text,
        source_name=chunk.source_name,
        source_type=chunk.source_type,
        source_url=chunk.source_url,   
    )


def retrieve(query: str, chunks: List[Chunk], top_k: int = 8) -> List[Tuple[Chunk, float]]:
    """
    Retrieve top_k most relevant chunks using TF-IDF + context stitching.
    - Scores all chunks with proper TF-IDF (not just keyword hit count)
    - Expands each retrieved chunk with its neighbours for better context
    - Returns more chunks (8 vs 6) to reduce "not found" false negatives
    """
    q_tokens = _tokenize(query)
    if not q_tokens:
        # No meaningful tokens — return spread of chunks across all sources
        step = max(1, len(chunks) // top_k)
        return [(chunks[i], 0.0) for i in range(0, len(chunks), step)][:top_k]

    # Score all chunks
    scored = [(c, _tfidf_score(q_tokens, c, chunks)) for c in chunks]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Take top candidates
    top_candidates = scored[:top_k]

    # Stitch each retrieved chunk with its neighbours
    stitched = [
        (_context_stitch(c, chunks, window=1), score)
        for c, score in top_candidates
    ]

    return stitched