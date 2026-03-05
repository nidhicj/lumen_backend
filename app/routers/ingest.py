import base64
import io
import re
import httpx

from fastapi import APIRouter, HTTPException
from app.models.schemas import IngestRequest, SourceType
from app.services.chunker import chunk_text
from app.services.store import add_chunks, add_source, get_sources

router = APIRouter()


async def _extract_pdf_text(b64_content: str) -> str:
    """Extract text from base64-encoded PDF using pdfminer."""
    try:
        from pdfminer.high_level import extract_text_to_fp
        from pdfminer.layout import LAParams
        pdf_bytes = base64.b64decode(b64_content)
        output = io.StringIO()
        extract_text_to_fp(io.BytesIO(pdf_bytes), output, laparams=LAParams())
        return output.getvalue()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"PDF extraction failed: {e}")


async def _fetch_url_text(url: str) -> str:
    """Fetch URL and strip HTML to plain text."""
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; LumenBot/1.0)"}
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            html = resp.text
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"URL fetch failed: {e}")

    # Strip scripts, styles, then all tags
    html = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<[^>]+>', ' ', html)
    # Collapse whitespace
    text = re.sub(r'[ \t]+', ' ', html)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()

    if len(text) < 100:
        raise HTTPException(status_code=422, detail="Page returned too little text. Try a different URL.")
    return text


@router.post("/")
async def ingest(req: IngestRequest):
    if len(get_sources(req.session_id)) >= 10:
        raise HTTPException(status_code=400, detail="Max 10 sources per session.")

    if req.source_type == SourceType.pdf:
        if not req.content:
            raise HTTPException(status_code=400, detail="content (base64 PDF) required.")
        text = await _extract_pdf_text(req.content)
        name = req.filename or "document.pdf"

    elif req.source_type == SourceType.url:
        if not req.url:
            raise HTTPException(status_code=400, detail="url required.")
        text = await _fetch_url_text(req.url)
        # Use hostname + path as display name
        clean = re.sub(r'^https?://(www\.)?', '', req.url).rstrip('/')
        name = clean[:60]

    elif req.source_type == SourceType.text:
        if not req.content:
            raise HTTPException(status_code=400, detail="content required.")
        text = req.content
        name = req.filename or "pasted-text.txt"

    else:
        raise HTTPException(status_code=400, detail="Unknown source type.")

    chunks = chunk_text(text, name, req.source_type)
    add_chunks(req.session_id, chunks)
    add_source(req.session_id, name)

    return {
        "status": "ok",
        "source": name,
        "chunks": len(chunks),
        "characters": len(text),
    }
