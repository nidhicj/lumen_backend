import os
import asyncio
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.gdrive import extract_folder_id, list_folder_files, download_file
from app.services.chunker import chunk_text
from app.services.store import add_chunks, add_source, get_sources
from app.models.schemas import SourceType

logger = logging.getLogger(__name__)
router = APIRouter()


def _drive_url(file_id: str, mime_type: str) -> str:
    """Construct a public viewable URL for a Google Drive file."""
    if mime_type == "application/vnd.google-apps.document":
        return f"https://docs.google.com/document/d/{file_id}/edit"
    return f"https://drive.google.com/file/d/{file_id}/view"


class DriveIngestRequest(BaseModel):
    session_id: str
    folder_url: str


@router.post("/")
async def ingest_drive_folder(req: DriveIngestRequest):
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not set on server.")

    # Extract folder ID from URL
    folder_id = extract_folder_id(req.folder_url)
    if not folder_id:
        raise HTTPException(status_code=400, detail="Could not extract folder ID from URL. "
                            "Make sure it's a valid Google Drive folder link.")

    # Check session source limit
    if len(get_sources(req.session_id)) >= 10:
        raise HTTPException(status_code=400, detail="Max 10 sources per session.")

    # List files in folder
    try:
        files = await list_folder_files(folder_id, api_key)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    if not files:
        raise HTTPException(
            status_code=422,
            detail="No supported files found in folder. "
                   "Supported types: PDF, TXT, MD, Google Docs, Word (.docx)"
        )

    # Cap at 20 files for MVP
    if len(files) > 20:
        files = files[:20]
        logger.warning(f"Folder has >20 files, capped at 20 for session {req.session_id}")

    results = []
    errors  = []

    # Process files — limit concurrency to 3 at a time to avoid rate limits
    semaphore = asyncio.Semaphore(3)

    async def process_file(f):
        async with semaphore:
            try:
                text, file_type = await download_file(f["id"], f["mimeType"], api_key)
                source_type = SourceType.pdf if file_type == "pdf" else SourceType.text

                if file_type == "pdf":
                    # Re-use PDF extraction from ingest router
                    from app.routers.ingest import _extract_pdf_text
                    text = await _extract_pdf_text(text)

                if not text or len(text.strip()) < 50:
                    errors.append({"file": f["name"], "error": "Too little text extracted"})
                    return

                source_url = _drive_url(f["id"], f["mimeType"])
                chunks = chunk_text(text, f["name"], source_type, source_url=source_url)
                add_chunks(req.session_id, chunks)
                add_source(req.session_id, f["name"])
                results.append({"file": f["name"], "chunks": len(chunks)})
                logger.info(f"Ingested {f['name']}: {len(chunks)} chunks")

            except Exception as e:
                errors.append({"file": f["name"], "error": str(e)})
                logger.warning(f"Failed to ingest {f['name']}: {e}")

    await asyncio.gather(*[process_file(f) for f in files])

    if not results:
        raise HTTPException(
            status_code=422,
            detail=f"Failed to ingest any files. Errors: {errors}"
        )

    total_chunks = sum(r["chunks"] for r in results)

    return {
        "status": "ok",
        "folder_id": folder_id,
        "files_found": len(files),
        "files_ingested": len(results),
        "files_failed": len(errors),
        "total_chunks": total_chunks,
        "ingested": results,
        "errors": errors,
    }