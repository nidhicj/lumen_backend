import os
import re
import io
import httpx
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

DRIVE_API   = "https://www.googleapis.com/drive/v3"
EXPORT_API  = "https://www.googleapis.com/drive/v3/files/{id}/export"
DOWNLOAD_API = "https://www.googleapis.com/drive/v3/files/{id}?alt=media"

# MIME types we support ingesting
SUPPORTED_MIME = {
    "application/pdf":                                          "pdf",
    "text/plain":                                               "text",
    "text/markdown":                                            "text",
    "application/vnd.google-apps.document":                    "gdoc",   # export as txt
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
}


def extract_folder_id(url: str) -> str | None:
    """
    Handles all common Google Drive folder URL formats:
      https://drive.google.com/drive/folders/FOLDER_ID
      https://drive.google.com/drive/folders/FOLDER_ID?usp=sharing
      https://drive.google.com/open?id=FOLDER_ID
    """
    patterns = [
        r"drive\.google\.com/drive/folders/([a-zA-Z0-9_-]+)",
        r"drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


async def list_folder_files(folder_id: str, api_key: str) -> List[Dict]:
    """
    Returns list of supported files in a public Google Drive folder.
    Handles pagination automatically.
    """
    files = []
    page_token = None

    async with httpx.AsyncClient(timeout=30) as client:
        while True:
            params = {
                "q": f"'{folder_id}' in parents and trashed=false",
                "fields": "nextPageToken, files(id, name, mimeType, size)",
                "key": api_key,
                "pageSize": 100,
            }
            if page_token:
                params["pageToken"] = page_token

            resp = await client.get(f"{DRIVE_API}/files", params=params)

            if resp.status_code == 403:
                raise Exception(
                    "Folder is not public. Please set sharing to 'Anyone with the link can view'."
                )
            if not resp.is_success:
                raise Exception(f"Drive API error {resp.status_code}: {resp.text}")

            data = resp.json()

            # Filter to supported file types only
            for f in data.get("files", []):
                if f.get("mimeType") in SUPPORTED_MIME:
                    files.append(f)
                else:
                    logger.info(f"Skipping unsupported file: {f['name']} ({f.get('mimeType')})")

            page_token = data.get("nextPageToken")
            if not page_token:
                break

    return files


async def download_file(file_id: str, mime_type: str, api_key: str) -> tuple[str, str]:
    """
    Downloads a file and returns (text_content, file_type).
    Google Docs are exported as plain text.
    PDFs are returned as base64 for pdfminer.
    """
    import base64

    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:

        if mime_type == "application/vnd.google-apps.document":
            # Export Google Doc as plain text
            resp = await client.get(
                EXPORT_API.format(id=file_id),
                params={"mimeType": "text/plain", "key": api_key}
            )
        else:
            # Direct download for PDF, txt, docx
            resp = await client.get(
                DOWNLOAD_API.format(id=file_id),
                params={"key": api_key}
            )

        if not resp.is_success:
            raise Exception(f"Download failed {resp.status_code}: {resp.text[:200]}")

        if mime_type == "application/pdf":
            # Return base64 so ingest router can run pdfminer on it
            return base64.b64encode(resp.content).decode(), "pdf"
        else:
            # Plain text (txt, md, exported gdoc)
            return resp.text, "text"