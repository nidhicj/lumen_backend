import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import ingest, chat, sessions, drive

app = FastAPI(title="Lumen RAG API", 
              version="0.1.0",
              docs_url=None,
              redoc_url=None)
# lumen-frontend-topaz.vercel.app

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://lumen-frontend-topaz.vercel.app",  # ← replace with your actual Vercel URL
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router, prefix="/api/ingest", tags=["ingest"])
app.include_router(chat.router,   prefix="/api/chat",   tags=["chat"])
app.include_router(sessions.router, prefix="/api/sessions", tags=["sessions"])
app.include_router(drive.router, prefix="/api/drive", tags=["drive"])

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/debug-env")
def debug_env():
    or_key = os.getenv("OPENROUTER_API_KEY", "NOT SET")
    g_key  = os.getenv("GOOGLE_API_KEY", "NOT SET")
    return {
        "openrouter": or_key[:12] if or_key != "NOT SET" else "NOT SET",
        "google":     g_key[:12]  if g_key  != "NOT SET" else "NOT SET",
    }