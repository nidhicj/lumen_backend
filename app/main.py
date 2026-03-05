from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import ingest, chat, sessions
import os
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Lumen RAG API", version="0.1.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router, prefix="/api/ingest", tags=["ingest"])
app.include_router(chat.router,   prefix="/api/chat",   tags=["chat"])
app.include_router(sessions.router, prefix="/api/sessions", tags=["sessions"])

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/debug-env")
def debug_env():
    key = os.getenv("OPENROUTER_API_KEY", "NOT SET")
    return {"key_set": key != "NOT SET", "prefix": key[:10] if key != "NOT SET" else ""}