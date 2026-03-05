import os
import httpx
from typing import List
from app.models.schemas import ChatMessage

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL  = "arcee-ai/trinity-large-preview:free"

FREE_MODELS = [
    "arcee-ai/trinity-large-preview:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "google/gemma-3-4b-it:free",
]


def _build_rag_prompt(question: str, context_chunks: list) -> str:
    context = "\n\n---\n\n".join(
        f"[{i+1}] Source: \"{c.source_name}\"\n{c.text}"
        for i, c in enumerate(context_chunks)
    )
    return f"""You are a precise document assistant. Answer using ONLY the document excerpts below.

Rules:
- Cite sources inline as [1], [2], etc. matching excerpt numbers
- If the answer is not in the excerpts, say: "I couldn't find this in the provided documents."
- Be concise but complete. Use markdown for structure when helpful.

Document excerpts:
{context}

---

Question: {question}"""


async def call_openrouter(
    question: str,
    context_chunks: list,
    history: List[ChatMessage],
    api_key: str,
    model: str = DEFAULT_MODEL,
) -> str:
    system_msg = {
        "role": "system",
        "content": (
            "You are Lumen, a helpful document intelligence assistant. "
            "You answer questions strictly based on the documents provided. "
            "Always cite your sources using [1], [2] notation."
        )
    }

    # Build messages: system + last 6 turns of history + current RAG prompt
    history_msgs = [
        {"role": m.role, "content": m.content}
        for m in history[-6:]
    ]
    user_msg = {"role": "user", "content": _build_rag_prompt(question, context_chunks)}

    messages = [system_msg] + history_msgs + [user_msg]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://lumen-rag.vercel.app",  # update with your domain
        "X-Title": "Lumen RAG",
    }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.2,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(OPENROUTER_URL, json=payload, headers=headers)
        if not resp.is_success:
            raise Exception(f"OpenRouter {resp.status_code}: {resp.text}")
        data = resp.json()

    return data["choices"][0]["message"]["content"]
