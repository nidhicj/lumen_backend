import os
import httpx
import logging
from typing import List
from app.models.schemas import ChatMessage

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

# Used only if the live OpenRouter model list can't be fetched at startup
# (e.g. OpenRouter is briefly down/slow while the container boots).
_STARTUP_FALLBACK_CHAIN = [
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
]


def _fetch_free_models() -> List[str]:
    resp = httpx.get(OPENROUTER_MODELS_URL, timeout=10)
    resp.raise_for_status()
    models = resp.json()["data"]
    # Any genuinely free, text-only-output model counts — no provider
    # allowlist, so the chain stays broad across providers instead of
    # drifting stale as OpenRouter's free-tier lineup changes. The
    # output_modalities check excludes non-chat models (e.g. Lyria, which
    # is free but outputs audio, not text).
    return [
        m["id"] for m in models
        if str(m.get("pricing", {}).get("prompt", "-1")) == "0"
        and m.get("architecture", {}).get("output_modalities") == ["text"]
    ]


try:
    FALLBACK_CHAIN = _fetch_free_models()
    if not FALLBACK_CHAIN:
        raise ValueError("OpenRouter returned no free models")
    for model_id in FALLBACK_CHAIN:
        logger.info(f"Free model available: {model_id}")
except Exception as e:
    logger.error(f"Could not fetch live free-model list from OpenRouter, using static fallback: {e}")
    FALLBACK_CHAIN = _STARTUP_FALLBACK_CHAIN


# Ordered fallback chain — if one fails, next is tried automatically
# FALLBACK_CHAIN = [
#     "arcee-ai/trinity-large-preview:free",
#     "meta-llama/llama-3.2-3b-instruct:free",
#     "mistralai/mistral-small-3.1-24b-instruct:free",
#     "google/gemma-3-4b-it:free",
# ]

# FALLBACK_CHAIN = [
#     "meta-llama/llama-3.3-70b-instruct:free",       # fastest, ~3-5s
#     "google/gemma-4-31b-it:free",                    # second fastest
    # "mistralai/mistral-small-3.2-24b-instruct",
    # "arcee-ai/trinity-large-preview",          # slowest, best quality
# ]

DEFAULT_MODEL = FALLBACK_CHAIN[0]

# Models that don't support a system role — must merge into first user message
NO_SYSTEM_ROLE_MODELS = {
    "google/gemma-3-4b-it:free",
    "google/gemma-3-12b-it:free",
    "google/gemma-3-27b-it:free",
}


def _build_messages(question: str, context_chunks: list, history: List[ChatMessage], model: str):
    system_content = (
        "You are a precise document intelligence assistant. "
        "Your job is to answer questions using ONLY the document excerpts provided. "
        "Rules you must follow:\n"
        "1. Always cite inline using [1], [2], etc. matching the excerpt number\n"
        "2. Synthesize across multiple excerpts when the answer spans several chunks\n"
        "3. If information is partially in the excerpts, share what you found and note what is missing\n"
        "4. Only say \'not found\' if you have genuinely searched all excerpts and found nothing relevant\n"
        "5. Never make up information not present in the excerpts\n"
        "6. Be thorough — a complete answer is better than a short one"
    )

    rag_context = "\n\n---\n\n".join(
        f"[{i+1}] Source: \"{c.source_name}\"\n{c.text}"
        for i, c in enumerate(context_chunks)
    )

    rag_prompt = (
        f"Here are the most relevant excerpts from the documents:\n\n"
        f"{rag_context}\n\n"
        f"---\n\n"
        f"Using the excerpts above, answer this question thoroughly.\n"
        f"Cite every claim with [excerpt number].\n"
        f"If the answer spans multiple excerpts, synthesize them into one complete answer.\n\n"
        f"Question: {question}"
    )

    history_msgs = [
        {"role": m.role, "content": m.content}
        for m in history[-6:]
    ]

    if model in NO_SYSTEM_ROLE_MODELS:
        # Gemma doesn't support system role — merge into first user message
        first_user = f"{system_content}\n\n{rag_prompt}"
        return history_msgs + [{"role": "user", "content": first_user}]
    else:
        return (
            [{"role": "system", "content": system_content}]
            + history_msgs
            + [{"role": "user", "content": rag_prompt}]
        )


async def _try_model(model: str, messages: list, api_key: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://lumen-rag.vercel.app",
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

    if resp.is_success:
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    raise Exception(f"{resp.status_code}: {resp.text}")


# async def call_openrouter(
#     question: str,
#     context_chunks: list,
#     history: List[ChatMessage],
#     api_key: str,
#     model: str = DEFAULT_MODEL,
# ) -> str:
#     # Requested model first, then fallbacks
#     chain = [model] + [m for m in FALLBACK_CHAIN if m != model]

#     last_error = None
#     for attempt_model in chain:
#         try:
#             messages = _build_messages(question, context_chunks, history, attempt_model)
#             logger.info(f"Trying model: {attempt_model}")
#             answer = await _try_model(attempt_model, messages, api_key)
#             if attempt_model != model:
#                 logger.warning(f"Fell back to {attempt_model} (original: {model})")
#             return answer
#         except Exception as e:
#             last_error = e
#             logger.warning(f"Model {attempt_model} failed: {e}")
#             continue

#     raise Exception(f"All models failed. Last error: {last_error}")

async def call_openrouter(question: str,
    context_chunks: list,
    history: List[ChatMessage],
    api_key: str,
    model: str = DEFAULT_MODEL,) -> tuple[str, str]:  # (answer, model_used)
    chain = [model] + [m for m in FALLBACK_CHAIN if m != model]

    last_error = None
    for attempt_model in chain:
        try:
            messages = _build_messages(question, context_chunks, history, attempt_model)
            logger.info(f"Trying model: {attempt_model}")
            answer = await _try_model(attempt_model, messages, api_key)
            if attempt_model != model:
                logger.warning(f"Fell back to {attempt_model} (original: {model})")
            return answer, attempt_model  # ← return both
        except Exception as e:
            last_error = e
            logger.warning(f"Model {attempt_model} failed: {e}")
            continue

    raise Exception(f"All models failed. Last error: {last_error}")

if "__name__" == "__main__":
    import asyncio
    async def test():
        answer, model_used = await call_openrouter(
            question="What is the capital of France?",
            context_chunks=[],
            history=[],
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        print(f"Model used: {model_used}\nAnswer: {answer}")
    asyncio.run(test())