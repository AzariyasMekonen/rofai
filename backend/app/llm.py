from functools import lru_cache

import numpy as np
from openai import AsyncOpenAI, OpenAI

from .config import get_settings

EMBED_BATCH = 64


class NotConfiguredError(RuntimeError):
    pass


@lru_cache
def chat_client() -> AsyncOpenAI:
    s = get_settings()
    if not s.llm_configured:
        raise NotConfiguredError("LLM_BASE_URL and LLM_MODEL must be set in .env")
    # Ollama ignores the key, but the client requires a non-empty value
    return AsyncOpenAI(base_url=s.llm_base_url, api_key=s.llm_api_key or "unused")


@lru_cache
def embed_client() -> OpenAI:
    s = get_settings()
    base_url = s.embed_base_url or s.llm_base_url
    if not (base_url and s.embed_model):
        raise NotConfiguredError("EMBED_MODEL and an endpoint (EMBED_BASE_URL or LLM_BASE_URL) must be set")
    return OpenAI(base_url=base_url, api_key=s.embed_api_key or s.llm_api_key or "unused")


def embed(texts: list[str]) -> np.ndarray:
    """Return L2-normalised embeddings, shape (len(texts), dim)."""
    client = embed_client()
    model = get_settings().embed_model
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        resp = client.embeddings.create(model=model, input=batch)
        vectors.extend(item.embedding for item in sorted(resp.data, key=lambda d: d.index))
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms
