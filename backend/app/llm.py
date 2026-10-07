from collections.abc import AsyncIterator
from functools import lru_cache

import numpy as np
from openai import AsyncOpenAI, OpenAI

from .config import get_settings
from .copilot_llm import CopilotChat

EMBED_BATCH = 64


class NotConfiguredError(RuntimeError):
    pass


def ensure_llm_configured() -> None:
    s = get_settings()
    if s.llm_configured:
        return
    if s.llm_provider == "copilot":
        raise NotConfiguredError("COPILOT_GITHUB_TOKEN must be set in .env")
    raise NotConfiguredError("LLM_BASE_URL and LLM_MODEL must be set in .env")


@lru_cache
def copilot_chat() -> CopilotChat:
    return CopilotChat(get_settings())


@lru_cache
def chat_client() -> AsyncOpenAI:
    s = get_settings()
    ensure_llm_configured()
    # Ollama ignores the key, but the client requires a non-empty value
    return AsyncOpenAI(base_url=s.llm_base_url, api_key=s.llm_api_key or "unused")


async def stream_chat(messages: list[dict[str, str]], max_tokens: int = 350) -> AsyncIterator[str]:
    ensure_llm_configured()
    s = get_settings()
    if s.llm_provider == "copilot":
        async for text in copilot_chat().stream(messages):
            yield text
        return
    completion = await chat_client().chat.completions.create(
        model=s.llm_model, messages=messages, stream=True, temperature=0.3, max_tokens=max_tokens
    )
    async for chunk in completion:
        # Azure sends content-filter frames with no choices
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


async def complete(messages: list[dict[str, str]], max_tokens: int = 100) -> str:
    return "".join([t async for t in stream_chat(messages, max_tokens)]).strip()


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
