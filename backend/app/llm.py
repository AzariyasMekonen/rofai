from collections.abc import AsyncIterator
from functools import lru_cache
import logging

import numpy as np
import openai
from openai import AsyncOpenAI, OpenAI

from .config import get_settings
from .copilot_llm import CopilotChat

log = logging.getLogger(__name__)
EMBED_BATCH = 64


class NotConfiguredError(RuntimeError):
    pass


class KeyRotator:
    def __init__(self, base_url: str, keys: list[str]):
        self.base_url = base_url
        self.keys = [k.strip() for k in keys if k.strip()]
        self._current_idx = 0
        self._client_cache: dict[str, AsyncOpenAI] = {}
        self._sync_client_cache: dict[str, OpenAI] = {}

    @property
    def current_key(self) -> str:
        if not self.keys:
            return "unused"
        return self.keys[self._current_idx % len(self.keys)]

    def get_client(self) -> AsyncOpenAI:
        key = self.current_key
        client = self._client_cache.get(key)
        if client is None:
            client = AsyncOpenAI(base_url=self.base_url, api_key=key)
            self._client_cache[key] = client
        return client

    def get_sync_client(self) -> OpenAI:
        key = self.current_key
        client = self._sync_client_cache.get(key)
        if client is None:
            client = OpenAI(base_url=self.base_url, api_key=key)
            self._sync_client_cache[key] = client
        return client

    def rotate_key(self, reason: str = "rate limit") -> str:
        if not self.keys or len(self.keys) <= 1:
            return self.current_key
        old_key = self.current_key
        self._current_idx = (self._current_idx + 1) % len(self.keys)
        new_key = self.current_key
        log.warning(
            "Rotating LLM API key (%s). Switched from ...%s to key %d/%d (...%s)",
            reason,
            old_key[-6:] if len(old_key) > 6 else "...",
            (self._current_idx % len(self.keys)) + 1,
            len(self.keys),
            new_key[-6:] if len(new_key) > 6 else "...",
        )
        return new_key

    def remove_current_key(self, reason: str = "invalid key") -> str:
        if not self.keys:
            return "unused"
        bad_key = self.current_key
        log.warning("Discarding invalid API key ...%s (%s)", bad_key[-6:], reason)
        self.keys.remove(bad_key)
        self._client_cache.pop(bad_key, None)
        self._sync_client_cache.pop(bad_key, None)
        if not self.keys:
            return "unused"
        self._current_idx = self._current_idx % len(self.keys)
        log.warning("Remaining keys in pool: %d. Switched to ...%s", len(self.keys), self.current_key[-6:])
        return self.current_key



def is_rotatable_error(exc: Exception) -> bool:
    if isinstance(exc, (openai.RateLimitError, openai.AuthenticationError, openai.PermissionDeniedError)):
        return True
    if isinstance(exc, openai.APIStatusError):
        if exc.status_code in (401, 403, 429, 500, 502, 503, 504):
            return True
        msg = str(exc).lower()
        if any(
            term in msg
            for term in (
                "rate limit",
                "quota",
                "resource_exhausted",
                "high demand",
                "overloaded",
                "unauthenticated",
                "invalid_api_key",
                "permission",
                "access_token",
            )
        ):
            return True
    return False


def is_auth_error(exc: Exception) -> bool:
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return True
    if isinstance(exc, openai.APIStatusError) and exc.status_code in (401, 403):
        return True
    msg = str(exc).lower()
    return any(
        term in msg
        for term in (
            "unauthenticated",
            "invalid_api_key",
            "access_token",
            "permission",
        )
    )


def rotate_or_remove(rotator: KeyRotator, exc: Exception, context: str = "") -> str:
    ctx = f"{context} " if context else ""
    if is_auth_error(exc):
        return rotator.remove_current_key(reason=f"{ctx}auth error: {exc}")
    return rotator.rotate_key(reason=f"{ctx}{type(exc).__name__}: {exc}")




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
def key_rotator() -> KeyRotator:
    s = get_settings()
    ensure_llm_configured()
    keys = s.api_keys
    if not keys and s.llm_api_key:
        keys = [s.llm_api_key]
    return KeyRotator(base_url=s.llm_base_url, keys=keys)


def chat_client() -> AsyncOpenAI:
    return key_rotator().get_client()


async def stream_chat(messages: list[dict[str, str]], max_tokens: int = 350) -> AsyncIterator[str]:
    ensure_llm_configured()
    s = get_settings()
    if s.llm_provider == "copilot":
        async for text in copilot_chat().stream(messages):
            yield text
        return

    rotator = key_rotator()
    max_attempts = max(1, len(rotator.keys)) if rotator.keys else 1

    for attempt in range(max_attempts):
        client = rotator.get_client()
        started_streaming = False
        try:
            completion = await client.chat.completions.create(
                model=s.llm_model, messages=messages, stream=True, temperature=0.3, max_tokens=max_tokens
            )
            async for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    started_streaming = True
                    yield chunk.choices[0].delta.content
            return
        except Exception as exc:
            if (is_rotatable_error(exc) or is_auth_error(exc)) and attempt < max_attempts - 1 and not started_streaming:
                rotate_or_remove(rotator, exc, context="stream_chat")
                continue
            raise



async def complete(messages: list[dict[str, str]], max_tokens: int = 150) -> str:
    return "".join([t async for t in stream_chat(messages, max_tokens)]).strip()


@lru_cache
def embed_client() -> OpenAI:
    s = get_settings()
    base_url = s.embed_base_url or s.llm_base_url
    if not (base_url and s.embed_model):
        raise NotConfiguredError("EMBED_MODEL and an endpoint (EMBED_BASE_URL or LLM_BASE_URL) must be set")
    # Use first available key if rotating keys exist
    api_key = s.embed_api_key or (s.api_keys[0] if s.api_keys else s.llm_api_key) or "unused"
    return OpenAI(base_url=base_url, api_key=api_key)


def embed(texts: list[str]) -> np.ndarray:
    """Return L2-normalised embeddings, shape (len(texts), dim)."""
    s = get_settings()
    base_url = s.embed_base_url or s.llm_base_url
    if not (base_url and s.embed_model):
        raise NotConfiguredError("EMBED_MODEL and an endpoint (EMBED_BASE_URL or LLM_BASE_URL) must be set")
    model = s.embed_model
    rotator = key_rotator()
    vectors: list[list[float]] = []

    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        max_attempts = max(1, len(rotator.keys)) if rotator.keys else 1
        resp = None
        for attempt in range(max_attempts):
            client = rotator.get_sync_client()
            try:
                resp = client.embeddings.create(model=model, input=batch)
                break
            except Exception as exc:
                if (is_rotatable_error(exc) or is_auth_error(exc)) and attempt < max_attempts - 1:
                    rotate_or_remove(rotator, exc, context="embed")
                    continue
                raise

        if resp is None:
            raise RuntimeError(f"Failed to embed batch starting at index {i}")

        data = (
            sorted(resp.data, key=lambda d: d.index)
            if all(getattr(d, "index", None) is not None for d in resp.data)
            else resp.data
        )
        vectors.extend(item.embedding for item in data)

    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms
