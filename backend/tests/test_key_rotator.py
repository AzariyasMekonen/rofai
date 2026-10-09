from unittest.mock import AsyncMock, MagicMock
import httpx
import openai
import pytest

from app.config import Settings
from app.llm import KeyRotator, is_rotatable_error, stream_chat


def test_key_rotator_cycles_keys():
    rotator = KeyRotator("https://example.com/v1", ["key1", "key2", "key3"])
    assert rotator.current_key == "key1"
    
    assert rotator.rotate_key() == "key2"
    assert rotator.current_key == "key2"
    
    assert rotator.rotate_key() == "key3"
    assert rotator.current_key == "key3"
    
    assert rotator.rotate_key() == "key1"
    assert rotator.current_key == "key1"


def test_key_rotator_handles_empty_keys():
    rotator = KeyRotator("https://example.com/v1", [])
    assert rotator.current_key == "unused"
    assert rotator.rotate_key() == "unused"


def test_is_rotatable_error():
    request = httpx.Request("POST", "https://example.com")
    response_429 = httpx.Response(429, request=request)
    response_503 = httpx.Response(503, request=request)
    response_400 = httpx.Response(400, request=request)

    response_401 = httpx.Response(401, request=request)
    response_403 = httpx.Response(403, request=request)

    rate_limit_err = openai.RateLimitError("Rate limit exceeded", response=response_429, body=None)
    status_503_err = openai.APIStatusError("Service unavailable", response=response_503, body=None)
    auth_err = openai.AuthenticationError("Invalid key", response=response_401, body=None)
    perm_err = openai.PermissionDeniedError("Permission denied", response=response_403, body=None)
    status_400_err = openai.APIStatusError("Bad request", response=response_400, body=None)

    assert is_rotatable_error(rate_limit_err) is True
    assert is_rotatable_error(status_503_err) is True
    assert is_rotatable_error(auth_err) is True
    assert is_rotatable_error(perm_err) is True
    assert is_rotatable_error(status_400_err) is False


def test_settings_loads_api_keys(tmp_path):
    keys_file = tmp_path / "keys.txt"
    keys_file.write_text("keyA\n# comment\nkeyB\n\nkeyC\n")

    settings = Settings(llm_keys_file=str(keys_file))
    assert settings.api_keys == ["keyA", "keyB", "keyC"]


@pytest.mark.anyio
async def test_stream_chat_rotates_on_rate_limit(monkeypatch):
    rotator = KeyRotator("https://example.com/v1", ["fail_key", "good_key"])
    monkeypatch.setattr("app.llm.key_rotator", lambda: rotator)
    monkeypatch.setattr("app.llm.ensure_llm_configured", lambda: None)
    
    request = httpx.Request("POST", "https://example.com")
    resp_429 = httpx.Response(429, request=request)

    async def mock_generator():
        chunk = MagicMock()
        chunk.choices = [MagicMock(delta=MagicMock(content="Success!"))]
        yield chunk

    def mock_get_client():
        mock_client = MagicMock()
        if rotator.current_key == "fail_key":
            mock_client.chat.completions.create = AsyncMock(
                side_effect=openai.RateLimitError("Resource exhausted", response=resp_429, body=None)
            )
        else:
            mock_client.chat.completions.create = AsyncMock(return_value=mock_generator())
        return mock_client

    monkeypatch.setattr(rotator, "get_client", mock_get_client)

    collected = []
    async for part in stream_chat([{"role": "user", "content": "Hello"}]):
        collected.append(part)

    assert collected == ["Success!"]
    assert rotator.current_key == "good_key"
