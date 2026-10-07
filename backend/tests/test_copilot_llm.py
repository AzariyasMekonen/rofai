import asyncio

import pytest

from app.config import Settings
from app.copilot_llm import CopilotChat, CopilotChatError, to_prompt
from copilot.session_events import (
    AssistantMessageData,
    AssistantMessageDeltaData,
    SessionErrorData,
    SessionIdleData,
)


def test_to_prompt_single_turn():
    system, prompt = to_prompt(
        [
            {"role": "system", "content": "persona"},
            {"role": "system", "content": "archive"},
            {"role": "user", "content": "When did it open?"},
        ]
    )
    assert system == "persona\n\narchive"
    assert prompt == "When did it open?"


def test_to_prompt_includes_history():
    _, prompt = to_prompt(
        [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello child"},
            {"role": "user", "content": "Who built the library?"},
        ]
    )
    assert "Visitor: Hi\nYou: Hello child" in prompt
    assert prompt.endswith("Visitor's new question: Who built the library?")


class _FakeSession:
    def __init__(self, emit, script):
        self._emit = emit
        self._script = script
        self.aborted = False
        self.disconnected = False
        self.prompt = None

    async def send(self, prompt):
        self.prompt = prompt
        for data in self._script:
            self._emit(type("Event", (), {"data": data})())
        return "msg-1"

    async def abort(self):
        self.aborted = True

    async def disconnect(self):
        self.disconnected = True


class _FakeClient:
    def __init__(self, script):
        self.script = script
        self.kwargs = None
        self.session = None
        self.deleted = []

    async def create_session(self, **kwargs):
        self.kwargs = kwargs
        self.session = _FakeSession(kwargs["on_event"], self.script)
        return self.session

    async def delete_session(self, session_id):
        self.deleted.append(session_id)


async def _run(script):
    chat = CopilotChat(Settings(copilot_github_token="github_pat_test"))
    fake = _FakeClient(script)
    chat._client = fake
    parts = [p async for p in chat.stream([{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}])]
    return parts, fake


def test_stream_yields_deltas_and_cleans_up():
    parts, fake = asyncio.run(
        _run(
            [
                AssistantMessageDeltaData(delta_content="Hello ", message_id="m"),
                AssistantMessageDeltaData(delta_content="child.", message_id="m"),
                AssistantMessageData(content="Hello child.", message_id="m"),
                SessionIdleData(),
            ]
        )
    )
    assert parts == ["Hello ", "child."]
    # Locked down: no tools, persona replaces the agent prompt
    assert fake.kwargs["available_tools"] == []
    assert fake.kwargs["system_message"] == {"mode": "replace", "content": "sys"}
    assert fake.session.prompt == "q"
    assert fake.session.disconnected and not fake.session.aborted
    assert fake.deleted == [fake.kwargs["session_id"]]


def test_stream_falls_back_to_final_message_when_not_streamed():
    parts, _ = asyncio.run(_run([AssistantMessageData(content="Whole answer.", message_id="m"), SessionIdleData()]))
    assert parts == ["Whole answer."]


def test_stream_raises_on_session_error_and_still_cleans_up():
    with pytest.raises(CopilotChatError):
        asyncio.run(_run([SessionErrorData(error_type="quota", message="limit reached")]))


def test_session_error_aborts_and_deletes_session():
    async def go():
        chat = CopilotChat(Settings(copilot_github_token="t"))
        fake = _FakeClient([SessionErrorData(error_type="quota", message="limit")])
        chat._client = fake
        with pytest.raises(CopilotChatError):
            async for _ in chat.stream([{"role": "user", "content": "q"}]):
                pass
        return fake

    fake = asyncio.run(go())
    assert fake.session.aborted and fake.session.disconnected
    assert fake.deleted == [fake.kwargs["session_id"]]
