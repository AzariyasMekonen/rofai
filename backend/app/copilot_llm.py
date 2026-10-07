"""Chat through the GitHub Copilot SDK, locked down for anonymous public visitors.

The runtime is started in "empty" mode with no tools, so a visitor's prompt cannot make
the agent read files or run commands. Each request gets a fresh session that is deleted
afterwards, so questions are not kept on disk.
"""

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator

from copilot import CopilotClient
from copilot.session_events import (
    AssistantMessageData,
    AssistantMessageDeltaData,
    SessionErrorData,
    SessionIdleData,
)

from .config import Settings

log = logging.getLogger("rofai.copilot")

RESPONSE_TIMEOUT_SECONDS = 60


class CopilotChatError(RuntimeError):
    pass


def to_prompt(messages: list[dict[str, str]]) -> tuple[str, str]:
    """Split OpenAI-style messages into (system message, single prompt with the transcript)."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    convo = [m for m in messages if m["role"] != "system"]
    if not convo:
        raise ValueError("no user message")
    if len(convo) == 1:
        return system, convo[0]["content"]
    transcript = "\n".join(
        f"{'Visitor' if m['role'] == 'user' else 'You'}: {m['content']}" for m in convo[:-1]
    )
    return system, f"Conversation so far:\n{transcript}\n\nVisitor's new question: {convo[-1]['content']}"


class CopilotChat:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client: CopilotClient | None = None
        self._lock = asyncio.Lock()

    async def _get_client(self) -> CopilotClient:
        async with self._lock:
            if self._client is None:
                s = self._settings
                s.copilot_home.mkdir(parents=True, exist_ok=True)
                client = CopilotClient(
                    mode="empty",
                    github_token=s.copilot_github_token,
                    use_logged_in_user=False,
                    base_directory=str(s.copilot_home),
                    session_idle_timeout_seconds=300,
                    log_level="warning",
                )
                await client.start()
                self._client = client
            return self._client

    async def close(self) -> None:
        async with self._lock:
            if self._client is not None:
                await self._client.stop()
                self._client = None

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        system, prompt = to_prompt(messages)
        client = await self._get_client()
        loop = asyncio.get_running_loop()
        events: asyncio.Queue = asyncio.Queue()
        session_id = f"visitor-{uuid.uuid4()}"

        session = await client.create_session(
            session_id=session_id,
            model=self._settings.copilot_model or None,
            system_message={"mode": "replace", "content": system},
            available_tools=[],
            streaming=True,
            infinite_sessions={"enabled": False},
            on_event=lambda event: loop.call_soon_threadsafe(events.put_nowait, event.data),
        )
        finished = False
        streamed = False
        try:
            await session.send(prompt)
            while True:
                data = await asyncio.wait_for(events.get(), RESPONSE_TIMEOUT_SECONDS)
                match data:
                    case AssistantMessageDeltaData() if data.parent_tool_call_id is None:
                        if data.delta_content:
                            streamed = True
                            yield data.delta_content
                    case AssistantMessageData() if data.parent_tool_call_id is None:
                        # Only used when the model did not stream
                        if not streamed and data.content:
                            yield data.content
                    case SessionErrorData():
                        hint = ""
                        if data.error_type == "authorization":
                            hint = (
                                " -> COPILOT_GITHUB_TOKEN must be a fine-grained PAT owned by your personal"
                                " account with the 'Copilot Requests' account permission, on an account with Copilot."
                            )
                        raise CopilotChatError(f"{data.error_type}: {data.message}{hint}")
                    case SessionIdleData():
                        finished = True
                        break
        finally:
            await self._cleanup(client, session, session_id, abort=not finished)

    async def _cleanup(self, client: CopilotClient, session, session_id: str, abort: bool) -> None:
        try:
            if abort:
                await session.abort()
            await session.disconnect()
            await client.delete_session(session_id)
        except Exception:
            log.warning("could not clean up Copilot session %s", session_id, exc_info=True)
