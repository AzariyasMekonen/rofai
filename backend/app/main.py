import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import speech
from .config import ROOT_DIR, get_settings
from .llm import NotConfiguredError, complete, copilot_chat, embed, ensure_llm_configured, stream_chat
from .rag import Language, build_messages
from .ratelimit import RateLimiter
from .store import Hit, VectorStore

log = logging.getLogger("rofai")

MAX_MESSAGE_CHARS = 500
MAX_HISTORY_TURNS = 8
MAX_TTS_CHARS = 600
MAX_AUDIO_SECONDS = 30
MAX_AUDIO_BYTES = 16000 * 2 * MAX_AUDIO_SECONDS + 1024

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = VectorStore.load(settings.index_dir)
    if store is None:
        log.warning("No index at %s; run `python -m app.ingest`", settings.index_dir)
    elif store.embed_model and store.embed_model != settings.embed_model:
        log.warning("Index built with %s but EMBED_MODEL is %s; re-run ingest", store.embed_model, settings.embed_model)
    app.state.store = store
    yield
    if settings.llm_provider == "copilot":
        await copilot_chat().close()


app = FastAPI(title="Tsehay Roschli anniversary guide", lifespan=lifespan, docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

chat_limit = RateLimiter(settings.rate_limit_chat_per_minute)
speech_limit = RateLimiter(settings.rate_limit_speech_per_minute)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=2000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    history: list[ChatTurn] = Field(default_factory=list, max_length=MAX_HISTORY_TURNS * 2)
    language: Language = "auto"


class TtsRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TTS_CHARS)
    language: Literal["auto", "en", "am", "om"] = "auto"


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/api/health")
async def health(request: Request) -> dict:
    store: VectorStore | None = request.app.state.store
    return {
        "ok": True,
        "llm": settings.llm_configured,
        "llmProvider": settings.llm_provider,
        "speech": settings.speech_configured,
        "indexChunks": len(store) if store else 0,
        "search": "vector" if _use_vectors(store) else "keyword",
        "persona": settings.persona_name,
        "school": settings.school_name,
    }


def _use_vectors(store: VectorStore | None) -> bool:
    return bool(store and store.has_vectors and settings.embeddings_configured)


_SEARCH_KEYWORDS_PROMPT = (
    "Translate the user's question into English search keywords for a school archive "
    "(names, places, years, topics). Output only the keywords on one line."
)


async def _retrieve(store: VectorStore | None, question: str) -> list[Hit]:
    if not store:
        return []
    if _use_vectors(store):
        query_vec = (await asyncio.to_thread(embed, [question]))[0]
        return store.search(query_vec, settings.top_k)
    query = question
    if speech.detect_text_language(question) == "am":
        # The archive is mostly English, so keyword search needs English terms
        try:
            keywords = await complete(
                [{"role": "system", "content": _SEARCH_KEYWORDS_PROMPT}, {"role": "user", "content": question}]
            )
            query = f"{question} {keywords}"
        except NotConfiguredError:
            raise
        except Exception:
            log.warning("keyword translation failed; searching with the original question", exc_info=True)
    return store.search_text(query, settings.top_k)


@app.post("/api/chat", dependencies=[Depends(chat_limit)])
async def chat(req: ChatRequest, request: Request) -> StreamingResponse:
    try:
        ensure_llm_configured()
        hits = await _retrieve(request.app.state.store, req.message.strip())
    except NotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("retrieval failed")
        raise HTTPException(status_code=502, detail="Could not search the archive.") from exc

    history = [t.model_dump() for t in req.history[-MAX_HISTORY_TURNS * 2 :]]
    messages = build_messages(settings, req.message.strip(), history, hits, req.language)

    async def stream() -> AsyncIterator[str]:
        yield _sse("sources", [{"source": h.chunk.source, "page": h.chunk.page, "score": round(h.score, 3)} for h in hits])
        try:
            async for text in stream_chat(messages):
                yield _sse("delta", {"text": text})
            yield _sse("done", {})
        except Exception:
            log.exception("chat completion failed")
            yield _sse("error", {"message": "I'm sorry, I couldn't answer just now. Please try again."})

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.post("/api/stt", dependencies=[Depends(speech_limit)])
async def stt(audio: UploadFile = File(...), language: Literal["auto", "en", "am"] = Form("auto")) -> dict:
    data = await audio.read(MAX_AUDIO_BYTES + 1)
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail=f"Audio longer than {MAX_AUDIO_SECONDS} seconds.")
    try:
        pcm = speech.read_wav_pcm(data, MAX_AUDIO_SECONDS)
        text, detected = await asyncio.to_thread(speech.recognize, pcm, language)
    except speech.InvalidAudioError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except speech.SpeechNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except speech.SpeechServiceError as exc:
        log.error("%s", exc)
        raise HTTPException(status_code=502, detail="Speech recognition failed.") from exc
    return {"text": text, "language": detected}


@app.post("/api/tts", dependencies=[Depends(speech_limit)])
async def tts(req: TtsRequest) -> dict:
    language = speech.detect_text_language(req.text) if req.language == "auto" else req.language
    if language == "om":
        # Azure Speech has no Afaan Oromo voice; the client shows text only
        return {"audio": None, "mime": None, "visemes": [], "language": "om"}
    try:
        audio, visemes = await asyncio.to_thread(speech.synthesize, req.text, language)
    except speech.SpeechNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except speech.SpeechServiceError as exc:
        log.error("%s", exc)
        raise HTTPException(status_code=502, detail="Speech synthesis failed.") from exc
    return {
        "audio": base64.b64encode(audio).decode("ascii"),
        "mime": "audio/mpeg",
        "visemes": visemes,
        "language": language,
    }


_dist = ROOT_DIR / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")
