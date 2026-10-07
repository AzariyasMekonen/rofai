# 40th Anniversary Voice Guide: Tsehay Roschli

A talking 3D avatar of the school's founder, Tsehay Roschli, for the 40th anniversary.
Visitors ask questions by voice or text in English or Amharic (Afaan Oromo is text only).
She answers in the first person, using only the school's own archive.

```
browser (three.js + VRM avatar)  ──►  FastAPI backend  ──►  GitHub Copilot SDK (or any OpenAI-compatible LLM)
        mic / speakers                 │                   Azure AI Speech (STT, TTS, visemes)
                                       └── data/index (archive search)
```

## Requirements

- [uv](https://docs.astral.sh/uv/) (it installs the Python version pinned in `backend/pyproject.toml`), Node.js 20+
- A language model, one of:
  - **GitHub Copilot** (`LLM_PROVIDER=copilot`): a fine-grained personal access token (`github_pat_…`)
    from an account with a Copilot plan. Every visitor question counts against that account's Copilot usage.
    The runtime is started in locked-down `empty` mode with no tools, and each visitor session is deleted after the answer.
  - **OpenAI-compatible** (`LLM_PROVIDER=openai`): Azure OpenAI / Foundry, OpenAI, or Ollama.
- An Azure AI Speech resource (the free F0 tier is enough for testing)

Copilot has no embeddings API, so without an embeddings endpoint the archive uses keyword (BM25) search.
Amharic questions are first turned into English keywords by the model, because the archive is mostly English.

## Setup

```bash
cp .env.example .env          # then fill in the keys
cd backend && uv sync
uv run python -m copilot download-runtime   # only for LLM_PROVIDER=copilot
cd ../frontend && npm install
```

### 1. Add the archive

- Put the OCR'd PDFs (and any `.md` / `.txt` files) in `data/sources/`.
- Optionally write `data/facts.md`: a short, checked fact sheet (founding date, principals,
  milestones, her biography). It ranks above the scans, which fixes OCR mistakes on key facts.

```bash
cd backend
uv run python -m app.ingest --dry-run   # check the OCR text looks right, no API calls
uv run python -m app.ingest             # build data/index (embeddings if configured, else keyword index)
uv run python -m app.query "When was the school founded?"
```

Re-run `app.ingest` and restart the backend whenever the documents change.

### 2. Add the avatar

Export her model from VRoid Studio / Blender as `frontend/public/models/tsehay.vrm`.
For lip-sync the model needs the standard VRM expressions `aa`, `ih`, `ou`, `ee`, `oh`, `blink`, `happy`.
Until the file exists a simple placeholder bust is shown.

### 3. Run (development)

```bash
cd backend && uv run uvicorn app.main:app --reload --port 8000
cd frontend && npm run dev     # open http://localhost:5173
```

Kiosk mode: `http://localhost:5173/?mode=kiosk` hides typing, goes fullscreen on first touch,
and clears the conversation after 90 seconds of inactivity. For a booth, run Chrome with
`--kiosk --autoplay-policy=no-user-gesture-required`.

### 4. Production

```bash
cd frontend && npm run build    # backend serves frontend/dist automatically
cd backend && uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --proxy-headers
```

Put it behind HTTPS, since browsers only allow microphone access on HTTPS or localhost.
Set `ALLOWED_ORIGINS` to the public URL. Rate limits are per IP and in-memory, so run a single worker.

## Tests

```bash
cd backend && uv run pytest
```

## Known limitations

- Azure does not provide visemes for Amharic voices, so Amharic lip-sync follows the audio loudness.
  English uses precise visemes.
- Azure Speech has no Afaan Oromo speech recognition or voice. Oromo questions can be typed and are answered as text.
- Answers are only as good as the OCR; spot-check `--dry-run` output and use `data/facts.md` for key facts.
