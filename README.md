# J.A.R.V.I.S — Personal Voice Assistant

A self-hosted, Iron-Butler personal assistant in the tradition of Tony Stark's JARVIS.
It runs **entirely on your own machine**: a local LLM (via Ollama) does the thinking,
local services carry out the work, and a FastAPI + vanilla-JS web front end is the
interface. No cloud LLM API keys are required to chat.

**Personality:** composed, dry, quietly brilliant. It addresses you as *"sir"*, makes
one sharp quip per exchange, and stays unflappable — in **every** reply path (chat,
vision, task execution).

**Hands-free:** say *"wake up Jarvis"* / *"hey Jarvis"* or **clap twice** to start
listening. Say *"shut down"* to power the mic off; a spoken wake word brings it back.

---

## Features

- 🎙️ **Voice in / voice out** — speech-to-text and text-to-speech (edge-tts), gapless streaming
- 🧠 **Local LLM brain** — Ollama (`nemotron-3.5-lightning` by default), no cloud keys
- 👁️ **Vision** — point the camera, ask what it sees (uses `llava`)
- 🔎 **Web search** — Tavily (optional; everything else works without it)
- ▶️ **Tasks** — open websites/apps, play music/videos, generate images, write content
- 📁 **Local file & folder tools** — create / list / delete on your PC (sandboxed, no shell)
- 🖐️ **Hands-free wake** — wake word or double-clap; **clap is latched off after the first "shut down"** for the rest of the session
- 🎭 **Consistent persona** — the JARVIS register is baked into both the system prompt *and* the canned task replies

---

## Architecture

```
browser (frontend/)                FastAPI (app/main.py)
  speech-to-text   ───────────►   /chat/stream, /realtime, /tts ...
  MediaSource audio ◄───────────    edge-tts (streamed per sentence)
                                     │
                              chat_service / realtime_service
                                     │
                              brain_service (intent router)
                                     │
                    ┌────────────────┼──────────────────┐
             llm_service        task_executor        vision_service
           (Ollama brain)      (open/play/files…)     (llava)
```

### Repository layout

| Path | Purpose |
|------|---------|
| `run.py` | Entry point — validates startup, generates the self-signed TLS cert, auto-generates thinking-audio, then starts uvicorn (HTTPS on by default) |
| `app/main.py` | FastAPI app + all HTTP endpoints + TTS streaming |
| `app/services/llm_service.py` | Local Ollama chat (streaming + non-streaming) |
| `app/services/brain_service.py` | Intent routing / decision layer |
| `app/services/task_executor.py` | Carries out tasks; holds the JARVIS-voice canned one-liners |
| `app/services/local_tools.py` | Sandboxed local file/folder tools (whitelist, no shell) |
| `app/services/vision_service.py` | Camera-image analysis via a vision model |
| `app/services/realtime_service.py` | Live web-search chat |
| `app/services/vector_store.py` | FAISS + sentence-transformer memory over `database/learning_data` |
| `app/generate_thinking_audio.py` | Generates the "thinking" UI sounds (`frontend/audio/starter_*.mp3`) |
| `download_model.py` | Downloads the local embedding model into `local_models/` |
| `config.py` | All configuration (reads `.env`) |
| `frontend/` | The web UI (vanilla JS + CSS) |
| `requirements.txt` | Python dependencies |

---

## Prerequisites

1. **Python 3.10+**
2. **Ollama** — https://ollama.com  (the local LLM runtime)
3. **ffmpeg** (recommended) — used by edge-tts for audio
4. A **web browser** with microphone access (Chrome is best for speech-to-text)
5. Internet access (for edge-tts voice, optional web search, and one-time model downloads)

> The microphone only works in a **secure context**. That means `https://` — which is why
> this project serves over HTTPS by default and auto-generates a self-signed certificate.
> Plain `http://<lan-ip>` will have its mic blocked by the browser.

---

## Setup (step by step)

### 1. Get the code

```bash
git clone https://github.com/gaurav24072002/JARVIS_2.0.git JARVIS
cd JARVIS
```

### 2. Create a virtual environment and install Python dependencies

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> `requirements.txt` includes `torch` and `transformers`, so the first install is a
> large download. This is one-time.

### 3. Install and start Ollama, then pull the models

```bash
# start the Ollama server (already running? skip)
ollama serve          # usually runs in the background after install
```

Pull the chat model (default) and a vision model:

```bash
ollama pull nemotron-3.5-lightning
ollama pull llava     # needed only for camera/vision features
```

Confirm:

```bash
ollama list
```

### 4. Download the local embedding model (one-time)

The project uses a **local** sentence-transformer for memory (not downloaded from the
cloud at runtime):

```bash
python download_model.py
```

This saves the model to `./local_models/all-MiniLM-L6-v2`.
(That directory is intentionally git-ignored — it is **generated by your code**, not shipped.)

### 5. Configure the environment

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

Open `.env` and, at minimum, confirm the Ollama model names match step 3.
Add a `TAVILY_API_KEY` **only** if you want live web search — everything else works without it.

Key settings (see [Configuration](#configuration-reference) below):

```
OLLAMA_MODEL=nemotron-3.5-lightning:latest
OLLAMA_VISION_MODEL=llava
TTS_VOICE=en-GB-RyanNeural
ASSISTANT_NAME=JARVIS
JARVIS_USER_TITLE=Sir
```

### 6. Run it

```bash
python run.py
```

On first start you'll see:

```
[OK] Ollama reachable at http://localhost:11434
[OK] Model 'nemotron-3.5-lightning:latest' is available locally
[TLS] Serving over HTTPS (https://<host>:8000)
```

`run.py` also:
- generates a self-signed TLS cert into `./certs` (one-time), and
- generates the UI "thinking" sounds into `frontend/audio/` (one-time).

### 7. Open the app and allow the microphone

Open **https://localhost:8000** (note the `https`).

1. The browser shows a **self-signed certificate warning** — click **Advanced → Proceed / Continue** **once**. Accepting it is what unlocks the mic.
2. Click the mic and **allow** microphone access when prompted.
3. Talk to JARVIS.

> To use it from **another device on your network**, find this machine's LAN IP
> (e.g. `192.168.x.x`) and open `https://<that-ip>:8000` in a browser on the other
> device, then accept the certificate warning there too.

---

## Using it

| Action | How |
|--------|-----|
| Talk to it | Click the mic, or use the text box |
| **Hands-free start** | Say **"wake up Jarvis"** / **"hey Jarvis"**, or **clap twice** |
| **Power down** | Say **"shut down"** (or *"power off"*, *"goodbye"*) — mic stops |
| **Power back up** | Say **"wake up Jarvis"** — claps stay off for the rest of the session after the first shutdown |
| Camera | Click the camera icon, send a message, or ask *"what can you see?"* |
| Play / open / search | *"play some music"*, *"open YouTube"*, *"search for …"* |
| Files | *"create a folder called reports"*, *"list my downloads"* (sandboxed to your home) |
| Stop the server | `Ctrl+C` in the terminal |

**Settings panel** (gear icon) toggles: auto-open activity/search panels, thinking
sounds, voice interruption, and hands-free wake.

---

## Configuration reference

All settings live in `.env` (or the process environment). Defaults shown in `config.py`.

| Variable | Default | Meaning |
|----------|---------|---------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_MODEL` | `nemotron-3.5-lightning:latest` | Chat / brain model |
| `OLLAMA_BRAIN_MODEL` | same as above | Decision-routing model |
| `OLLAMA_VISION_MODEL` | `llava` | Vision (camera) model |
| `OLLAMA_REQUEST_TIMEOUT` | `120` | LLM request timeout (seconds) |
| `TAVILY_API_KEY` | *(empty)* | Web search key; empty = search disabled |
| `JARVIS_HTTPS` | `1` | Serve over HTTPS (`0` = plain HTTP, mic won't work over LAN IP) |
| `TTS_VOICE` | `en-GB-RyanNeural` | edge-tts voice (try `en-GB-ThomasNeural`) |
| `TTS_RATE` | `+8%` | Speech speed |
| `TTS_PITCH` | `-8Hz` | Pitch offset (lower = deeper) |
| `ASSISTANT_NAME` | `Jarvis` | Name the assistant uses for itself |
| `JARVIS_USER_TITLE` | *(empty)* | How to address you (e.g. `Sir`) |
| `JARVIS_OWNER_NAME` | *(empty)* | Owner name referenced in the persona |
| `TASK_EXECUTION_TIMEOUT` | `30` | Per-task timeout (seconds) |
| `VISION_MAX_IMAGE_BYTES` | `5000000` | Max image size accepted for vision |

> **Note on the local model path:** the embedding model is pinned to
> `./local_models/all-MiniLM-L6-v2` (see `config.py` → `EMBEDDING_MODEL`). Run
> `python download_model.py` once to populate it.

---

## Local file & folder tools

`app/services/local_tools.py` exposes **create / list / delete** for files and folders.
It is deliberately safe:

- **Fixed whitelist** of operations — it never runs a shell.
- **Confinement** to allowed base directories (your home folder).
- **Path validation** that rejects metacharacters and traversal.

You'll get a clear *"I can only work inside: …"* message if you ask it to touch a path
outside the allowed area.

---

## TTS / voice

Text-to-speech uses [edge-tts](https://github.com/rany2/edge-tts). Sentences are
generated on the server as they stream and pushed to the browser, where a
**MediaSource** pipeline plays them **gaplessly** (no per-sentence reload lag). Browsers
without MediaSource automatically fall back to per-chunk playback.

To change the voice, set `TTS_VOICE` in `.env` (e.g. `en-GB-ThomasNeural`) and restart.
Sample clips for comparison live in `frontend/audio/voice_samples/` (generated, git-ignored).

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| **"Ollama is not reachable"** at startup | Start it: `ollama serve`, or fix `OLLAMA_BASE_URL`. |
| **Mic does nothing** | Use `https://`, accept the self-signed cert once, and grant mic permission. Chrome is best. |
| **Mic works on laptop but not from phone/other PC** | Open `https://<lan-ip>:8000` there and accept the cert on that device. |
| **Model not found** | `ollama pull <OLLAMA_MODEL>` (and `ollama pull llava` for vision). |
| **Embedding / vector errors** | Run `python download_model.py` once. |
| **No web search results** | Add a `TAVILY_API_KEY` in `.env`. Other features still work. |
| **Port 8000 in use** | Stop the other process, or change the port in `run.py`. |
| **Slow / first run long** | The Ollama model loads into memory on first use; it's warm afterwards. |
| **Want a deeper / calmer voice** | Lower `TTS_PITCH` (e.g. `-12Hz`), adjust `TTS_RATE`. |

---

## Security notes

- `.env` (which may contain your `TAVILY_API_KEY`) is **git-ignored** — it is never committed.
- The LLM and embedding model run locally; chat content is not sent to a cloud LLM.
- Web search (if enabled) sends your queries to the Tavily service.
- The self-signed certificate is for local/LAN use; for public exposure put a real
  reverse proxy / TLS in front.

---

## What is (deliberately) *not* in this repository

These are **generated by the project's own code** or are local artifacts, so they are
git-ignored rather than shipped:

- `venv/`, `__pycache__/` — Python environment
- `local_models/` — downloaded embedding model (run `download_model.py`)
- `frontend/audio/` — thinking-audio + voice samples (generated by `app/generate_thinking_audio.py`)
- `database/vector_store/`, `database/chats_data/`, `database/camera_captures/` — runtime data
- `certs/` — self-signed TLS (auto-generated on first run)
- `.env` — your secrets

---

## Project

Built as a personal assistant. Local-first, self-hosted, and opinionated about wit.
