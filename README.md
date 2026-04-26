# Voicebot Dashboard (Dockerized)

This project is fully containerized with a multi-service Docker Compose setup and now launches a real FastAPI-powered dashboard web app by default:

- `llama-cpp`: local `llama.cpp` server (GGUF-backed)
- `voicebot`: web app + voicebot runtime

## Quick Start

1. Build and run everything:

```bash
docker compose up --build
```

2. Open `http://localhost:7860` in your browser.

3. Stop:

```bash
docker compose down
```

## Windows One-Step Launch

On Windows, you can start Docker and automatically open the dashboard in your local browser with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-dashboard.ps1
```

If you also want logs in the terminal:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-dashboard.ps1 -Logs
```

That launcher also starts a native host-side audio bridge so the bot speaks automatically through your local speakers without relying on browser playback controls.

You can also use the Python launcher to start Docker + host audio bridge together:

```bash
python run_voicebot.py
```

By default this launcher also starts host push-to-talk STT (faster-whisper `base.en` + Silero VAD).
Hold `F8` to speak and release to send your utterance as a chat query.

Useful flags:

```bash
python run_voicebot.py --stt-hotkey f9
python run_voicebot.py --no-stt
python run_voicebot.py --down-on-exit
```

If you start Docker manually, run the bridge separately:

```powershell
.\venv\Scripts\python.exe .\scripts\voicebot_audio_bridge.py
```

## Configuration

You can override these environment variables when running Compose:

- `LLAMA_CPP_MODEL_FILE` (default: `gemma-3-1b-it-Q8_0.gguf`, loaded from `./model`)
- `LLAMA_CPP_MODEL` (default: `gemma3-1b-it-Q8`)
- `LLAMA_CPP_DISABLE_THINK` (default: `1`, keep thinking mode off)
- `VOICEBOT_AUDIO_MODE` (`silent` or `live`, default in compose: `silent`)

Examples:

```bash
LLAMA_CPP_MODEL_FILE=gemma-3-1b-it-Q8_0.gguf docker compose up --build
```

```bash
VOICEBOT_AUDIO_MODE=live docker compose up --build
```

## Notes

- Ensure the model file exists at `model/gemma-3-1b-it-Q8_0.gguf` (or set `LLAMA_CPP_MODEL_FILE`).
- Rebuild the image after dependency changes with `docker compose build --no-cache`.
- Hugging Face/Kokoro cache is persisted in the `huggingface-cache` volume.
- `silent` audio mode is the most reliable default for containerized environments. Use `live` only if your Docker host is configured for audio passthrough.
- The web app shows live LLM text, worker status, TTS chunk flow, runtime timeline, and turn latency metrics from the real runtime.
