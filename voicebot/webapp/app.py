from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from voicebot.config import VoicebotSettings
from voicebot.webapp.service import DashboardService


STATIC_DIR = Path(__file__).resolve().parent / "static"
logger = logging.getLogger("voicebot.webapp")


class ChatRequest(BaseModel):
    text: str


class MemoryTabRequest(BaseModel):
    tab: str


def create_app(settings: VoicebotSettings | None = None) -> FastAPI:
    app = FastAPI(title="raven voice dashboard", version="1.0.0")
    resolved_settings = settings or VoicebotSettings.from_env()
    service = DashboardService(resolved_settings)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.on_event("startup")
    async def _startup() -> None:
        host = os.getenv("VOICEBOT_DASHBOARD_HOST", "127.0.0.1")
        port = os.getenv("VOICEBOT_DASHBOARD_PORT", "7860")
        dashboard_url = f"http://localhost:{port}/"
        api_url = f"http://localhost:{port}/api/health"
        logger.info("Dashboard startup complete")
        logger.info("Dashboard URL: %s (bind host: %s)", dashboard_url, host)
        logger.info("Backend health URL: %s", api_url)
        if resolved_settings.warmup_enabled:
            asyncio.create_task(service.warmup(), name="voicebot_webapp_warmup")

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await service.close()

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/state")
    async def state() -> dict:
        return service.snapshot()

    @app.get("/api/audio/latest")
    async def latest_audio() -> Response:
        audio = service.get_latest_audio()
        if audio is None:
            raise HTTPException(status_code=404, detail="No synthesized audio is available yet.")
        return Response(content=audio, media_type="audio/wav")

    @app.post("/api/chat")
    async def chat(payload: ChatRequest) -> dict:
        try:
            return await service.start_turn(payload.text)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/memory/tab")
    async def set_memory_tab(payload: MemoryTabRequest) -> dict:
        try:
            active_tab = service.set_active_memory_tab(payload.tab)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        snapshot = service.snapshot()
        return {
            "active_memory_tab": active_tab,
            "notes": snapshot["notes"],
            "reminders": snapshot["reminders"],
        }

    return app
