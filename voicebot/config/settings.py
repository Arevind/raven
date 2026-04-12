from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class VoicebotSettings:
    ollama_url: str = "http://127.0.0.1:11434/api/chat"
    ollama_model: str = "gemma3:1b"
    kokoro_voice: str = "af_heart"
    kokoro_lang: str = "a"
    kokoro_repo_id: str = "hexgrad/Kokoro-82M"
    sample_rate: int = 24000
    tts_max_chars: int = 280
    tts_min_chars: int = 80
    tts_soft_chars: int = 0
    http_timeout_seconds: float = 300.0
    audio_queue_maxsize: int = 32
    warmup_enabled: bool = True

    @classmethod
    def from_env(cls) -> "VoicebotSettings":
        def _int(name: str, default: int) -> int:
            raw = os.getenv(name)
            return int(raw) if raw else default

        def _float(name: str, default: float) -> float:
            raw = os.getenv(name)
            return float(raw) if raw else default

        default = cls()
        return cls(
            ollama_url=os.getenv("OLLAMA_URL", default.ollama_url),
            ollama_model=os.getenv("OLLAMA_MODEL", default.ollama_model),
            kokoro_voice=os.getenv("KOKORO_VOICE", default.kokoro_voice),
            kokoro_lang=os.getenv("KOKORO_LANG", default.kokoro_lang),
            kokoro_repo_id=os.getenv("KOKORO_REPO_ID", default.kokoro_repo_id),
            sample_rate=_int("SAMPLE_RATE", default.sample_rate),
            tts_max_chars=_int("TTS_MAX_CHARS", default.tts_max_chars),
            tts_min_chars=_int("TTS_MIN_CHARS", default.tts_min_chars),
            tts_soft_chars=_int("TTS_SOFT_CHARS", default.tts_soft_chars),
            http_timeout_seconds=_float("HTTP_TIMEOUT_SECONDS", default.http_timeout_seconds),
            audio_queue_maxsize=_int("AUDIO_QUEUE_MAXSIZE", default.audio_queue_maxsize),
            warmup_enabled=os.getenv("VOICEBOT_WARMUP", "1") != "0",
        )

