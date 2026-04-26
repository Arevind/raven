from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class VoicebotSettings:
    llama_cpp_url: str = "http://127.0.0.1:8080/v1/chat/completions"
    llama_cpp_model: str = "gemma3n-e2b"
    llama_cpp_disable_think: bool = True
    kokoro_voice: str = "af_heart"
    kokoro_lang: str = "a"
    kokoro_repo_id: str = "hexgrad/Kokoro-82M"
    sample_rate: int = 24000
    tts_max_chars: int = 280
    tts_min_chars: int = 80
    tts_soft_chars: int = 0
    http_timeout_seconds: float = 300.0
    audio_queue_maxsize: int = 32
    audio_mode: str = "live"
    warmup_enabled: bool = True

    @classmethod
    def from_env(cls) -> "VoicebotSettings":
        def _int(name: str, default: int) -> int:
            raw = os.getenv(name)
            return int(raw) if raw else default

        def _float(name: str, default: float) -> float:
            raw = os.getenv(name)
            return float(raw) if raw else default

        def _bool(name: str, default: bool) -> bool:
            raw = os.getenv(name)
            if raw is None:
                return default
            return raw.strip().lower() not in {"0", "false", "no", "off"}

        default = cls()
        audio_mode = os.getenv("VOICEBOT_AUDIO_MODE", default.audio_mode).strip().lower()
        if audio_mode not in {"live", "silent"}:
            audio_mode = default.audio_mode
        return cls(
            llama_cpp_url=os.getenv("LLAMA_CPP_URL", default.llama_cpp_url),
            llama_cpp_model=os.getenv("LLAMA_CPP_MODEL", default.llama_cpp_model),
            llama_cpp_disable_think=_bool("LLAMA_CPP_DISABLE_THINK", default.llama_cpp_disable_think),
            kokoro_voice=os.getenv("KOKORO_VOICE", default.kokoro_voice),
            kokoro_lang=os.getenv("KOKORO_LANG", default.kokoro_lang),
            kokoro_repo_id=os.getenv("KOKORO_REPO_ID", default.kokoro_repo_id),
            sample_rate=_int("SAMPLE_RATE", default.sample_rate),
            tts_max_chars=_int("TTS_MAX_CHARS", default.tts_max_chars),
            tts_min_chars=_int("TTS_MIN_CHARS", default.tts_min_chars),
            tts_soft_chars=_int("TTS_SOFT_CHARS", default.tts_soft_chars),
            http_timeout_seconds=_float("HTTP_TIMEOUT_SECONDS", default.http_timeout_seconds),
            audio_queue_maxsize=_int("AUDIO_QUEUE_MAXSIZE", default.audio_queue_maxsize),
            audio_mode=audio_mode,
            warmup_enabled=os.getenv("VOICEBOT_WARMUP", "1") != "0",
        )
