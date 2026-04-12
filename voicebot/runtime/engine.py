from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Awaitable, Callable

from voicebot.domain.interfaces import AudioSink, ChatClient, Chunker, TTSProvider
from voicebot.domain.models import SessionState, TurnMetrics, TurnResult
from voicebot.orchestration.graph import ConversationOrchestrator


OnDelta = Callable[[str], Awaitable[None] | None]


class VoicebotEngine:
    def __init__(
        self,
        chat_client: ChatClient,
        tts_provider: TTSProvider,
        audio_sink: AudioSink,
        chunker_factory: Callable[[], Chunker],
        session: SessionState | None = None,
    ) -> None:
        self.chat_client = chat_client
        self.tts_provider = tts_provider
        self.audio_sink = audio_sink
        self.chunker_factory = chunker_factory
        self.session = session or SessionState()
        self.orchestrator = ConversationOrchestrator(chat_client, tts_provider, audio_sink)
        self._closed = False

    async def warmup(self) -> None:
        await self.tts_provider.warmup()

    async def run_turn(self, user_text: str, on_delta: OnDelta | None = None) -> TurnResult:
        if self._closed:
            raise RuntimeError("Engine is closed.")
        started_at = perf_counter()

        state = {
            "user_text": user_text,
            "session": self.session,
            "chunker": self.chunker_factory(),
            "on_delta": on_delta,
            "assistant_text": "",
            "tts_text_chunks": [],
            "first_token_at": None,
            "first_audio_at": None,
            "started_at": started_at,
        }
        out = await self.orchestrator.run(state)
        ended_at = perf_counter()

        first_token_at = out["first_token_at"] or ended_at
        first_audio_at = out["first_audio_at"] or ended_at
        metrics = TurnMetrics(
            started_at=started_at,
            first_token_latency_ms=(first_token_at - started_at) * 1000.0,
            first_audio_latency_ms=(first_audio_at - started_at) * 1000.0,
            total_turn_latency_ms=(ended_at - started_at) * 1000.0,
        )
        return TurnResult(assistant_text=out["assistant_text"], metrics=metrics)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.audio_sink.close()
        await self.chat_client.close()
        await asyncio.sleep(0)

