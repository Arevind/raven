from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Awaitable, Callable

from voicebot.domain.interfaces import AudioSink, ChatClient, Chunker, RuntimeObserver, TTSProvider
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
        observer: RuntimeObserver | None = None,
    ) -> None:
        self.chat_client = chat_client
        self.tts_provider = tts_provider
        self.audio_sink = audio_sink
        self.chunker_factory = chunker_factory
        self.session = session or SessionState()
        self.observer = observer
        self.orchestrator = ConversationOrchestrator(chat_client, tts_provider, audio_sink, observer=observer)
        self._closed = False

    async def warmup(self) -> None:
        if self.observer is not None:
            self.observer.record("warmup_started")
        try:
            await self.tts_provider.warmup()
        except Exception as exc:
            if self.observer is not None:
                self.observer.record("warmup_failed", message=str(exc))
            raise
        if self.observer is not None:
            self.observer.record("warmup_completed")

    async def run_turn(self, user_text: str, on_delta: OnDelta | None = None) -> TurnResult:
        if self._closed:
            raise RuntimeError("Engine is closed.")
        started_at = perf_counter()
        if self.observer is not None:
            self.observer.record(
                "turn_started",
                user_text=user_text,
                history_turns=len(self.session.history),
            )

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
            "llm_completed_at": None,
            "playback_completed_at": None,
            "tts_generation_total_ms": 0.0,
            "tts_chunk_count": 0,
            "audio_chunks_enqueued": 0,
            "audio_samples_enqueued": 0,
            "next_tts_chunk_index": 0,
        }
        try:
            out = await self.orchestrator.run(state)
        except Exception as exc:
            if self.observer is not None:
                self.observer.record("turn_failed", stage="runtime", message=str(exc))
            raise
        ended_at = perf_counter()
        observer_snapshot = None
        if self.observer is not None and hasattr(self.observer, "snapshot"):
            observer_snapshot = self.observer.snapshot()  # type: ignore[assignment]

        first_token_at = out["first_token_at"] or ended_at
        first_audio_at = out["first_audio_at"] or ended_at
        llm_completed_at = out["llm_completed_at"] or ended_at
        tts_generation_latency_ms = out["tts_generation_total_ms"]
        tts_chunk_count = out["tts_chunk_count"]
        audio_chunks_enqueued = out["audio_chunks_enqueued"]
        audio_samples_enqueued = out["audio_samples_enqueued"]

        if observer_snapshot is not None:
            if not tts_generation_latency_ms:
                tts_generation_latency_ms = sum(
                    chunk.get("synth_latency_ms", 0.0) for chunk in observer_snapshot.get("tts_chunks", [])
                )
            if not tts_chunk_count:
                tts_chunk_count = len(observer_snapshot.get("tts_chunks", []))
            if not audio_chunks_enqueued:
                audio_chunks_enqueued = sum(
                    1 for chunk in observer_snapshot.get("tts_chunks", []) if chunk.get("audio_samples", 0) > 0
                )
            if not audio_samples_enqueued:
                audio_samples_enqueued = (
                    observer_snapshot.get("workers", {})
                    .get("audio", {})
                    .get("details", {})
                    .get("samples", 0)
                )

        metrics = TurnMetrics(
            started_at=started_at,
            first_token_latency_ms=(first_token_at - started_at) * 1000.0,
            first_audio_latency_ms=(first_audio_at - started_at) * 1000.0,
            total_turn_latency_ms=(ended_at - started_at) * 1000.0,
            llm_stream_latency_ms=(llm_completed_at - started_at) * 1000.0,
            tts_generation_latency_ms=tts_generation_latency_ms,
            tts_chunks_generated=tts_chunk_count,
            audio_chunks_enqueued=audio_chunks_enqueued,
            audio_samples_enqueued=audio_samples_enqueued,
        )
        if self.observer is not None:
            self.observer.record(
                "turn_completed",
                assistant_text=out["assistant_text"],
                metrics=metrics,
            )
        return TurnResult(assistant_text=out["assistant_text"], metrics=metrics)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.audio_sink.close()
        await self.chat_client.close()
        await asyncio.sleep(0)
