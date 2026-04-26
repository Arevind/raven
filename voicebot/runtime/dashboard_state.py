from __future__ import annotations

from dataclasses import dataclass, field
import logging
from threading import Lock
from time import perf_counter
from typing import Any

from voicebot.domain.models import TurnMetrics


def _now() -> float:
    return perf_counter()


logger = logging.getLogger("voicebot.runtime")


@dataclass(slots=True)
class WorkerSnapshot:
    name: str
    status: str = "idle"
    last_duration_ms: float = 0.0
    started_at: float | None = None
    completed_at: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TTSChunkSnapshot:
    index: int
    text: str
    char_count: int
    status: str = "queued"
    queued_at: float = field(default_factory=_now)
    started_at: float | None = None
    completed_at: float | None = None
    synth_latency_ms: float = 0.0
    audio_samples: int = 0


@dataclass(slots=True)
class DashboardObserver:
    bot_name: str
    model_name: str
    voice_name: str
    audio_mode: str
    langgraph_enabled: bool = False
    current_turn_id: int = 0
    current_status: str = "idle"
    avatar_state: str = "idle"
    current_user_text: str = ""
    current_assistant_text: str = ""
    latest_error: str = ""
    warmup_state: str = "pending"
    turn_started_at: float | None = None
    turn_completed_at: float | None = None
    metrics: TurnMetrics | None = None
    tts_chunks: list[TTSChunkSnapshot] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    turn_history: list[dict[str, Any]] = field(default_factory=list)
    workers: dict[str, WorkerSnapshot] = field(
        default_factory=lambda: {
            "llm": WorkerSnapshot(name="LLM"),
            "chunker": WorkerSnapshot(name="Chunker"),
            "tts": WorkerSnapshot(name="TTS"),
            "audio": WorkerSnapshot(name="Audio"),
            "orchestrator": WorkerSnapshot(name="Orchestrator"),
        }
    )
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def record(self, event_type: str, **payload: Any) -> None:
        handler = getattr(self, f"_handle_{event_type}", None)
        with self._lock:
            timestamp = _now()
            if handler is not None:
                handler(timestamp, **payload)
            self.events.append(
                {
                    "at": timestamp,
                    "type": event_type,
                    "payload": payload,
                }
            )
            if len(self.events) > 200:
                self.events = self.events[-200:]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            started_at = self.turn_started_at
            completed_at = self.turn_completed_at
            latest_events = list(self.events[-12:])
            chunks = [
                {
                    "index": chunk.index,
                    "text": chunk.text,
                    "char_count": chunk.char_count,
                    "status": chunk.status,
                    "synth_latency_ms": chunk.synth_latency_ms,
                    "audio_samples": chunk.audio_samples,
                }
                for chunk in self.tts_chunks
            ]
            workers = {
                key: {
                    "name": worker.name,
                    "status": worker.status,
                    "last_duration_ms": worker.last_duration_ms,
                    "details": dict(worker.details),
                }
                for key, worker in self.workers.items()
            }
            return {
                "bot_name": self.bot_name,
                "model_name": self.model_name,
                "voice_name": self.voice_name,
                "audio_mode": self.audio_mode,
                "langgraph_enabled": self.langgraph_enabled,
                "current_turn_id": self.current_turn_id,
                "current_status": self.current_status,
                "avatar_state": self.avatar_state,
                "current_user_text": self.current_user_text,
                "current_assistant_text": self.current_assistant_text,
                "latest_error": self.latest_error,
                "warmup_state": self.warmup_state,
                "turn_started_at": started_at,
                "turn_completed_at": completed_at,
                "turn_elapsed_ms": (
                    ((completed_at or _now()) - started_at) * 1000.0 if started_at is not None else 0.0
                ),
                "metrics": None
                if self.metrics is None
                else {
                    "first_token_latency_ms": self.metrics.first_token_latency_ms,
                    "first_audio_latency_ms": self.metrics.first_audio_latency_ms,
                    "total_turn_latency_ms": self.metrics.total_turn_latency_ms,
                    "llm_stream_latency_ms": self.metrics.llm_stream_latency_ms,
                    "tts_generation_latency_ms": self.metrics.tts_generation_latency_ms,
                    "tts_chunks_generated": self.metrics.tts_chunks_generated,
                    "audio_chunks_enqueued": self.metrics.audio_chunks_enqueued,
                    "audio_samples_enqueued": self.metrics.audio_samples_enqueued,
                },
                "tts_chunks": chunks,
                "workers": workers,
                "events": latest_events,
                "turn_history": list(self.turn_history[-8:]),
            }

    def _set_worker(
        self,
        worker_key: str,
        status: str,
        timestamp: float,
        duration_ms: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        worker = self.workers[worker_key]
        worker.status = status
        if status in {"running", "thinking", "speaking", "waiting"}:
            worker.started_at = timestamp
        else:
            worker.completed_at = timestamp
        if duration_ms is not None:
            worker.last_duration_ms = duration_ms
        if details:
            worker.details.update(details)

    def _find_chunk(self, chunk_index: int) -> TTSChunkSnapshot | None:
        for chunk in self.tts_chunks:
            if chunk.index == chunk_index:
                return chunk
        return None

    def _handle_warmup_started(self, timestamp: float, **_: Any) -> None:
        self.warmup_state = "warming"
        self.current_status = "warming"
        self.avatar_state = "thinking"
        self._set_worker("orchestrator", "waiting", timestamp)
        logger.info("Warmup started")

    def _handle_warmup_completed(self, timestamp: float, **_: Any) -> None:
        self.warmup_state = "ready"
        self.current_status = "idle"
        self.avatar_state = "idle"
        self._set_worker("orchestrator", "idle", timestamp)
        logger.info("Warmup completed")

    def _handle_warmup_failed(self, timestamp: float, message: str = "", **_: Any) -> None:
        self.warmup_state = "failed"
        self.current_status = "idle"
        self.avatar_state = "idle"
        self.latest_error = message
        self._set_worker("orchestrator", "idle", timestamp)
        logger.warning("Warmup failed: %s", message)

    def _handle_turn_started(self, timestamp: float, user_text: str, history_turns: int, **_: Any) -> None:
        self.current_turn_id += 1
        self.current_status = "thinking"
        self.avatar_state = "thinking"
        self.current_user_text = user_text
        self.current_assistant_text = ""
        self.turn_started_at = timestamp
        self.turn_completed_at = None
        self.metrics = None
        self.latest_error = ""
        self.tts_chunks = []
        self._set_worker("llm", "thinking", timestamp, details={"history_turns": history_turns})
        self._set_worker("chunker", "idle", timestamp, details={"emitted_chunks": 0})
        self._set_worker("tts", "idle", timestamp, details={"queued_chunks": 0})
        self._set_worker("audio", "idle", timestamp, details={"samples": 0})
        self._set_worker("orchestrator", "running", timestamp)
        logger.info("User message received (history_turns=%s): %s", history_turns, user_text)

    def _handle_llm_request_started(self, timestamp: float, model: str, **_: Any) -> None:
        self._set_worker("llm", "thinking", timestamp, details={"model": model})
        logger.info("LLM request started (model=%s)", model)

    def _handle_llm_delta(self, timestamp: float, delta: str, assistant_text: str, **_: Any) -> None:
        self.current_assistant_text = assistant_text
        self.avatar_state = "thinking"
        worker = self.workers["llm"]
        worker.details["delta_count"] = int(worker.details.get("delta_count", 0)) + 1
        worker.details["streamed_chars"] = len(assistant_text)
        worker.details["last_delta_chars"] = len(delta)
        if delta.strip():
            logger.info("LLM delta: %s", delta.strip())

    def _handle_llm_first_token(self, timestamp: float, latency_ms: float, **_: Any) -> None:
        self._set_worker("llm", "running", timestamp, duration_ms=latency_ms)

    def _handle_llm_completed(self, timestamp: float, latency_ms: float, assistant_chars: int, **_: Any) -> None:
        self._set_worker(
            "llm",
            "idle",
            timestamp,
            duration_ms=latency_ms,
            details={"assistant_chars": assistant_chars},
        )
        logger.info("LLM stream completed in %.0f ms (%s chars)", latency_ms, assistant_chars)

    def _handle_chunk_emitted(self, timestamp: float, text: str, chunk_index: int, queue_depth: int, **_: Any) -> None:
        self.tts_chunks.append(TTSChunkSnapshot(index=chunk_index, text=text, char_count=len(text)))
        worker = self.workers["chunker"]
        worker.status = "running"
        worker.details["emitted_chunks"] = int(worker.details.get("emitted_chunks", 0)) + 1
        worker.details["queue_depth"] = queue_depth
        self.workers["tts"].details["queued_chunks"] = len(self.tts_chunks)

    def _handle_chunk_flush_completed(self, timestamp: float, **_: Any) -> None:
        self._set_worker("chunker", "idle", timestamp)

    def _handle_tts_started(self, timestamp: float, chunk_index: int, text: str, **_: Any) -> None:
        chunk = self._find_chunk(chunk_index)
        if chunk is None:
            chunk = TTSChunkSnapshot(index=chunk_index, text=text, char_count=len(text))
            self.tts_chunks.append(chunk)
        chunk.status = "running"
        chunk.started_at = timestamp
        self.avatar_state = "speaking"
        self.current_status = "speaking"
        self._set_worker("tts", "speaking", timestamp, details={"active_chunk": chunk_index})
        logger.info("TTS started (chunk=%s, chars=%s)", chunk_index, len(text))

    def _handle_tts_completed(
        self,
        timestamp: float,
        chunk_index: int,
        latency_ms: float,
        audio_samples: int,
        **_: Any,
    ) -> None:
        chunk = self._find_chunk(chunk_index)
        if chunk is not None:
            chunk.status = "done"
            chunk.completed_at = timestamp
            chunk.synth_latency_ms = latency_ms
            chunk.audio_samples = audio_samples
        worker = self.workers["tts"]
        worker.details["completed_chunks"] = int(worker.details.get("completed_chunks", 0)) + 1
        worker.details["audio_samples"] = int(worker.details.get("audio_samples", 0)) + audio_samples
        self._set_worker("tts", "idle", timestamp, duration_ms=latency_ms)
        logger.info(
            "TTS completed (chunk=%s, latency=%.0f ms, samples=%s)",
            chunk_index,
            latency_ms,
            audio_samples,
        )

    def _handle_audio_enqueued(self, timestamp: float, chunk_index: int, audio_samples: int, **_: Any) -> None:
        worker = self.workers["audio"]
        worker.status = "running"
        worker.completed_at = timestamp
        worker.last_duration_ms = 0.0
        worker.details["last_chunk"] = chunk_index
        worker.details["samples"] = int(worker.details.get("samples", 0)) + audio_samples

    def _handle_playback_wait_started(self, timestamp: float, **_: Any) -> None:
        self._set_worker("audio", "waiting", timestamp)

    def _handle_playback_completed(self, timestamp: float, latency_ms: float, **_: Any) -> None:
        self._set_worker("audio", "idle", timestamp, duration_ms=latency_ms)

    def _handle_turn_completed(self, timestamp: float, assistant_text: str, metrics: TurnMetrics, **_: Any) -> None:
        self.current_assistant_text = assistant_text
        self.metrics = metrics
        self.current_status = "idle"
        self.avatar_state = "idle"
        self.turn_completed_at = timestamp
        self.turn_history.append(
            {
                "turn_id": self.current_turn_id,
                "user_text": self.current_user_text,
                "assistant_text": assistant_text,
                "total_turn_latency_ms": metrics.total_turn_latency_ms,
                "first_token_latency_ms": metrics.first_token_latency_ms,
                "first_audio_latency_ms": metrics.first_audio_latency_ms,
            }
        )
        if len(self.turn_history) > 20:
            self.turn_history = self.turn_history[-20:]
        self._set_worker("orchestrator", "idle", timestamp, duration_ms=metrics.total_turn_latency_ms)
        self._set_worker("audio", "idle", timestamp)
        logger.info("Assistant final response: %s", assistant_text)
        logger.info(
            "Turn completed (total=%.0f ms, first_token=%.0f ms, first_audio=%.0f ms, tts_chunks=%s)",
            metrics.total_turn_latency_ms,
            metrics.first_token_latency_ms,
            metrics.first_audio_latency_ms,
            metrics.tts_chunks_generated,
        )

    def _handle_turn_failed(self, timestamp: float, stage: str, message: str, **_: Any) -> None:
        self.current_status = "error"
        self.avatar_state = "idle"
        self.turn_completed_at = timestamp
        self.latest_error = f"{stage}: {message}"
        self._set_worker("orchestrator", "idle", timestamp)
        logger.warning("Turn failed at %s: %s", stage, message)
