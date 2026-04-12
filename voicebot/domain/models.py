from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import List


@dataclass(slots=True)
class ChatTurn:
    user_text: str
    assistant_text: str
    created_at_monotonic: float = field(default_factory=perf_counter)


@dataclass(slots=True)
class TTSChunk:
    text: str
    char_count: int


@dataclass(slots=True)
class SessionState:
    history: List[ChatTurn] = field(default_factory=list)


@dataclass(slots=True)
class TurnMetrics:
    started_at: float
    first_token_latency_ms: float
    first_audio_latency_ms: float
    total_turn_latency_ms: float


@dataclass(slots=True)
class TurnResult:
    assistant_text: str
    metrics: TurnMetrics

