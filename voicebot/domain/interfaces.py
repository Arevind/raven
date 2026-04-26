from __future__ import annotations

from typing import Any, AsyncIterator, Protocol, Sequence

import numpy as np

from .models import ChatTurn


class ChatClient(Protocol):
    async def stream_reply(self, user_text: str, history: Sequence[ChatTurn]) -> AsyncIterator[str]:
        ...

    async def close(self) -> None:
        ...


class TTSProvider(Protocol):
    async def synthesize_stream(self, text: str) -> AsyncIterator[np.ndarray]:
        ...

    async def warmup(self) -> None:
        ...


class AudioSink(Protocol):
    def enqueue(self, audio_chunk: np.ndarray) -> None:
        ...

    def wait_until_idle(self) -> None:
        ...

    def close(self) -> None:
        ...


class Chunker(Protocol):
    def feed(self, text: str) -> list[str]:
        ...

    def flush(self) -> str | None:
        ...


class RuntimeObserver(Protocol):
    def record(self, event_type: str, **payload: Any) -> None:
        ...
