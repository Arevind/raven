from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from statistics import mean
from typing import AsyncIterator, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voicebot.domain.models import ChatTurn
from voicebot.runtime import VoicebotEngine
from voicebot.text import SentenceChunker


class SyntheticChatClient:
    def __init__(self, deltas: list[str], delay_ms: float) -> None:
        self.deltas = deltas
        self.delay_ms = delay_ms

    async def stream_reply(self, user_text: str, history: Sequence[ChatTurn]) -> AsyncIterator[str]:
        del user_text, history
        for part in self.deltas:
            await asyncio.sleep(self.delay_ms / 1000.0)
            yield part

    async def close(self) -> None:
        return


class SyntheticTTSProvider:
    def __init__(self, delay_ms: float) -> None:
        self.delay_ms = delay_ms

    async def synthesize_stream(self, text: str) -> AsyncIterator[np.ndarray]:
        del text
        await asyncio.sleep(self.delay_ms / 1000.0)
        yield np.ones(1024, dtype=np.float32)

    async def warmup(self) -> None:
        return


class NullAudioSink:
    def enqueue(self, audio_chunk: np.ndarray) -> None:
        del audio_chunk

    def wait_until_idle(self) -> None:
        return

    def close(self) -> None:
        return


async def run_benchmark(turns: int, llm_delay_ms: float, tts_delay_ms: float) -> None:
    engine = VoicebotEngine(
        chat_client=SyntheticChatClient(
            deltas=["This is ", "a benchmark ", "response."],
            delay_ms=llm_delay_ms,
        ),
        tts_provider=SyntheticTTSProvider(delay_ms=tts_delay_ms),
        audio_sink=NullAudioSink(),
        chunker_factory=lambda: SentenceChunker(max_chars=280, min_chars=1, soft_chars=0),
    )

    results = []
    for i in range(turns):
        result = await engine.run_turn(f"turn-{i}")
        results.append(result.metrics)

    await engine.close()

    print(f"Turns: {turns}")
    print(f"Avg first token latency (ms): {mean(x.first_token_latency_ms for x in results):.2f}")
    print(f"Avg first audio latency (ms): {mean(x.first_audio_latency_ms for x in results):.2f}")
    print(f"Avg total turn latency (ms): {mean(x.total_turn_latency_ms for x in results):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic voicebot latency benchmark")
    parser.add_argument("--turns", type=int, default=20)
    parser.add_argument("--llm-delay-ms", type=float, default=15.0)
    parser.add_argument("--tts-delay-ms", type=float, default=20.0)
    args = parser.parse_args()
    asyncio.run(run_benchmark(args.turns, args.llm_delay_ms, args.tts_delay_ms))


if __name__ == "__main__":
    main()
