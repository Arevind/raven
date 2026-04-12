import asyncio
import unittest
from typing import AsyncIterator, Sequence

import numpy as np

from voicebot.domain.models import ChatTurn
from voicebot.runtime.engine import VoicebotEngine
from voicebot.text import SentenceChunker


class MockChatClient:
    async def stream_reply(self, user_text: str, history: Sequence[ChatTurn]) -> AsyncIterator[str]:
        del user_text, history
        for delta in ["Hello ", "world."]:
            await asyncio.sleep(0)
            yield delta

    async def close(self) -> None:
        return


class FailingChatClient:
    async def stream_reply(self, user_text: str, history: Sequence[ChatTurn]) -> AsyncIterator[str]:
        del user_text, history
        raise RuntimeError("network down")
        yield ""

    async def close(self) -> None:
        return


class MockTTSProvider:
    async def synthesize_stream(self, text: str) -> AsyncIterator[np.ndarray]:
        del text
        yield np.ones(8, dtype=np.float32)

    async def warmup(self) -> None:
        return


class MockAudioSink:
    def __init__(self) -> None:
        self.enqueued: list[np.ndarray] = []
        self.wait_calls = 0
        self.closed = False

    def enqueue(self, audio_chunk: np.ndarray) -> None:
        self.enqueued.append(audio_chunk)

    def wait_until_idle(self) -> None:
        self.wait_calls += 1

    def close(self) -> None:
        self.closed = True


class EngineIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_to_tts_to_audio_flow(self) -> None:
        audio = MockAudioSink()
        engine = VoicebotEngine(
            chat_client=MockChatClient(),
            tts_provider=MockTTSProvider(),
            audio_sink=audio,
            chunker_factory=lambda: SentenceChunker(max_chars=200, min_chars=1, soft_chars=0),
        )

        deltas: list[str] = []
        result = await engine.run_turn("hi", on_delta=lambda d: deltas.append(d))

        self.assertEqual("".join(deltas), "Hello world.")
        self.assertEqual(result.assistant_text, "Hello world.")
        self.assertEqual(len(engine.session.history), 1)
        self.assertGreaterEqual(len(audio.enqueued), 1)
        self.assertEqual(audio.wait_calls, 1)
        self.assertGreaterEqual(result.metrics.first_token_latency_ms, 0.0)
        self.assertGreaterEqual(result.metrics.first_audio_latency_ms, 0.0)
        self.assertGreaterEqual(result.metrics.total_turn_latency_ms, 0.0)

        await engine.close()
        self.assertTrue(audio.closed)

    async def test_error_does_not_append_turn_history(self) -> None:
        engine = VoicebotEngine(
            chat_client=FailingChatClient(),
            tts_provider=MockTTSProvider(),
            audio_sink=MockAudioSink(),
            chunker_factory=lambda: SentenceChunker(max_chars=200, min_chars=1, soft_chars=0),
        )
        with self.assertRaises(RuntimeError):
            await engine.run_turn("hello")
        self.assertEqual(len(engine.session.history), 0)
        await engine.close()


if __name__ == "__main__":
    unittest.main()

