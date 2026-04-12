from __future__ import annotations

import asyncio
from typing import AsyncIterator, Optional

import numpy as np
from kokoro import KPipeline


def sanitize_for_tts(text: str) -> str:
    cleaned = "".join(ch if (ch.isprintable() and ord(ch) < 0x10000) else " " for ch in text)
    return " ".join(cleaned.split()).strip()


class KokoroTTSProvider:
    def __init__(
        self,
        lang_code: str,
        voice: str,
        repo_id: str,
    ) -> None:
        self.lang_code = lang_code
        self.voice = voice
        self.repo_id = repo_id
        self._pipeline: Optional[KPipeline] = None
        self._init_lock = asyncio.Lock()

    async def warmup(self) -> None:
        try:
            async for _ in self.synthesize_stream("Hi."):
                break
        except Exception:
            return

    async def synthesize_stream(self, text: str) -> AsyncIterator[np.ndarray]:
        cleaned = sanitize_for_tts(text)
        if not cleaned:
            return

        pipeline = await self._get_pipeline()
        chunks = await asyncio.to_thread(self._collect_chunks, pipeline, cleaned, self.voice)
        if not chunks:
            raise RuntimeError("Kokoro produced no audio output.")
        yield np.concatenate(chunks, axis=0)

    async def _get_pipeline(self) -> KPipeline:
        if self._pipeline is not None:
            return self._pipeline
        async with self._init_lock:
            if self._pipeline is not None:
                return self._pipeline
            self._pipeline = await asyncio.to_thread(
                KPipeline,
                lang_code=self.lang_code,
                repo_id=self.repo_id,
            )
            return self._pipeline

    @staticmethod
    def _collect_chunks(pipeline: KPipeline, text: str, voice: str) -> list[np.ndarray]:
        chunks: list[np.ndarray] = []
        for _, _, audio in pipeline(text, voice=voice):
            chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        return chunks
