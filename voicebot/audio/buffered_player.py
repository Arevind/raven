from __future__ import annotations

import io
import threading
import wave
from typing import Optional

import numpy as np

from voicebot.domain.interfaces import AudioSink


class BufferedAudioPlayer:
    def __init__(self, sample_rate: int, delegate: AudioSink | None = None) -> None:
        self.sample_rate = sample_rate
        self.delegate = delegate
        self._lock = threading.Lock()
        self._chunks: list[np.ndarray] = []
        self._latest_wav: bytes | None = None
        self._revision = 0

    def begin_capture(self) -> None:
        with self._lock:
            self._chunks = []

    def enqueue(self, audio_chunk: np.ndarray) -> None:
        mono = np.asarray(audio_chunk, dtype=np.float32).reshape(-1)
        if mono.size == 0:
            return
        with self._lock:
            self._chunks.append(mono.copy())
        if self.delegate is not None:
            self.delegate.enqueue(mono)

    def finalize_capture(self) -> None:
        with self._lock:
            if not self._chunks:
                self._latest_wav = None
                self._revision += 1
                return
            combined = np.concatenate(self._chunks, axis=0)
            self._latest_wav = self._to_wav_bytes(combined)
            self._revision += 1

    def clear_capture(self) -> None:
        with self._lock:
            self._chunks = []
            self._latest_wav = None

    def get_latest_wav(self) -> bytes | None:
        with self._lock:
            return self._latest_wav

    def get_revision(self) -> int:
        with self._lock:
            return self._revision

    def wait_until_idle(self) -> None:
        if self.delegate is not None:
            self.delegate.wait_until_idle()

    def close(self) -> None:
        if self.delegate is not None:
            self.delegate.close()

    def _to_wav_bytes(self, samples: np.ndarray) -> bytes:
        clipped = np.clip(samples, -1.0, 1.0)
        pcm = (clipped * 32767.0).astype(np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(pcm.tobytes())
        return buffer.getvalue()
