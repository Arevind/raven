from __future__ import annotations

import queue
import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd


class AudioPlayer:
    def __init__(
        self,
        sample_rate: int,
        queue_maxsize: int = 32,
        apply_ramp: bool = True,
    ) -> None:
        self.sample_rate = sample_rate
        self.apply_ramp = apply_ramp
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=max(1, queue_maxsize))
        self._current = np.empty(0, dtype=np.float32)
        self._current_pos = 0
        self._closed = False
        self._lock = threading.Lock()
        self._stream: Optional[sd.OutputStream] = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._callback,
            blocksize=0,
        )
        self._stream.start()

    def enqueue(self, audio_chunk: np.ndarray) -> None:
        mono = np.asarray(audio_chunk, dtype=np.float32).reshape(-1)
        if mono.size == 0:
            return

        if self.apply_ramp:
            ramp_len = min(240, mono.size // 2)
            if ramp_len > 0:
                ramp = np.linspace(0.0, 1.0, ramp_len, dtype=np.float32)
                mono[:ramp_len] *= ramp
                mono[-ramp_len:] *= ramp[::-1]

        if self._closed:
            return
        self._queue.put(mono, timeout=3)

    def wait_until_idle(self) -> None:
        while not self._queue.empty() or self._current_pos < self._current.size:
            if self._closed:
                return
            time.sleep(0.01)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            stream = self._stream
            self._stream = None
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()

    def _callback(self, outdata: np.ndarray, frames: int, _time_info, _status) -> None:
        out = np.zeros(frames, dtype=np.float32)
        filled = 0

        while filled < frames:
            if self._current_pos >= self._current.size:
                try:
                    self._current = self._queue.get_nowait()
                    self._current_pos = 0
                except queue.Empty:
                    break

            remaining = self._current.size - self._current_pos
            if remaining <= 0:
                continue

            take = min(frames - filled, remaining)
            out[filled : filled + take] = self._current[self._current_pos : self._current_pos + take]
            self._current_pos += take
            filled += take

            if self._current_pos >= self._current.size:
                self._queue.task_done()

        outdata[:, 0] = out

