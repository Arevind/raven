from __future__ import annotations

import numpy as np


class NullAudioPlayer:
    def enqueue(self, audio_chunk: np.ndarray) -> None:
        del audio_chunk

    def wait_until_idle(self) -> None:
        return

    def close(self) -> None:
        return
