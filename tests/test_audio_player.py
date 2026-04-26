import queue
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

if "sounddevice" not in sys.modules:
    fake_sounddevice = types.SimpleNamespace(OutputStream=MagicMock())
    sys.modules["sounddevice"] = fake_sounddevice  # type: ignore[assignment]

from voicebot.audio.player import AudioPlayer


class AudioPlayerQueueTests(unittest.TestCase):
    @patch("voicebot.audio.player.sd.OutputStream")
    def test_enqueue_retries_when_queue_temporarily_full(self, stream_cls: MagicMock) -> None:
        stream = MagicMock()
        stream_cls.return_value = stream
        player = AudioPlayer(sample_rate=24000, queue_maxsize=1)

        mocked_queue = MagicMock()
        mocked_queue.put.side_effect = [queue.Full(), None]
        player._queue = mocked_queue  # type: ignore[assignment]

        player.enqueue(np.ones(16, dtype=np.float32))

        self.assertEqual(mocked_queue.put.call_count, 2)
        player.close()


if __name__ == "__main__":
    unittest.main()
