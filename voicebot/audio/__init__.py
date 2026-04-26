from .buffered_player import BufferedAudioPlayer
from .null_player import NullAudioPlayer
try:
    from .player import AudioPlayer
except ModuleNotFoundError as exc:
    class AudioPlayer:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            raise RuntimeError(
                "Live audio playback requires the optional sounddevice dependency."
            ) from exc

__all__ = ["AudioPlayer", "BufferedAudioPlayer", "NullAudioPlayer"]
