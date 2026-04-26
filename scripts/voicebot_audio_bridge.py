from __future__ import annotations

import io
import json
import sys
import time
import wave
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

try:
    import sounddevice as sd
except Exception:  # noqa: BLE001
    sd = None

try:
    import winsound
except Exception:  # noqa: BLE001
    winsound = None


STATE_URL = "http://localhost:7860/api/state"
AUDIO_URL = "http://localhost:7860/api/audio/latest"


def fetch_json(url: str) -> dict:
    with urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_bytes(url: str) -> bytes:
    request = Request(url, headers={"Cache-Control": "no-cache"})
    with urlopen(request, timeout=20) as response:
        return response.read()


def decode_wav_bytes(payload: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(payload), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise RuntimeError(f"Unsupported WAV sample width: {sample_width}")

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sample_rate


def play_wav(payload: bytes) -> None:
    if winsound is not None and sys.platform.startswith("win"):
        winsound.PlaySound(payload, winsound.SND_MEMORY)
        return

    if sd is None:
        raise RuntimeError("No playback backend is available. Install sounddevice or use Windows winsound.")

    audio, sample_rate = decode_wav_bytes(payload)
    if audio.size == 0:
        return
    sd.play(audio, samplerate=sample_rate, blocking=True)


def main() -> int:
    print("Voicebot audio bridge started. Waiting for synthesized speech...", flush=True)
    last_revision = 0

    while True:
        try:
            state = fetch_json(STATE_URL)
            revision = int(state.get("latest_audio_revision", 0))
            available = bool(state.get("latest_audio_available", False))
            busy = bool(state.get("busy", False))
            if available and not busy and revision > last_revision:
                payload = fetch_bytes(f"{AUDIO_URL}?rev={revision}")
                play_wav(payload)
                last_revision = revision
            time.sleep(0.35)
        except KeyboardInterrupt:
            print("Voicebot audio bridge stopped.", flush=True)
            return 0
        except HTTPError as exc:
            if exc.code != 404:
                print(f"[audio-bridge] HTTP error: {exc}", file=sys.stderr, flush=True)
            time.sleep(1.0)
        except URLError:
            time.sleep(1.0)
        except Exception as exc:  # noqa: BLE001
            print(f"[audio-bridge] {exc}", file=sys.stderr, flush=True)
            time.sleep(1.0)


if __name__ == "__main__":
    raise SystemExit(main())
