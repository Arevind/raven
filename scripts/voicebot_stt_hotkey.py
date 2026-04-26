from __future__ import annotations

import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from pynput import keyboard


API_BASE = os.getenv("VOICEBOT_API_BASE", "http://localhost:7860").rstrip("/")
CHAT_URL = f"{API_BASE}/api/chat"
STATE_URL = f"{API_BASE}/api/state"
HEALTH_URL = f"{API_BASE}/api/health"
HOTKEY_NAME = os.getenv("VOICEBOT_STT_HOTKEY", "f8").strip().lower()
MODEL_SIZE = os.getenv("VOICEBOT_STT_MODEL", "base.en").strip()
MODEL_DEVICE = os.getenv("VOICEBOT_STT_DEVICE", "cpu").strip()
MODEL_COMPUTE_TYPE = os.getenv("VOICEBOT_STT_COMPUTE_TYPE", "int8").strip()
SAMPLE_RATE = int(os.getenv("VOICEBOT_STT_SAMPLE_RATE", "16000"))
MIN_SECONDS = float(os.getenv("VOICEBOT_STT_MIN_SECONDS", "0.35"))


def _parse_hotkey(name: str):
    if len(name) == 1:
        return keyboard.KeyCode.from_char(name)
    return getattr(keyboard.Key, name)


def _key_matches(observed, target) -> bool:
    if isinstance(target, keyboard.KeyCode):
        return isinstance(observed, keyboard.KeyCode) and observed.char == target.char
    return observed == target


def _fetch_json(url: str, timeout: float = 8.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _wait_for_dashboard(timeout_seconds: int = 180) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            payload = _fetch_json(HEALTH_URL)
            if payload.get("status") == "ok":
                return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1)
    raise RuntimeError(f"Dashboard health not ready at {HEALTH_URL}")


def _wait_until_idle(max_wait_seconds: float = 60.0) -> bool:
    deadline = time.time() + max_wait_seconds
    while time.time() < deadline:
        try:
            state = _fetch_json(STATE_URL)
            if not bool(state.get("busy", False)):
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5)
    return False


def _post_chat(text: str) -> None:
    body = json.dumps({"text": text}).encode("utf-8")
    request = urllib.request.Request(
        CHAT_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15):
            return
    except urllib.error.HTTPError as exc:
        if exc.code == 409:
            if _wait_until_idle(max_wait_seconds=45):
                with urllib.request.urlopen(request, timeout=15):
                    return
        raise


class PushToTalkSTT:
    def __init__(self) -> None:
        self.hotkey = _parse_hotkey(HOTKEY_NAME)
        self.model = WhisperModel(MODEL_SIZE, device=MODEL_DEVICE, compute_type=MODEL_COMPUTE_TYPE)
        self._chunks: list[np.ndarray] = []
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._lock = threading.Lock()
        self._recording = False
        self._stopping = threading.Event()

    def _audio_callback(self, indata: np.ndarray, _frames, _time_info, _status) -> None:
        with self._lock:
            if self._recording:
                self._chunks.append(indata.copy())

    def _on_press(self, key) -> None:
        if not _key_matches(key, self.hotkey):
            return
        with self._lock:
            if self._recording:
                return
            self._recording = True
            self._chunks = []
        print("STT listening... release hotkey to send.", flush=True)

    def _on_release(self, key) -> None:
        if not _key_matches(key, self.hotkey):
            return
        with self._lock:
            if not self._recording:
                return
            self._recording = False
            chunks = self._chunks
            self._chunks = []
        if not chunks:
            print("No audio captured.", flush=True)
            return
        audio = np.concatenate(chunks, axis=0).reshape(-1).astype(np.float32)
        duration = audio.size / float(SAMPLE_RATE)
        if duration < MIN_SECONDS:
            print("Audio too short, ignored.", flush=True)
            return
        self._audio_queue.put(audio)
        print("Transcribing...", flush=True)

    def _transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            audio,
            language="en",
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 250,
                "speech_pad_ms": 200,
            },
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
        )
        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
        return " ".join(text.split()).strip()

    def run(self) -> int:
        _wait_for_dashboard()
        print(f"STT ready. Hold {HOTKEY_NAME.upper()} to talk.", flush=True)

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
            blocksize=0,
        ):
            with keyboard.Listener(on_press=self._on_press, on_release=self._on_release) as listener:
                try:
                    while not self._stopping.is_set():
                        try:
                            audio = self._audio_queue.get(timeout=0.2)
                        except queue.Empty:
                            continue
                        try:
                            text = self._transcribe(audio)
                            if not text:
                                print("No speech detected.", flush=True)
                                continue
                            print(f"Heard: {text}", flush=True)
                            _post_chat(text)
                            print("Sent to Raven.", flush=True)
                        except Exception as exc:  # noqa: BLE001
                            print(f"[stt] {exc}", flush=True)
                except KeyboardInterrupt:
                    pass
                finally:
                    listener.stop()
        return 0


def main() -> int:
    try:
        app = PushToTalkSTT()
        return app.run()
    except Exception as exc:  # noqa: BLE001
        print(
            f"STT startup failed: {exc}\n"
            "Install dependencies with: pip install faster-whisper pynput",
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
