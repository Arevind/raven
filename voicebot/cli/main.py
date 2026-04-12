from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Callable

from voicebot.audio import AudioPlayer
from voicebot.config import VoicebotSettings
from voicebot.providers import KokoroTTSProvider, OllamaChatClient
from voicebot.runtime import VoicebotEngine
from voicebot.text import SentenceChunker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI chatbot with streaming TTS")
    parser.add_argument(
        "--kokoro-voice",
        default=None,
        help="Kokoro voice id (e.g. af_heart).",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Disable startup warmup.",
    )
    return parser.parse_args()


def _chunker_factory(settings: VoicebotSettings) -> Callable[[], SentenceChunker]:
    def _make() -> SentenceChunker:
        return SentenceChunker(
            max_chars=settings.tts_max_chars,
            min_chars=settings.tts_min_chars,
            soft_chars=settings.tts_soft_chars,
        )

    return _make


async def _interactive_loop(settings: VoicebotSettings) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    chat_client = OllamaChatClient(
        model=settings.ollama_model,
        url=settings.ollama_url,
        timeout_seconds=settings.http_timeout_seconds,
    )
    tts_provider = KokoroTTSProvider(
        lang_code=settings.kokoro_lang,
        voice=settings.kokoro_voice,
        repo_id=settings.kokoro_repo_id,
    )
    audio_sink = AudioPlayer(
        sample_rate=settings.sample_rate,
        queue_maxsize=settings.audio_queue_maxsize,
    )
    engine = VoicebotEngine(
        chat_client=chat_client,
        tts_provider=tts_provider,
        audio_sink=audio_sink,
        chunker_factory=_chunker_factory(settings),
    )

    print(
        f"CLI Chatbot started (model: {settings.ollama_model}, "
        f"tts_provider: kokoro, voice: {settings.kokoro_voice}, "
        f"langgraph: {'on' if engine.orchestrator.using_langgraph else 'fallback'})"
    )
    if not engine.orchestrator.using_langgraph:
        print("LangGraph is not installed. Install with `pip install langgraph` for graph runtime.")
    print("Type /quit to exit.\n")

    try:
        if settings.warmup_enabled:
            await engine.warmup()

        while True:
            try:
                user_text = (await asyncio.to_thread(input, "You: ")).strip()
            except (KeyboardInterrupt, EOFError):
                print("\nBye.")
                break

            if not user_text:
                continue
            if user_text.lower() in {"/quit", "/exit"}:
                print("Bye.")
                break

            print("Bot: ", end="", flush=True)
            try:
                await engine.run_turn(user_text, on_delta=lambda delta: print(delta, end="", flush=True))
                print("\n")
            except Exception as exc:
                print(f"\n[Error] {exc}\n")
    finally:
        await engine.close()


def run() -> None:
    args = parse_args()
    settings = VoicebotSettings.from_env()
    if args.kokoro_voice:
        settings.kokoro_voice = args.kokoro_voice
    if args.no_warmup:
        settings.warmup_enabled = False
    asyncio.run(_interactive_loop(settings))


if __name__ == "__main__":
    run()

