from __future__ import annotations

import argparse
import logging
import os

import uvicorn

from voicebot.config import VoicebotSettings
from voicebot.webapp import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the Astra Voice dashboard web app")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=7860, help="Port for the dashboard server.")
    parser.add_argument("--no-warmup", action="store_true", help="Disable startup warmup.")
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    # Keep voicebot runtime logs visible, but silence noisy dependency request logs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    os.environ["VOICEBOT_DASHBOARD_HOST"] = args.host
    os.environ["VOICEBOT_DASHBOARD_PORT"] = str(args.port)
    settings = VoicebotSettings.from_env()
    if args.no_warmup:
        settings.warmup_enabled = False
    app = create_app(settings)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="warning",
        access_log=False,
    )


if __name__ == "__main__":
    run()
