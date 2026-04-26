from __future__ import annotations

import argparse
import atexit
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    result = subprocess.run(cmd, cwd=str(cwd), check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _wait_for_dashboard(health_url: str, timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    last_error = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                if 200 <= response.status < 500:
                    return
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        time.sleep(2)
    raise RuntimeError(f"Timed out waiting for dashboard at {health_url}. Last error: {last_error}")


def _resolve_bridge_python(root: Path) -> str:
    if sys.platform.startswith("win"):
        win_venv = root / "venv" / "Scripts" / "python.exe"
        if win_venv.exists():
            return str(win_venv)
    else:
        nix_venv = root / "venv" / "bin" / "python"
        if nix_venv.exists():
            return str(nix_venv)
    return sys.executable


def main() -> int:
    parser = argparse.ArgumentParser(description="Start Voicebot Docker stack + host audio bridge + host STT.")
    parser.add_argument("--no-build", action="store_true", help="Skip docker image build.")
    parser.add_argument("--no-stt", action="store_true", help="Do not start host push-to-talk STT.")
    parser.add_argument("--stt-hotkey", default="f8", help="Push-to-talk hotkey for STT (default: f8).")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=180,
        help="How long to wait for dashboard health before failing.",
    )
    parser.add_argument(
        "--down-on-exit",
        action="store_true",
        help="Run `docker compose down` when this launcher exits.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    dashboard_url = "http://localhost:7860"
    health_url = f"{dashboard_url}/api/health"
    bridge_script = root / "scripts" / "voicebot_audio_bridge.py"
    stt_script = root / "scripts" / "voicebot_stt_hotkey.py"
    bridge_python = _resolve_bridge_python(root)
    bridge_proc: subprocess.Popen[str] | None = None
    stt_proc: subprocess.Popen[str] | None = None

    compose_cmd = ["docker", "compose", "up", "-d"]
    if not args.no_build:
        compose_cmd.append("--build")

    print("Starting Docker services...")
    _run(compose_cmd, cwd=root)

    print(f"Waiting for dashboard health: {health_url}")
    _wait_for_dashboard(health_url, timeout_seconds=args.timeout_seconds)
    print(f"Dashboard is ready at {dashboard_url}")

    print("Starting host audio bridge...")
    bridge_proc = subprocess.Popen(
        [bridge_python, str(bridge_script)],
        cwd=str(root),
    )
    print(f"Audio bridge running (pid={bridge_proc.pid}).")

    if not args.no_stt:
        print(f"Starting push-to-talk STT (hotkey: {args.stt_hotkey})...")
        env = dict(**os.environ)
        env["VOICEBOT_STT_HOTKEY"] = args.stt_hotkey
        env["VOICEBOT_API_BASE"] = dashboard_url
        stt_proc = subprocess.Popen(
            [bridge_python, str(stt_script)],
            cwd=str(root),
            env=env,
        )
        print(f"STT running (pid={stt_proc.pid}).")

    print("Press Ctrl+C to stop.")

    def _cleanup() -> None:
        nonlocal bridge_proc, stt_proc
        if stt_proc is not None and stt_proc.poll() is None:
            stt_proc.terminate()
            try:
                stt_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                stt_proc.kill()
        if bridge_proc is not None and bridge_proc.poll() is None:
            bridge_proc.terminate()
            try:
                bridge_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                bridge_proc.kill()
        if args.down_on_exit:
            subprocess.run(["docker", "compose", "down"], cwd=str(root), check=False)

    atexit.register(_cleanup)

    try:
        while True:
            if bridge_proc.poll() is not None:
                raise RuntimeError("Audio bridge exited unexpectedly.")
            if stt_proc is not None and stt_proc.poll() is not None:
                raise RuntimeError("STT process exited unexpectedly.")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping launcher...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
