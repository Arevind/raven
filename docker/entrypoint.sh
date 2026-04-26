#!/usr/bin/env sh
set -eu

LLAMA_CPP_URL="${LLAMA_CPP_URL:-http://llama-cpp:8080/v1/chat/completions}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-120}"

python - <<'PY'
import os
import sys
import time
import urllib.request

base_url = os.environ["LLAMA_CPP_URL"].rsplit("/v1/chat/completions", 1)[0]
candidate_health_urls = [f"{base_url}/health", f"{base_url}/v1/models"]
deadline = time.time() + int(os.environ.get("WAIT_TIMEOUT_SECONDS", "120"))
last_err = ""

while time.time() < deadline:
    for health_url in candidate_health_urls:
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                if 200 <= response.status < 500:
                    sys.exit(0)
        except Exception as exc:  # noqa: BLE001
            last_err = f"{health_url}: {exc}"
    time.sleep(2)

print(f"Timed out waiting for llama.cpp at {base_url}. Last error: {last_err}", file=sys.stderr)
sys.exit(1)
PY

exec "$@"
