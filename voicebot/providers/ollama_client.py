from __future__ import annotations

import json
from typing import AsyncIterator, Sequence

import httpx

from voicebot.domain.models import ChatTurn


class OllamaChatClient:
    def __init__(
        self,
        model: str,
        url: str,
        timeout_seconds: float = 300.0,
        keep_alive: str = "30m",
    ) -> None:
        self.model = model
        self.url = url
        self.keep_alive = keep_alive
        self._client = httpx.AsyncClient(timeout=timeout_seconds)

    async def stream_reply(
        self,
        user_text: str,
        history: Sequence[ChatTurn],
    ) -> AsyncIterator[str]:
        messages: list[dict[str, str]] = []
        for turn in history:
            messages.append({"role": "user", "content": turn.user_text})
            messages.append({"role": "assistant", "content": turn.assistant_text})
        messages.append({"role": "user", "content": user_text})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "keep_alive": self.keep_alive,
        }

        try:
            async with self._client.stream("POST", self.url, json=payload) as response:
                response.raise_for_status()
                async for raw_line in response.aiter_lines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("error"):
                        raise RuntimeError(f"Ollama error: {data['error']}")

                    delta = data.get("message", {}).get("content", "")
                    if delta:
                        yield delta

                    if data.get("done"):
                        break
        except httpx.HTTPStatusError as exc:
            details = exc.response.text
            raise RuntimeError(f"Ollama HTTP error {exc.response.status_code}: {details}") from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(
                "Could not connect to Ollama at http://127.0.0.1:11434. "
                "Start Ollama first (for example: `ollama serve`)."
            ) from exc

    async def close(self) -> None:
        await self._client.aclose()

