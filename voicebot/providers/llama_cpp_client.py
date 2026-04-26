from __future__ import annotations

import asyncio
import json
import re
from typing import Any, AsyncIterator, Callable, Sequence

import httpx

from voicebot.domain.models import ChatTurn


class _ThinkTagFilter:
    START_TAG = "<think>"
    END_TAG = "</think>"

    def __init__(self) -> None:
        self._pending = ""
        self._inside_think = False

    def feed(self, delta: str) -> str:
        if not delta:
            return ""
        self._pending += delta
        out_parts: list[str] = []

        while self._pending:
            if self._inside_think:
                end_idx = self._pending.find(self.END_TAG)
                if end_idx == -1:
                    keep = len(self.END_TAG) - 1
                    self._pending = self._pending[-keep:] if len(self._pending) > keep else self._pending
                    break
                self._pending = self._pending[end_idx + len(self.END_TAG) :]
                self._inside_think = False
                continue

            start_idx = self._pending.find(self.START_TAG)
            if start_idx != -1:
                if start_idx > 0:
                    out_parts.append(self._pending[:start_idx])
                self._pending = self._pending[start_idx + len(self.START_TAG) :]
                self._inside_think = True
                continue

            keep = self._suffix_prefix_len(self._pending, self.START_TAG)
            emit_upto = len(self._pending) - keep
            if emit_upto > 0:
                out_parts.append(self._pending[:emit_upto])
                self._pending = self._pending[emit_upto:]
            break

        return "".join(out_parts)

    def flush(self) -> str:
        if self._inside_think:
            self._pending = ""
            return ""
        out = self._pending
        self._pending = ""
        return out

    @staticmethod
    def _suffix_prefix_len(text: str, token: str) -> int:
        max_check = min(len(text), len(token) - 1)
        for size in range(max_check, 0, -1):
            if text.endswith(token[:size]):
                return size
        return 0


class LlamaCppChatClient:
    def __init__(
        self,
        model: str,
        url: str,
        timeout_seconds: float = 300.0,
        disable_think_mode: bool = True,
        system_prompt: str | None = None,
        context_provider: Callable[[], str] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        tool_spec: str = "",
        tool_max_rounds: int = 2,
        tool_fallback_selector: Callable[[str], tuple[str, dict[str, Any]] | None] | None = None,
    ) -> None:
        self.model = model
        self.url = url
        self.disable_think_mode = disable_think_mode
        self.system_prompt = system_prompt or ""
        self.context_provider = context_provider
        self.tool_executor = tool_executor
        self.tool_spec = tool_spec.strip()
        self.tool_max_rounds = max(1, tool_max_rounds)
        self.tool_fallback_selector = tool_fallback_selector
        self._client = httpx.AsyncClient(timeout=timeout_seconds)

    TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

    def _build_messages(self, user_text: str, history: Sequence[ChatTurn]) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        system_parts: list[str] = []
        if self.system_prompt.strip():
            system_parts.append(self.system_prompt.strip())
        if self.disable_think_mode:
            system_parts.append(
                "Thinking mode is disabled. Do not produce chain-of-thought, reasoning traces, "
                "or any content inside <think> tags. Reply with only the final answer."
            )
        if self.context_provider is not None:
            context = self.context_provider().strip()
            if context:
                system_parts.append(context)
        if self.tool_executor is not None and self.tool_spec:
            system_parts.append(self.tool_spec)
        if system_parts:
            messages.append({"role": "system", "content": "\n\n".join(system_parts)})
        for turn in history:
            messages.append({"role": "user", "content": turn.user_text})
            messages.append({"role": "assistant", "content": turn.assistant_text})
        messages.append({"role": "user", "content": user_text})
        return messages

    def _parse_tool_call(self, text: str) -> tuple[str, dict[str, Any]] | None:
        match = self.TOOL_CALL_PATTERN.search(text)
        payload_text = match.group(1) if match is not None else ""
        stripped = text.strip().strip("`")
        if stripped.lower().startswith("json\n"):
            stripped = stripped.split("\n", 1)[1].strip()
        if not payload_text and stripped:
            # Accept direct JSON payloads.
            if stripped.startswith("{") and stripped.endswith("}"):
                payload_text = stripped
        if not payload_text and stripped:
            # Accept shorthand forms such as:
            # add_note{"summary":"..."} or add_note({"summary":"..."})
            fn_match = re.match(
                r"^([a-z_][a-z0-9_]*)\s*(?:\(\s*)?(\{.*\})\s*(?:\))?$",
                stripped,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if fn_match is not None:
                name = fn_match.group(1).strip()
                args_text = fn_match.group(2).strip()
                try:
                    arguments = json.loads(args_text)
                except json.JSONDecodeError:
                    return None
                if isinstance(arguments, dict):
                    return name, arguments
                return None
        if not payload_text:
            return None
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            return None
        name = str(payload.get("name", "")).strip()
        arguments = payload.get("arguments", {})
        if not name or not isinstance(arguments, dict):
            return None
        return name, arguments

    @staticmethod
    async def _yield_text_chunks(text: str) -> AsyncIterator[str]:
        if not text:
            return
        for piece in re.findall(r"\S+\s*", text):
            if piece:
                yield piece
                await asyncio.sleep(0)

    async def _collect_model_reply(self, messages: list[dict[str, str]]) -> str:
        parts: list[str] = []
        async for delta in self._stream_model(messages):
            parts.append(delta)
        return "".join(parts).strip()

    @staticmethod
    def _tool_result_to_text(tool_name: str, tool_result: dict[str, Any]) -> str:
        if not tool_result.get("ok", False):
            return str(tool_result.get("error", "Tool execution failed."))
        if tool_name == "add_note":
            item = tool_result.get("item", {})
            return f"Added to notes: {item.get('summary', '')}."
        if tool_name == "add_reminder":
            item = tool_result.get("item", {})
            due = str(item.get("due_hint", "")).strip()
            if due:
                return f"Added to reminders: {item.get('summary', '')} (due: {due})."
            return f"Added to reminders: {item.get('summary', '')}."
        if tool_name in {"list_notes", "list_reminders"}:
            items = tool_result.get("items", [])
            if not items:
                return "none."
            lines = []
            for item in items[:10]:
                summary = str(item.get("summary", "")).strip()
                due_hint = str(item.get("due_hint", "")).strip()
                if due_hint:
                    lines.append(f"- {summary} (due: {due_hint})")
                else:
                    lines.append(f"- {summary}")
            return "\n".join(lines)
        if tool_name in {"update_note", "update_reminder"}:
            item = tool_result.get("item", {})
            return f"Updated: {item.get('summary', '')}."
        if tool_name in {"delete_note", "delete_reminder"}:
            return "Deleted." if tool_result.get("deleted") else "Nothing was deleted."
        return "Done."

    async def _stream_model(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }

        think_filter = _ThinkTagFilter()
        try:
            async with self._client.stream("POST", self.url, json=payload) as response:
                response.raise_for_status()
                async for raw_line in response.aiter_lines():
                    line = raw_line.strip()
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if line == "[DONE]":
                        break

                    data = json.loads(line)
                    error = data.get("error")
                    if error:
                        raise RuntimeError(f"llama.cpp error: {error}")

                    choices = data.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {}).get("content", "")
                    if not delta:
                        continue
                    safe_delta = think_filter.feed(delta) if self.disable_think_mode else delta
                    if safe_delta:
                        yield safe_delta
        except httpx.HTTPStatusError as exc:
            details = "<response body unavailable>"
            try:
                details = (await exc.response.aread()).decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                pass
            raise RuntimeError(f"llama.cpp HTTP error {exc.response.status_code}: {details}") from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(
                "Could not connect to llama.cpp at http://127.0.0.1:8080. "
                "Start llama-server first and confirm /v1/chat/completions is reachable."
            ) from exc

        if self.disable_think_mode:
            tail = think_filter.flush()
            if tail:
                yield tail

    async def stream_reply(
        self,
        user_text: str,
        history: Sequence[ChatTurn],
    ) -> AsyncIterator[str]:
        messages = self._build_messages(user_text=user_text, history=history)
        if self.tool_executor is None or not self.tool_spec:
            async for delta in self._stream_model(messages):
                yield delta
            return

        rounds = 0
        final_text = ""
        fallback_used = False
        last_tool_name: str | None = None
        last_tool_result: dict[str, Any] | None = None
        while rounds < self.tool_max_rounds:
            rounds += 1
            reply_text = await self._collect_model_reply(messages)
            parsed = self._parse_tool_call(reply_text)
            used_fallback_this_round = False
            if parsed is None and not fallback_used and self.tool_fallback_selector is not None:
                parsed = self.tool_fallback_selector(user_text)
                fallback_used = parsed is not None
                used_fallback_this_round = parsed is not None
            if parsed is None:
                final_text = reply_text
                break
            tool_name, tool_args = parsed
            tool_result = self.tool_executor(tool_name, tool_args)
            last_tool_name, last_tool_result = tool_name, tool_result
            if (
                not tool_result.get("ok", False)
                and str(tool_result.get("error", "")).startswith("unknown tool")
                and not fallback_used
                and self.tool_fallback_selector is not None
            ):
                fallback = self.tool_fallback_selector(user_text)
                if fallback is not None:
                    tool_name, tool_args = fallback
                    tool_result = self.tool_executor(tool_name, tool_args)
                    last_tool_name, last_tool_result = tool_name, tool_result
                    fallback_used = True
                    used_fallback_this_round = True
            assistant_tool_text = reply_text
            if used_fallback_this_round:
                assistant_tool_text = (
                    "<tool_call>"
                    f"{json.dumps({'name': tool_name, 'arguments': tool_args}, ensure_ascii=True)}"
                    "</tool_call>"
                )
            messages.append({"role": "assistant", "content": assistant_tool_text})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Tool result JSON:\n"
                        f"{json.dumps(tool_result, ensure_ascii=True)}\n"
                        "Now continue and answer the user's request naturally."
                    ),
                }
            )

        if not final_text:
            final_text = await self._collect_model_reply(messages)
        if self._parse_tool_call(final_text) is not None and last_tool_name is not None and last_tool_result is not None:
            final_text = self._tool_result_to_text(last_tool_name, last_tool_result)
        async for chunk in self._yield_text_chunks(final_text):
            yield chunk

    async def close(self) -> None:
        await self._client.aclose()
