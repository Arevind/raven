from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from voicebot.audio import AudioPlayer, BufferedAudioPlayer, NullAudioPlayer
from voicebot.config import VoicebotSettings
from voicebot.providers import KokoroTTSProvider, LlamaCppChatClient
from voicebot.runtime import VoicebotEngine
from voicebot.runtime.dashboard_state import DashboardObserver
from voicebot.text import SentenceChunker
from voicebot.webapp.memory_store import MemoryStore


RAVEN_SYSTEM_PROMPT = (
    "You are Raven, a local voice assistant for productivity and study support. "
    "Your purpose is to help the user think clearly, stay organized, and take action. "
    "Be concise, practical, and friendly. "
    "When asked about notes or reminders, rely on the provided memory context as the source of truth."
)

RAVEN_MEMORY_TOOL_SPEC = (
    "You can use memory CRUD tools for notes and reminders.\n"
    "If a user asks to create, read, update, delete, summarize, or manage notes/reminders, call exactly one tool first.\n"
    "Tool call format (and only this format):\n"
    "<tool_call>{\"name\":\"tool_name\",\"arguments\":{...}}</tool_call>\n"
    "Available tools:\n"
    "- add_note {\"summary\": string}\n"
    "- add_reminder {\"summary\": string, \"due_hint\": string?}\n"
    "- list_notes {\"limit\": int?, \"query\": string?}\n"
    "- list_reminders {\"limit\": int?, \"query\": string?}\n"
    "- update_note {\"id\": string, \"summary\": string}\n"
    "- update_reminder {\"id\": string, \"summary\": string?, \"due_hint\": string?}\n"
    "- delete_note {\"id\": string}\n"
    "- delete_reminder {\"id\": string}\n"
    "If no tool is needed, answer normally without a tool call."
)


def _chunker_factory(settings: VoicebotSettings):
    def _make() -> SentenceChunker:
        return SentenceChunker(
            max_chars=settings.tts_max_chars,
            min_chars=settings.tts_min_chars,
            soft_chars=settings.tts_soft_chars,
        )

    return _make


@dataclass(slots=True)
class ConversationMessage:
    role: str
    content: str


@dataclass(slots=True)
class MemoryIntentResult:
    kind: str
    summary: str
    due_hint: str | None = None


class DashboardService:
    def __init__(self, settings: VoicebotSettings) -> None:
        self.settings = settings
        self.observer = DashboardObserver(
            bot_name="raven",
            model_name=settings.llama_cpp_model,
            voice_name=settings.kokoro_voice,
            audio_mode=settings.audio_mode,
        )
        self.engine: VoicebotEngine | None = None
        self._engine_lock = asyncio.Lock()
        self._turn_lock = asyncio.Lock()
        self._warmup_lock = asyncio.Lock()
        self._turn_task: asyncio.Task[None] | None = None
        self._messages: list[ConversationMessage] = []
        self._warmed_up = False
        self._audio_sink: BufferedAudioPlayer | None = None
        memory_path = Path.cwd() / ".voicebot_data" / "raven_memory.jsonl"
        self._memory_store = MemoryStore(memory_path)
        self._active_memory_tab = "notes"

    @staticmethod
    def _extract_memory_tab(text: str) -> str | None:
        lowered = text.lower()
        if "reminder tab" in lowered or "reminders tab" in lowered:
            return "reminders"
        if "notes tab" in lowered or "note tab" in lowered:
            return "notes"
        if any(token in lowered for token in ("show reminders", "open reminders", "switch to reminders", "make reminders active")):
            return "reminders"
        if any(token in lowered for token in ("show notes", "open notes", "switch to notes", "make notes active")):
            return "notes"
        return None

    @staticmethod
    def _looks_like_reminder(text: str) -> bool:
        lowered = text.lower()
        base_match = any(
            token in lowered
            for token in (
                "remind me",
                "set a reminder",
                "create a reminder",
                "add a reminder",
                "make a reminder",
                "save a reminder",
                "reminder:",
            )
        )
        return (
            base_match
            or re.search(
                r"\b(?:add|create|make|save|set)\s+(?:me\s+)?(?:a\s+)?(?:new\s+)?reminder\b",
                lowered,
            )
            is not None
        )

    @staticmethod
    def _looks_like_note(text: str) -> bool:
        lowered = text.lower()
        base_match = any(
            token in lowered
            for token in (
                "note this",
                "write this down",
                "take a note",
                "note:",
                "save this note",
                "add a note",
                "create a note",
                "make a note",
                "save a note",
            )
        )
        return (
            base_match
            or re.search(r"\bnote\b.*\bdown\b", lowered) is not None
            or re.search(
                r"\b(?:add|create|make|save)\s+(?:me\s+)?(?:a\s+)?(?:new\s+)?note\b",
                lowered,
            )
            is not None
        )

    @staticmethod
    def _summarize_memory(text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        cleaned = re.sub(
            r"^(please\s+)?(raven[:,]?\s+)?(can you\s+)?(remind me|set (?:me\s+)?(?:a\s+)?(?:new\s+)?reminder|create (?:me\s+)?(?:a\s+)?(?:new\s+)?reminder|add (?:me\s+)?(?:a\s+)?(?:new\s+)?reminder|make (?:me\s+)?(?:a\s+)?(?:new\s+)?reminder|save (?:me\s+)?(?:a\s+)?(?:new\s+)?reminder|note this|take a note|write this down|add (?:me\s+)?(?:a\s+)?(?:new\s+)?note|create (?:me\s+)?(?:a\s+)?(?:new\s+)?note|make (?:me\s+)?(?:a\s+)?(?:new\s+)?note|save (?:me\s+)?(?:a\s+)?(?:new\s+)?note)\s+(mentioning\s+|telling\s+|about\s+|for\s+|that\s+|to\s+)?",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip(" .")
        if not cleaned:
            cleaned = text.strip()
        sentence = cleaned.split(".")[0].strip()
        candidate = sentence if sentence else cleaned
        return candidate[:180].rstrip()

    @staticmethod
    def _extract_due_hint(text: str) -> str | None:
        match = re.search(
            r"\b(?:at|on|by|before|after)\s+([^.,;]{2,50})",
            text,
            flags=re.IGNORECASE,
        )
        if match is not None:
            hint = match.group(1).strip(" .")
            hint = re.split(r"\bto\b", hint, maxsplit=1, flags=re.IGNORECASE)[0].strip(" .")
            return hint if hint else None
        relaxed = re.search(r"\b(today|tomorrow|tonight|next\s+\w+)\b", text, flags=re.IGNORECASE)
        if relaxed is not None:
            return relaxed.group(1).strip(" .")
        return None

    @staticmethod
    def _trim_reminder_prefix(summary: str) -> str:
        trimmed = re.sub(
            r"^(?:at|on|by|before|after)\s+[^,.;]{1,40}\s+to\s+",
            "",
            summary.strip(),
            flags=re.IGNORECASE,
        ).strip()
        return trimmed if trimmed else summary.strip()

    def _handle_memory_intents(self, text: str) -> MemoryIntentResult | None:
        tab = self._extract_memory_tab(text)
        if tab is not None:
            self._active_memory_tab = tab
        if self._looks_like_reminder(text):
            summary = self._summarize_memory(text)
            due_hint = self._extract_due_hint(text)
            summary = self._trim_reminder_prefix(summary)
            self._memory_store.add(kind="reminder", summary=summary, source_text=text, due_hint=due_hint)
            self._active_memory_tab = "reminders"
            return MemoryIntentResult(kind="reminder", summary=summary, due_hint=due_hint)
        elif self._looks_like_note(text):
            summary = self._summarize_memory(text)
            self._memory_store.add(kind="note", summary=summary, source_text=text)
            self._active_memory_tab = "notes"
            return MemoryIntentResult(kind="note", summary=summary)
        return None

    @staticmethod
    def _format_memory_confirmation(intent: MemoryIntentResult) -> str:
        if intent.kind == "reminder":
            if intent.due_hint:
                return f"Added to reminders: {intent.summary} (due: {intent.due_hint})."
            return f"Added to reminders: {intent.summary}."
        return f"Added to notes: {intent.summary}."

    def _memory_context_for_llm(self) -> str:
        memory = self._memory_store.snapshot()
        notes = memory.get("notes", [])[:8]
        reminders = memory.get("reminders", [])[:8]

        note_lines = [
            f"- {item.get('summary', '').strip()}"
            for item in notes
            if str(item.get("summary", "")).strip()
        ]
        reminder_lines = []
        for item in reminders:
            summary = str(item.get("summary", "")).strip()
            if not summary:
                continue
            due_hint = str(item.get("due_hint", "")).strip()
            if due_hint:
                reminder_lines.append(f"- {summary} (due: {due_hint})")
            else:
                reminder_lines.append(f"- {summary}")

        notes_block = "\n".join(note_lines) if note_lines else "- none"
        reminders_block = "\n".join(reminder_lines) if reminder_lines else "- none"
        return (
            "Memory context (most recent first):\n"
            f"Notes:\n{notes_block}\n"
            f"Reminders:\n{reminders_block}"
        )

    def _execute_memory_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool_name = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
        try:
            if tool_name == "add_note":
                summary = str(arguments.get("summary", "")).strip()
                if not summary:
                    return {"ok": False, "error": "summary is required"}
                item = self._memory_store.add(kind="note", summary=summary, source_text="[tool] add_note")
                self._active_memory_tab = "notes"
                return {"ok": True, "item": item}

            if tool_name == "add_reminder":
                summary = str(arguments.get("summary", "")).strip()
                if not summary:
                    return {"ok": False, "error": "summary is required"}
                due_hint_raw = arguments.get("due_hint")
                due_hint = str(due_hint_raw).strip() if due_hint_raw else None
                item = self._memory_store.add(
                    kind="reminder",
                    summary=summary,
                    source_text="[tool] add_reminder",
                    due_hint=due_hint,
                )
                self._active_memory_tab = "reminders"
                return {"ok": True, "item": item}

            if tool_name == "list_notes":
                limit = int(arguments.get("limit", 24))
                query = str(arguments.get("query", "")).strip() or None
                items = self._memory_store.list_items(kind="note", limit=limit, query=query)
                return {"ok": True, "items": items, "count": len(items)}

            if tool_name == "list_reminders":
                limit = int(arguments.get("limit", 24))
                query = str(arguments.get("query", "")).strip() or None
                items = self._memory_store.list_items(kind="reminder", limit=limit, query=query)
                return {"ok": True, "items": items, "count": len(items)}

            if tool_name == "update_note":
                item_id = str(arguments.get("id", "")).strip()
                summary = str(arguments.get("summary", "")).strip()
                updated = self._memory_store.update_item(item_id=item_id, summary=summary)
                if updated is None:
                    return {"ok": False, "error": "note not found"}
                self._active_memory_tab = "notes"
                return {"ok": True, "item": updated}

            if tool_name == "update_reminder":
                item_id = str(arguments.get("id", "")).strip()
                summary_raw = arguments.get("summary")
                due_hint_raw = arguments.get("due_hint")
                summary = str(summary_raw).strip() if summary_raw is not None else None
                due_hint = str(due_hint_raw).strip() if due_hint_raw is not None else None
                updated = self._memory_store.update_item(item_id=item_id, summary=summary, due_hint=due_hint)
                if updated is None:
                    return {"ok": False, "error": "reminder not found"}
                self._active_memory_tab = "reminders"
                return {"ok": True, "item": updated}

            if tool_name == "delete_note":
                item_id = str(arguments.get("id", "")).strip()
                deleted = self._memory_store.delete_item(item_id)
                self._active_memory_tab = "notes"
                return {"ok": True, "deleted": deleted}

            if tool_name == "delete_reminder":
                item_id = str(arguments.get("id", "")).strip()
                deleted = self._memory_store.delete_item(item_id)
                self._active_memory_tab = "reminders"
                return {"ok": True, "deleted": deleted}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        return {"ok": False, "error": f"unknown tool: {tool_name}"}

    def _fallback_memory_tool_for_text(self, text: str) -> tuple[str, dict[str, Any]] | None:
        lowered = text.lower()

        if self._looks_like_reminder(text):
            summary = self._trim_reminder_prefix(self._summarize_memory(text))
            due_hint = self._extract_due_hint(text)
            if summary:
                args: dict[str, Any] = {"summary": summary}
                if due_hint:
                    args["due_hint"] = due_hint
                return "add_reminder", args

        if self._looks_like_note(text):
            summary = self._summarize_memory(text)
            if summary:
                return "add_note", {"summary": summary}

        if any(token in lowered for token in ("what notes", "list notes", "show notes", "saved notes", "summarize notes")):
            return "list_notes", {"limit": 24}

        if any(
            token in lowered
            for token in ("what reminders", "active reminders", "list reminders", "show reminders", "saved reminders")
        ):
            return "list_reminders", {"limit": 24}

        return None

    def set_active_memory_tab(self, tab: str) -> str:
        normalized = tab.strip().lower()
        if normalized not in {"notes", "reminders"}:
            raise ValueError("Tab must be notes or reminders.")
        self._active_memory_tab = normalized
        return self._active_memory_tab

    async def ensure_engine(self) -> VoicebotEngine:
        async with self._engine_lock:
            if self.engine is not None:
                return self.engine

            chat_client = LlamaCppChatClient(
                model=self.settings.llama_cpp_model,
                url=self.settings.llama_cpp_url,
                timeout_seconds=self.settings.http_timeout_seconds,
                disable_think_mode=self.settings.llama_cpp_disable_think,
                system_prompt=RAVEN_SYSTEM_PROMPT,
                context_provider=self._memory_context_for_llm,
                tool_executor=self._execute_memory_tool,
                tool_spec=RAVEN_MEMORY_TOOL_SPEC,
                tool_fallback_selector=self._fallback_memory_tool_for_text,
            )
            tts_provider = KokoroTTSProvider(
                lang_code=self.settings.kokoro_lang,
                voice=self.settings.kokoro_voice,
                repo_id=self.settings.kokoro_repo_id,
            )

            delegate = None
            if self.settings.audio_mode != "silent":
                try:
                    delegate = AudioPlayer(
                        sample_rate=self.settings.sample_rate,
                        queue_maxsize=self.settings.audio_queue_maxsize,
                    )
                except Exception as exc:  # noqa: BLE001
                    self.settings.audio_mode = "silent"
                    self.observer.record("warmup_failed", message=f"Audio init fallback: {exc}")
                    delegate = None

            if delegate is None and self.settings.audio_mode == "silent":
                self.observer.audio_mode = "host-bridge"
            else:
                self.observer.audio_mode = "live+host-bridge" if delegate is not None else "host-bridge"

            audio_sink = BufferedAudioPlayer(
                sample_rate=self.settings.sample_rate,
                delegate=delegate if delegate is not None else NullAudioPlayer(),
            )
            self._audio_sink = audio_sink

            self.engine = VoicebotEngine(
                chat_client=chat_client,
                tts_provider=tts_provider,
                audio_sink=audio_sink,
                chunker_factory=_chunker_factory(self.settings),
                observer=self.observer,
            )
            self.observer.langgraph_enabled = self.engine.orchestrator.using_langgraph
            return self.engine

    async def warmup(self) -> None:
        if self._warmed_up or not self.settings.warmup_enabled:
            return

        async with self._warmup_lock:
            if self._warmed_up or not self.settings.warmup_enabled:
                return
            engine = await self.ensure_engine()
            try:
                await engine.warmup()
            except Exception as exc:  # noqa: BLE001
                self.observer.record("warmup_failed", message=str(exc))
            self._warmed_up = True

    async def start_turn(self, user_text: str) -> dict[str, Any]:
        text = user_text.strip()
        if not text:
            raise ValueError("Prompt cannot be empty.")
        if self._turn_task is not None and not self._turn_task.done():
            raise RuntimeError("A turn is already running.")
        tab = self._extract_memory_tab(text)
        if tab is not None:
            self._active_memory_tab = tab
            self._messages.append(ConversationMessage(role="user", content=text))
            self._messages.append(ConversationMessage(role="assistant", content=f"Switched to {tab} tab."))
            return self.snapshot()

        await self.warmup()
        await self.ensure_engine()
        if self._audio_sink is not None:
            self._audio_sink.begin_capture()

        self._messages.append(ConversationMessage(role="user", content=text))
        self._messages.append(ConversationMessage(role="assistant", content=""))
        self._turn_task = asyncio.create_task(self._run_turn(text), name="voicebot_web_turn")
        return self.snapshot()

    async def _run_turn(self, user_text: str) -> None:
        async with self._turn_lock:
            assert self.engine is not None
            try:
                result = await self.engine.run_turn(user_text)
            except Exception as exc:  # noqa: BLE001
                self.observer.record("turn_failed", stage="runtime", message=str(exc))
                if self._audio_sink is not None:
                    self._audio_sink.clear_capture()
                if self._messages and self._messages[-1].role == "assistant":
                    self._messages[-1].content = f"Error: {exc}"
                return

            if self._audio_sink is not None:
                self._audio_sink.finalize_capture()
            if self._messages and self._messages[-1].role == "assistant":
                self._messages[-1].content = result.assistant_text

    async def close(self) -> None:
        task = self._turn_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        if self.engine is not None:
            await self.engine.close()

    def snapshot(self) -> dict[str, Any]:
        snapshot = self.observer.snapshot()
        messages = [
            {"role": message.role, "content": message.content}
            for message in self._messages
        ]
        if messages and messages[-1]["role"] == "assistant":
            live_text = snapshot["current_assistant_text"]
            if live_text:
                messages[-1]["content"] = live_text
        snapshot["messages"] = messages
        snapshot["busy"] = self._turn_task is not None and not self._turn_task.done()
        snapshot["latest_audio_available"] = self._audio_sink is not None and self._audio_sink.get_latest_wav() is not None
        snapshot["latest_audio_revision"] = self._audio_sink.get_revision() if self._audio_sink is not None else 0
        memory = self._memory_store.snapshot()
        snapshot["notes"] = memory["notes"]
        snapshot["reminders"] = memory["reminders"]
        snapshot["active_memory_tab"] = self._active_memory_tab
        return snapshot

    def get_latest_audio(self) -> bytes | None:
        if self._audio_sink is None:
            return None
        return self._audio_sink.get_latest_wav()
