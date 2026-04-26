import unittest

from voicebot.config.settings import VoicebotSettings
from voicebot.webapp.service import DashboardService


class _FakeMemoryStore:
    def __init__(self) -> None:
        self.notes: list[dict] = []
        self.reminders: list[dict] = []
        self._next_id = 1

    def add(self, kind: str, summary: str, source_text: str, due_hint: str | None = None) -> dict:
        item = {
            "id": f"id-{self._next_id}",
            "kind": kind,
            "summary": summary,
            "source_text": source_text,
            "due_hint": due_hint,
            "created_at": "now",
        }
        self._next_id += 1
        if kind == "note":
            self.notes.insert(0, item)
        else:
            self.reminders.insert(0, item)
        return item

    def list_items(self, kind: str, limit: int = 24, query: str | None = None) -> list[dict]:
        rows = self.notes if kind == "note" else self.reminders
        out = list(rows)[:limit]
        if query:
            needle = query.lower()
            out = [item for item in out if needle in item["summary"].lower()]
        return out

    def update_item(self, item_id: str, summary: str | None = None, due_hint: str | None = None) -> dict | None:
        rows = self.notes + self.reminders
        for item in rows:
            if item["id"] != item_id:
                continue
            if summary is not None and summary.strip():
                item["summary"] = summary.strip()
            if due_hint is not None:
                item["due_hint"] = due_hint
            return item
        return None

    def delete_item(self, item_id: str) -> bool:
        before = len(self.notes)
        self.notes = [item for item in self.notes if item["id"] != item_id]
        if len(self.notes) < before:
            return True
        before = len(self.reminders)
        self.reminders = [item for item in self.reminders if item["id"] != item_id]
        return len(self.reminders) < before

    def snapshot(self) -> dict[str, list[dict]]:
        return {"notes": list(self.notes), "reminders": list(self.reminders)}


class MemoryCommandFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_memory_tool_add_and_list_reminder(self) -> None:
        service = DashboardService(VoicebotSettings())
        service._memory_store = _FakeMemoryStore()

        add_out = service._execute_memory_tool("add_reminder", {"summary": "buying groceries"})
        list_out = service._execute_memory_tool("list_reminders", {"limit": 5})

        self.assertTrue(add_out["ok"])
        self.assertEqual(add_out["item"]["summary"], "buying groceries")
        self.assertTrue(list_out["ok"])
        self.assertEqual(list_out["count"], 1)
        self.assertEqual(list_out["items"][0]["summary"], "buying groceries")
        self.assertEqual(service._active_memory_tab, "reminders")

    async def test_memory_context_contains_notes_and_reminders(self) -> None:
        service = DashboardService(VoicebotSettings())
        fake_store = _FakeMemoryStore()
        fake_store.add("note", "finish chapter 2", "note this")
        fake_store.add("reminder", "buy groceries", "add reminder", due_hint="tomorrow")
        service._memory_store = fake_store

        context = service._memory_context_for_llm()

        self.assertIn("Notes:", context)
        self.assertIn("- finish chapter 2", context)
        self.assertIn("Reminders:", context)
        self.assertIn("- buy groceries (due: tomorrow)", context)

    async def test_fallback_selector_maps_note_prompt_to_add_note_tool(self) -> None:
        service = DashboardService(VoicebotSettings())
        out = service._fallback_memory_tool_for_text("Create a new note telling I am a good kid")
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out[0], "add_note")
        self.assertEqual(out[1]["summary"], "I am a good kid")


if __name__ == "__main__":
    unittest.main()
