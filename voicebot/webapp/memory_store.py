from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock
from typing import Any
import uuid


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class MemoryItem:
    id: str
    kind: str
    summary: str
    source_text: str
    created_at: str
    due_hint: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "summary": self.summary,
            "source_text": self.source_text,
            "created_at": self.created_at,
            "due_hint": self.due_hint,
        }


class MemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._items: list[MemoryItem] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for raw_line in self.path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = str(data.get("kind", "")).lower()
            if kind not in {"note", "reminder"}:
                continue
            summary = str(data.get("summary", "")).strip()
            if not summary:
                continue
            self._items.append(
                MemoryItem(
                    id=str(data.get("id") or uuid.uuid4().hex),
                    kind=kind,
                    summary=summary,
                    source_text=str(data.get("source_text", "")).strip(),
                    created_at=str(data.get("created_at", _now_iso())),
                    due_hint=(str(data.get("due_hint")).strip() if data.get("due_hint") else None),
                )
            )

    def _append(self, item: MemoryItem) -> None:
        payload = json.dumps(item.as_dict(), ensure_ascii=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")

    def _persist_all(self) -> None:
        payload = "\n".join(json.dumps(item.as_dict(), ensure_ascii=True) for item in self._items)
        with self.path.open("w", encoding="utf-8") as handle:
            if payload:
                handle.write(payload)
                handle.write("\n")

    def add(self, kind: str, summary: str, source_text: str, due_hint: str | None = None) -> dict[str, Any]:
        cleaned_kind = kind.lower().strip()
        if cleaned_kind not in {"note", "reminder"}:
            raise ValueError("Memory kind must be note or reminder.")
        item = MemoryItem(
            id=uuid.uuid4().hex,
            kind=cleaned_kind,
            summary=summary.strip(),
            source_text=source_text.strip(),
            created_at=_now_iso(),
            due_hint=(due_hint.strip() if due_hint else None),
        )
        with self._lock:
            self._items.append(item)
            self._append(item)
        return item.as_dict()

    def list_items(self, kind: str, limit: int = 24, query: str | None = None) -> list[dict[str, Any]]:
        cleaned_kind = kind.lower().strip()
        if cleaned_kind not in {"note", "reminder"}:
            raise ValueError("Memory kind must be note or reminder.")
        query_text = (query or "").strip().lower()
        bounded_limit = max(1, min(int(limit), 100))
        with self._lock:
            matched = [item for item in self._items if item.kind == cleaned_kind]
            if query_text:
                matched = [item for item in matched if query_text in item.summary.lower()]
            rows = [item.as_dict() for item in matched][-bounded_limit:]
        return list(reversed(rows))

    def update_item(
        self,
        item_id: str,
        summary: str | None = None,
        due_hint: str | None = None,
    ) -> dict[str, Any] | None:
        target_id = item_id.strip()
        if not target_id:
            return None
        with self._lock:
            for idx, item in enumerate(self._items):
                if item.id != target_id:
                    continue
                if summary is not None:
                    cleaned_summary = summary.strip()
                    if cleaned_summary:
                        item.summary = cleaned_summary
                item.due_hint = due_hint.strip() if due_hint else None
                self._items[idx] = item
                self._persist_all()
                return item.as_dict()
        return None

    def delete_item(self, item_id: str) -> bool:
        target_id = item_id.strip()
        if not target_id:
            return False
        with self._lock:
            before = len(self._items)
            self._items = [item for item in self._items if item.id != target_id]
            deleted = len(self._items) < before
            if deleted:
                self._persist_all()
            return deleted

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        with self._lock:
            notes = [item.as_dict() for item in self._items if item.kind == "note"][-24:]
            reminders = [item.as_dict() for item in self._items if item.kind == "reminder"][-24:]
        return {
            "notes": list(reversed(notes)),
            "reminders": list(reversed(reminders)),
        }
