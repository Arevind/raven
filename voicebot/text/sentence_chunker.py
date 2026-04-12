from __future__ import annotations

import re
from dataclasses import dataclass


SENTENCE_BOUNDARY_RE = re.compile(r'([.!?]+["\')\]]?\s+|\n+)')


@dataclass(slots=True)
class SentenceChunker:
    max_chars: int = 280
    min_chars: int = 80
    soft_chars: int = 0
    buffer: str = ""

    def feed(self, text: str) -> list[str]:
        self.buffer += text
        out: list[str] = []

        while self.buffer:
            sentence_endings = list(SENTENCE_BOUNDARY_RE.finditer(self.buffer))
            chosen_end = 0
            for match in sentence_endings:
                if match.end() <= self.max_chars:
                    chosen_end = match.end()
                else:
                    break

            if chosen_end > 0:
                if chosen_end < self.min_chars and len(self.buffer) <= self.max_chars:
                    break
                chunk = self.buffer[:chosen_end].strip()
                self.buffer = self.buffer[chosen_end:]
                if chunk:
                    out.append(chunk)
                continue

            if len(self.buffer) <= self.max_chars:
                break

            split_at = self.buffer.rfind(", ", 0, self.max_chars)
            if split_at <= 0:
                split_at = self.buffer.rfind("; ", 0, self.max_chars)
            if split_at <= 0:
                split_at = self.buffer.rfind(": ", 0, self.max_chars)
            if split_at <= 0:
                split_at = self.buffer.rfind(" ", 0, self.max_chars)
            if split_at <= 0:
                split_at = self.max_chars
            else:
                split_at += 1

            chunk = self.buffer[:split_at].strip()
            self.buffer = self.buffer[split_at:].lstrip()
            if chunk:
                out.append(chunk)

        if self.soft_chars > 0 and len(self.buffer) >= self.soft_chars and " " in self.buffer:
            split_at = self.buffer.rfind(" ")
            if split_at > 0:
                chunk = self.buffer[:split_at].strip()
                self.buffer = self.buffer[split_at:].lstrip()
                if chunk:
                    out.append(chunk)

        return out

    def flush(self) -> str | None:
        tail = self.buffer.strip()
        self.buffer = ""
        return tail if tail else None

