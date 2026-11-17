"""Hackathon submission: a knowledge system that knows what's true *now*.

build() returns an object with ingest(snapshot), answer(question, as_of=None), conflicts().
All the work lives in the dupaco_kb package next to this file; see NOTES.md.
Requires ANTHROPIC_API_KEY (Claude writes the member-facing answers).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Use the harness' own dataclasses when we can find them (exact contract); otherwise fall back
# to identical local copies -- the scorer accepts any object with the same fields.
try:
    _h = next(d / "harness" for d in [_HERE, *_HERE.parents] if (d / "harness" / "contract.py").exists())
    if str(_h) not in sys.path:
        sys.path.insert(0, str(_h))
    from contract import Answer, Conflict, Source  # noqa: E402
except Exception:  # pragma: no cover - only when the harness folder is elsewhere
    try:
        from contract import Answer, Conflict, Source  # type: ignore  # noqa: E402
    except Exception:
        @dataclass
        class Source:  # type: ignore[no-redef]
            url: str
            snapshot_id: Optional[str] = None
            quote: Optional[str] = None

        @dataclass
        class Answer:  # type: ignore[no-redef]
            text: str
            sources: list = field(default_factory=list)
            confident: bool = True
            conflict_note: Optional[str] = None

        @dataclass
        class Conflict:  # type: ignore[no-redef]
            topic: str
            values: list
            sources: list
            resolution: Optional[str] = None
            reason: Optional[str] = None

from dupaco_kb.engine import Engine  # noqa: E402


class KnowledgeSystem:
    def __init__(self):
        self.engine = Engine()

    def ingest(self, snapshot) -> None:
        self.engine.ingest(snapshot)

    def answer(self, question: str, as_of: Optional[str] = None) -> Answer:
        r = self.engine.answer(question, as_of)
        return Answer(text=r["text"],
                      sources=[Source(url=s["url"], snapshot_id=s.get("snapshot_id"), quote=s.get("quote"))
                               for s in r["sources"]],
                      confident=r["confident"], conflict_note=r.get("conflict_note"))

    def conflicts(self) -> list:
        return [Conflict(topic=c["topic"], values=list(c["values"]),
                         sources=[Source(url=s["url"], snapshot_id=s.get("snapshot_id"), quote=s.get("quote"))
                                  for s in c["sources"]],
                         resolution=c.get("resolution"), reason=c.get("reason"))
                for c in self.engine.conflicts()]


def build() -> KnowledgeSystem:
    return KnowledgeSystem()
