"""Which snapshot should answer a question?

* as_of given  -> the newest snapshot taken on or before that date.
* otherwise, a past date written in the question ("as of May 2026", "back in June 2026",
  "on 2026-05-20") selects the snapshot that was live at the end of that period.
* otherwise (or for "today", future dates) -> the latest snapshot.
"""
from __future__ import annotations

import calendar
import re
from datetime import date
from typing import Optional

from .pages import MONTHS

_ISO = re.compile(r"\b(20\d\d)-(\d{1,2})-(\d{1,2})\b")
_MDY = re.compile(r"\b(\d{1,2})/(\d{1,2})/(20\d\d)\b")
_MONTH_YEAR = re.compile(r"\b(january|february|march|april|may|june|july|august|september|october|"
                         r"november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\.?"
                         r"(?:\s+(\d{1,2})(?:st|nd|rd|th)?,?)?\s+(20\d\d)\b", re.I)
_PAST_CUE = re.compile(r"\b(as of|as at|back in|in|on|during|was|were|did|used to|at the time|previously|"
                       r"earlier|before)\b", re.I)


def _to_date(s: str) -> Optional[date]:
    try:
        y, m, d = (int(x) for x in s[:10].split("-"))
        return date(y, m, d)
    except Exception:
        return None


def date_in_question(q: str) -> Optional[date]:
    """A date the question is anchored to, if it states one (end of month for 'May 2026')."""
    m = _ISO.search(q)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = _MDY.search(q)
    if m:
        return date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
    m = _MONTH_YEAR.search(q)
    if m:
        key = m.group(1).lower()
        mon = MONTHS.get(key) or next(i for k, i in MONTHS.items() if k.startswith(key[:3]))
        y = int(m.group(3))
        d = int(m.group(2)) if m.group(2) else calendar.monthrange(y, mon)[1]
        return date(y, mon, d)
    return None


def resolve(snaps: list[tuple[str, str]], as_of: Optional[str], question: str):
    """Return (snapshot_index, effective_date_for_answer, how)."""
    dates = [_to_date(t) for _, t in snaps]
    latest = len(snaps) - 1
    target = _to_date(as_of) if as_of else None
    how = "as_of" if target else "latest"
    if target is None:
        qd = date_in_question(question)
        if qd and dates and qd < dates[-1] and _PAST_CUE.search(question):
            target, how = qd, "question_date"
        elif qd and dates and qd >= dates[-1]:
            return latest, qd, "question_today"
    if target is None:
        return latest, dates[-1] if dates else None, how
    idx = 0
    for i, d in enumerate(dates):
        if d and d <= target:
            idx = i
    return idx, target, how
