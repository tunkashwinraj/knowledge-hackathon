"""Deterministic checks applied to every LLM answer before it is returned.

* refusal   : an "I don't know" must have confident=False and contain no digits at all.
* grounding : every number in the answer must appear in the evidence (or the question).
* urls      : every URL in the answer must exist somewhere on the crawled site.
* behaviour : no questions back to the member, no requests for ZIP/member number,
              no "according to the provided facts" style leakage.
* stale     : the answer must not quote a value that the evidence itself shows to be
              superseded (same product/tier/metric, older or less authoritative page).
"""
from __future__ import annotations

import re

from .clean import numbers, percents

REFUSAL_TEXT = ("I'm sorry, but that isn't listed on our website, so I can't confirm it. "
                "Please contact the credit union directly and a member of our team can help.")

_BEHAVIOUR = re.compile(
    r"(which (one|term|product|account|certificate|branch|option|card|loan)\b[^.?!]*\?|would you like (me )?to know"
    r"|could you (please )?(tell|provide|share|specify|let)|please (provide|share|specify|tell me|let me know (which|your))"
    r"|\bzip ?code\b|\bmember number\b|in these facts|\bthe (provided )?(excerpts?|context)\b"
    r"|provided (excerpts|information|context|facts|sources)|these (excerpts|facts|documents)|checked facts"
    r"|based on the (excerpts|information|context|documents)|i'?m not asking again)", re.I)
_URL = re.compile(r"(?:https?://)?(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|org|net|test|gov)(?:/[^\s)\]\"'<>,]*)?", re.I)


def _vals(nums):
    return {v for _, v in nums}


def ungrounded_numbers(answer: str, evidence_text: str, question: str) -> list[float]:
    allowed = _vals(numbers(evidence_text)) | _vals(numbers(question))
    return sorted(v for v in _vals(numbers(answer)) if v not in allowed)


def bad_urls(answer: str, known_urls: set[str], evidence_text: str) -> list[str]:
    out = []
    ev = evidence_text.lower()
    known = {u.lower().replace("https://", "").replace("http://", "").replace("www.", "").rstrip("/")
             for u in known_urls}
    for m in _URL.finditer(answer):
        u = m.group(0).rstrip(".;:!?").lower()
        bare = u.replace("https://", "").replace("http://", "").replace("www.", "").rstrip("/")
        if bare in known or bare in ev or any(k.startswith(bare) for k in known):
            continue
        out.append(m.group(0))
    return out


def behaviour_problems(answer: str) -> list[str]:
    return [m.group(0) for m in _BEHAVIOUR.finditer(answer)]


def stale_values(answer: str, evidence_facts: list) -> list[tuple[str, str, str]]:
    """Facts in the evidence that share product+qualifiers+metric but disagree. Returns
    (stale_value, current_value, label) for stale values that the answer quotes without
    also quoting the current one."""
    from .conflicts import pick_winner, same_fact   # local import to avoid a cycle
    ans = set(round(x, 4) for x in percents(answer))
    if not ans:
        return []
    groups: list[list] = []
    for f in evidence_facts:
        for g in groups:
            if same_fact(g[0], f):
                g.append(f)
                break
        else:
            groups.append([f])
    winners_everywhere = set()
    problems = []
    group_wins = []
    for g in groups:
        w = pick_winner(g)
        group_wins.append((g, w))
        winners_everywhere.update(w.value)
    for g, w in group_wins:
        losers = {v for f in g for v in f.value if f.value != w.value}
        for v in losers:
            if v in ans and v not in winners_everywhere and not (set(w.value) & ans):
                problems.append((f"{v:g}%", w.value_str, w.label))
    return problems


def strip_sentences_with(answer: str, bad_vals: list[float]) -> str:
    sents = re.split(r"(?<=[.!?])\s+", answer)
    keep = [s for s in sents if not (_vals(numbers(s)) & set(bad_vals))]
    return " ".join(keep).strip()


def is_refusal_text(answer: str) -> bool:
    return bool(re.search(r"isn't listed|is not listed|not (published|available|listed)|i (do not|don't|can't|cannot) "
                          r"(know|find|confirm)|please contact|no information", answer, re.I))
