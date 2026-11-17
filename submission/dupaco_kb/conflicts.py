"""Find places where the website contradicts itself and decide which value is right.

Two families:
1. Rate-table facts: the same product + qualifiers (term, tier, model-year band, jumbo...)
   + metric shown with different values, on different pages and/or in different snapshots.
2. Text claims in prose that recur across pages (branch counts, counties served, employee
   counts, Forbes recognitions) and drift as old releases stay online.

Winner rules: newest snapshot first, then official rate/product pages, then the latest
"effective as of" date, then page authority. For prose claims: the most recent dated page.
"""
from __future__ import annotations

import re
from collections import defaultdict
from urllib.parse import urlparse

from .clean import norm_text
from .pages import LABELS, us_date
from .store import quals_match

METRIC_NAME = {"APY": "APY", "APR": "APR", "DIV": "dividend rate", "RATE": "rate"}


def same_fact(a, b) -> bool:
    if a.product != b.product or a.metric != b.metric or a.ambiguous or b.ambiguous:
        return False
    if a.product == "other":
        return norm_text(a.label) == norm_text(b.label)
    if not (a.named and b.named):
        # product unknown from the table itself: only compare the same table of the same page
        # (i.e. how one table changed between snapshots)
        return a.url == b.url and norm_text(a.label) == norm_text(b.label)
    return quals_match(a.quals, b.quals)


def _rank(f, latest_snap: int = 0):
    official = 1 if f.authority >= 0.85 else 0
    return (latest_snap, official, f.eff_date or "0000-00-00", f.authority)


def pick_winner(group, snap_of=None):
    if snap_of is None:
        return max(group, key=_rank)
    return max(group, key=lambda f: _rank(f, snap_of(f)))


def _short(url: str) -> str:
    p = urlparse(url)
    return (p.path or "/").rstrip("/") or "/"


def _term_variants(quals: dict) -> str:
    out = []
    for t in quals.get("terms", ()):
        n, unit = t.split()
        out.append(f"{n}-{unit}")
    return " ".join(out)


def _topic(f) -> str:
    label = re.sub(r"\s*\|\s*", ", ", f.label)
    extra = _term_variants(f.quals)
    prod = f.product if f.product != "other" else ""
    return re.sub(r"\s+", " ", f"{extra} {prod} {METRIC_NAME.get(f.metric, f.metric)} - {label}").strip()


# ------------------------------------------------------------------ prose claims
_ORD = "first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth"
CLAIMS = [
    ("number of branch locations", re.compile(r"\b(\d{1,3})\s+(?:full-service\s+)?branch(?:es| office locations| locations| offices)\b", re.I), None),
    ("number of counties served", re.compile(r"\b(\d{1,3})\s+counties\b", re.I), None),
    ("number of employees", re.compile(r"\b(?:nearly|more than|over|about|approximately)\s+([\d,]{2,6})\s+employees\b", re.I), None),
    ("Forbes recognitions (times ranked)", re.compile(rf"\b({_ORD})\s+(?:time|year)\b", re.I), re.compile(r"forbes", re.I)),
]


def prose_claims(store, sidx: int):
    found = defaultdict(list)    # claim -> [(value, version)]
    for v in store.versions:
        if sidx not in v.snaps:
            continue
        for name, rx, page_rx in CLAIMS:
            if page_rx and not page_rx.search(v.title + " " + v.url):
                continue
            vals = {m.group(1).lower().replace(",", "") for m in rx.finditer(v.text)}
            if len(vals) == 1:          # a page that itself lists several values is ambiguous
                found[name].append((vals.pop(), v))
    return found


_METRIC_ORDER = {"APY": 0, "APR": 0, "RATE": 1, "DIV": 2}


def _merge_metrics(items: list[dict]) -> list[dict]:
    """One table row usually yields two conflicts (APY + dividend rate, or rate + APR).
    They are the same contradiction, so report them once with all values."""
    by_row: dict = {}
    order = []
    for it in items:
        k = it.get("_row")
        if k is None:
            order.append(it)
            continue
        if k not in by_row:
            by_row[k] = []
            order.append(k)
        by_row[k].append(it)
    out = []
    for k in order:
        if isinstance(k, dict):
            out.append(k)
            continue
        group = sorted(by_row[k], key=lambda x: _METRIC_ORDER.get(x["_metric"], 3))
        main = dict(group[0])
        for extra in group[1:]:
            main["values"] = main["values"] + [v for v in extra["values"] if v not in main["values"]]
            main["sources"] = main["sources"] + extra["sources"]
            main["resolution"] += " / " + extra["resolution"]
            main["reason"] += " | " + extra["reason"]
            main["_score"] = max(main["_score"], extra["_score"])
        out.append(main)
    return out


# ------------------------------------------------------------------ build
def build_conflicts(store, max_items: int = 400) -> list[dict]:
    if not store.snaps:
        return []
    latest = store.latest_index()
    snap_ids = [s for s, _ in store.snaps]

    def last_snap(f):
        return max(store.versions[f.vid].snaps)

    # 1. table facts: bucket cheaply, then cluster with the full matcher
    buckets = defaultdict(list)
    for f in store.facts:
        q = f.quals
        key = (f.product, f.metric, q["terms"], q["amount"], q["band"], q["strict"],
               norm_text(f.label) if f.product == "other" else "")
        buckets[key].append(f)
    out = []
    for facts in buckets.values():
        clusters: list[list] = []
        for f in facts:
            for c in clusters:
                if same_fact(c[0], f):
                    c.append(f)
                    break
            else:
                clusters.append([f])
        for c in clusters:
            distinct = {f.value for f in c}
            if len(distinct) < 2:
                continue
            if not any(f.authority >= 0.85 for f in c):
                continue
            w = pick_winner(c, last_snap)
            if last_snap(w) != latest:
                continue          # fact no longer on the site; not an open contradiction
            by_val = {}
            for f in sorted(c, key=lambda f: _rank(f, last_snap(f))):
                by_val[f.value] = f          # keeps the best-ranked page per value
            losers = [f for val, f in by_val.items() if val != w.value]
            reasons = []
            for lf in losers:
                why = []
                if last_snap(lf) < last_snap(w):
                    why.append(f"older snapshot ({store.snaps[last_snap(lf)][1]})")
                if lf.authority < w.authority:
                    why.append(LABELS.get(lf.ptype, lf.ptype))
                if (lf.eff_date or "") < (w.eff_date or ""):
                    why.append(f"effective {us_date(lf.eff_date) or 'undated'}")
                reasons.append(f"{lf.value_str} on {_short(lf.url)} ({', '.join(why) or 'superseded'})")
            reason = (f"{w.value_str} is current: {LABELS.get(w.ptype, w.ptype)} {_short(w.url)} in the "
                      f"latest snapshot" + (f", effective {us_date(w.eff_date)}" if w.eff_date else "") +
                      ". Superseded: " + "; ".join(reasons))
            values = [f.value_str for f in losers] + [w.value_str]
            sources = [{"url": f.url, "snapshot_id": snap_ids[last_snap(f)],
                        "quote": f"{f.label}: {f.value_str}"} for f in losers + [w]]
            score = len(c) + (5 if w.ptype == "rates" else 0)
            out.append({"topic": _topic(w), "values": values, "sources": sources,
                        "resolution": f"{w.value_str} {METRIC_NAME.get(w.metric, w.metric)}",
                        "reason": reason, "_score": score, "_row": (w.product, repr(sorted(w.quals.items()))),
                        "_metric": w.metric})
    out = _merge_metrics(out)

    # 2. prose claims (latest snapshot), most recent dated page wins
    for name, items in prose_claims(store, latest).items():
        vals = {val for val, _ in items}
        if len(vals) < 2:
            continue
        def recency(item):
            val, v = item
            # dated statements (news, articles) beat undated pages, newest first
            return (1 if v.pub_date else 0, v.pub_date or "", v.authority)
        best_val, best_v = max(items, key=recency)
        per_val = {}
        for val, v in sorted(items, key=recency):
            per_val[val] = v
        values = [val for val in per_val if val != best_val] + [best_val]
        sources = [{"url": v.url, "snapshot_id": snap_ids[latest], "quote": f"{name}: {val}"}
                   for val, v in per_val.items()]
        out.append({"topic": name, "values": values, "sources": sources, "resolution": best_val,
                    "reason": f"most recent page ({_short(best_v.url)}, {best_v.pub_date or best_v.fetched}) says "
                              f"{best_val}; older pages still say {', '.join(values[:-1])}", "_score": 50})

    out.sort(key=lambda x: -x["_score"])
    return out[:max_items]
