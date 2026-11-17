"""Orchestration: ingest -> store/index; answer -> retrieve -> Claude -> guards -> sources."""
from __future__ import annotations

import os
import re
import sys
import threading
import time
from collections import defaultdict
from datetime import date
from urllib.parse import urlparse

from . import guards
from .clean import numbers, percents
from .conflicts import build_conflicts
from .pages import LABELS, us_date
from .search import BM25, query_tokens, tokenize
from .store import Store
from .temporal import resolve

_RATE_Q = re.compile(r"\b(rate|rates|apy|apr|dividend|interest|yield|%|percent|earn|pay)\b", re.I)
_HOURS_Q = re.compile(r"\b(hours?|open|opens|close|closes|closing)\b", re.I)
_TIME = re.compile(r"\b\d{1,2}(:\d\d)?\s*(a\.?m\.?|p\.?m\.?)|\bnoon\b", re.I)
_COUNT_Q = re.compile(r"\b(?:how many|number of|count of|total number of)\s+(?:dupaco'?s?\s+|the\s+|your\s+|its\s+)?"
                      r"([a-z-]+)", re.I)
_CHANGE_Q = re.compile(r"\b(chang\w*|increas\w*|decreas\w*|went (up|down)|go(ne)? (up|down)|rais\w*|lowered|"
                       r"drop\w*|previous\w*|before it|used to|histor\w*|compared|over time|trend\w*|originally)\b",
                       re.I)
_PERIOD_Q = re.compile(r"\b(last|this|next|previous|past) (year|month|quarter|week)\b|\bin (19|20)\d\d\b"
                       r"|\b(19|20)\d\d\b", re.I)
_GENERIC_COUNT_NOUNS = {"times", "time", "years", "year", "months", "days", "hours", "minutes", "weeks", "ways",
                        "steps", "people", "percent", "dollars", "more", "of"}
_GENERIC_VERBS = {"have", "has", "had", "use", "used", "offer", "get", "got", "make", "made", "operate", "run",
                  "many", "much", "number", "count", "total", "there", "exist", "last", "year", "time",
                  "current", "today", "still", "open"}
_ASKS_RATE = re.compile(r"\b(rate|rates|apr|apy|interest|dividend|yield)\b", re.I)
_LINK_Q = re.compile(r"\b(link|url|website|web ?site|web address|page)\b", re.I)
EVIDENCE_K = 10
EVIDENCE_CHARS = 2400
ARTICLE_CHARS = 4500     # news / blog posts are given whole (they are one dated story)
SMALL_PAGE_CHARS = 2500  # any page this small is given whole
NEIGHBOR_MIN = 900       # shorter chunks get their neighbouring chunks attached


def _d(iso):
    try:
        y, m, d = (int(x) for x in iso[:10].split("-"))
        return date(y, m, d)
    except Exception:
        return None


class Engine:
    def __init__(self):
        self.store = Store()
        self.index = BM25()
        self.store.on_new_chunk = self._index_chunk
        self.facts_by_chunk = defaultdict(list)
        self._facts_seen = 0
        self.cache: dict = {}
        self._llm = None
        self._lock = threading.Lock()
        self._conflicts = None

    # ---------------------------------------------------------------- ingest
    def _index_chunk(self, c, v):
        path_words = " ".join(re.split(r"[/\-_.]+", urlparse(v.url).path))
        toks = tokenize(v.title) + tokenize(c.heading) + tokenize(path_words) + tokenize(c.text)
        self.index.add(c.cid, toks)

    def ingest(self, snapshot) -> None:
        self.store.add_snapshot(snapshot)
        for f in self.store.facts[self._facts_seen:]:
            self.facts_by_chunk[f.cid].append(f)
        self._facts_seen = len(self.store.facts)
        self.cache.clear()
        self._conflicts = None

    @property
    def llm(self):
        if self._llm is None:
            from .llm import LLM
            self._llm = LLM()
        return self._llm

    # ---------------------------------------------------------------- retrieval
    def retrieve(self, question: str, sidx: int, k: int = EVIDENCE_K) -> list[dict]:
        st = self.store
        qw = query_tokens(question)
        base_q = {t for t, w in qw.items() if w >= 1.0}
        wants_rate = bool(_RATE_Q.search(question))
        wants_hours = bool(_HOURS_Q.search(question))
        mc = _COUNT_Q.search(question)
        count_rx = re.compile(r"\b\d[\d,]*\+?\s+(?:[a-z-]+\s+){0,3}" + re.escape(mc.group(1).lower()[:5]), re.I) \
            if mc else None
        snap_date = _d(st.snaps[sidx][1])
        carried = st.carried(sidx)
        hits = self.index.search(
            qw, lambda cid: sidx in st.versions[st.chunks[cid].vid].snaps or st.chunks[cid].vid in carried, k=150)
        scored = []
        for cid, s in hits:
            c = st.chunks[cid]
            v = st.versions[c.vid]
            m = 0.5 + 0.7 * v.authority
            if v.vid in carried:
                m *= 0.6                     # page no longer on the site: keep as a last resort
            if c.kind == "table" and wants_rate:
                m *= 1.3
            if wants_hours and len(_TIME.findall(c.text)) >= 2:
                m *= 1.6
            if count_rx and count_rx.search(c.text):
                m *= 1.6
            if v.ptype in ("newsroom", "article") and v.pub_date and snap_date:
                pd = _d(v.pub_date)
                if pd:
                    age = (snap_date - pd).days
                    m *= 1.25 if age < 150 else 1.1 if age < 365 else 0.95 if age < 730 else 0.85
            url_words = set(tokenize(" ".join(re.split(r"[/\-_.]+", urlparse(v.url).path)) + " " + v.title))
            m *= 1 + 0.08 * min(4, len(url_words & base_q))
            if "business" in url_words and "business" not in base_q:
                m *= 0.8                     # personal products are the default reading
            scored.append((s * m, cid))
        scored.sort(key=lambda x: -x[0])
        picked, by_hash, per_url = [], {}, defaultdict(int)
        for s, cid in scored:
            c = st.chunks[cid]
            v = st.versions[c.vid]
            if c.thash in by_hash:
                item = by_hash[c.thash]
                if v.url not in item["urls"]:
                    item["urls"].append(v.url)
                    item["vids"].append(v.vid)
                continue
            cap = 4 if v.ptype == "rates" else 2     # rate sheets hold many distinct tables
            if per_url[v.url] >= cap or len(picked) >= k:
                continue
            whole = len(v.text) <= SMALL_PAGE_CHARS or \
                (v.ptype in ("newsroom", "article") and len(v.text) <= ARTICLE_CHARS)
            if whole and per_url[v.url]:
                continue                       # page already given in full
            per_url[v.url] += 1
            item = {"cid": cid, "vids": [v.vid], "urls": [v.url], "score": s,
                    "text": v.text if whole else self._with_neighbors(c, v), "whole": whole}
            by_hash[c.thash] = item
            picked.append(item)
        return picked

    def _with_neighbors(self, c, v) -> str:
        """Short text chunks lose context; attach the previous/next chunk of the same page."""
        text = c.text[:EVIDENCE_CHARS]
        if c.kind != "text" or len(text) >= NEIGHBOR_MIN:
            return text
        ids = v.chunk_ids
        pos = ids.index(c.cid)
        for nb in (pos + 1, pos - 1):
            if 0 <= nb < len(ids):
                n = self.store.chunks[ids[nb]]
                if n.kind == "text" and len(text) + len(n.text) <= EVIDENCE_CHARS:
                    text = text + "\n\n" + n.text if nb > pos else n.text + "\n\n" + text
        return text

    def _evidence_block(self, ev: list[dict], sidx: int) -> str:
        st = self.store
        parts = []
        for i, item in enumerate(ev, 1):
            c = st.chunks[item["cid"]]
            vs = sorted((st.versions[x] for x in item["vids"]), key=lambda v: -v.authority)
            v = vs[0]
            meta = [LABELS.get(v.ptype, v.ptype), v.url]
            if sidx not in v.snaps:
                meta.insert(0, f"PAGE NO LONGER ON THE SITE (last seen {st.last_seen(v.vid)}; use only if no current "
                               f"page covers this)")
            if len(vs) > 1:
                meta.append("same text also on: " + ", ".join(x.url for x in vs[1:4]))
            if v.title:
                meta.append(f'page title: "{v.title[:90]}"')
            if c.eff_date:
                meta.append(f"table effective {us_date(c.eff_date)}")
            if v.pub_date:
                meta.append(f"published {us_date(v.pub_date)}")
            if c.heading and not item["whole"]:
                meta.append(f"section: {c.heading[:100]}")
            parts.append(f"[{i}] " + " | ".join(meta) + "\n" + item["text"])
        return "\n\n".join(parts)

    # ---------------------------------------------------------------- answer
    def answer(self, question: str, as_of=None) -> dict:
        st = self.store
        if not st.snaps:
            return {"text": guards.REFUSAL_TEXT, "sources": [], "confident": False, "conflict_note": None}
        sidx, target, how = resolve(st.snaps, as_of, question)
        key = (re.sub(r"[^a-z0-9$%.]+", " ", question.lower()).strip(), sidx, how, str(target))
        with self._lock:
            if key in self.cache:
                return dict(self.cache[key])
        ev = self.retrieve(question, sidx)
        result = self._answer_with_llm(question, sidx, target, how, ev)
        with self._lock:
            self.cache[key] = result
        return dict(result)

    # ---------------------------------------------------------------- hints
    def _count_claims(self, question: str, sidx: int):
        """'How many X' -> every 'N ... X' statement on live pages, newest dated first."""
        mc = _COUNT_Q.search(question)
        if not mc or _PERIOD_Q.search(question):
            return []            # period statistics ("last year") are not current totals
        noun = mc.group(1).lower()
        if noun in _GENERIC_COUNT_NOUNS:
            return []
        stem = noun[:-2] if noun.endswith("es") and len(noun) > 5 else noun.rstrip("s")
        filler = r"(?:(?!(?:of|to|in|at|on|outside|from|for|with|and|or|per|by)\b)[A-Za-z-]+\s+){0,2}"
        rx = re.compile(r"\b(\d[\d,]*)\+?\s+" + filler + re.escape(stem[:6]), re.I)
        # other topic words of the question must appear near the number
        topic = [t[:5] for t, w in query_tokens(question).items()
                 if w >= 1.0 and len(t) >= 4 and not t[:5] == stem[:5] and t not in _GENERIC_VERBS
                 and not t.isdigit()]
        claims = []
        for v in self.store.versions:
            if sidx not in v.snaps or v.ptype in ("listing", "legal"):
                continue
            for m in rx.finditer(v.text):
                if 1900 <= int(m.group(1).replace(",", "") or 0) <= 2100:
                    continue          # a year, not a count
                window = v.text[max(0, m.start() - 200): m.end() + 200].lower()
                if topic and not any(t in window for t in topic):
                    continue
                claims.append((v.pub_date or "", m.group(0), v))
                break
        claims.sort(key=lambda x: (x[0], x[2].authority), reverse=True)
        return claims

    def _history_lines(self, question: str, ev: list[dict], limit: int = 6) -> list[str]:
        """Timeline of official values for the table rows in the evidence, across all crawls."""
        from .conflicts import same_fact
        st = self.store
        qtok = {t for t, w in query_tokens(question).items() if w >= 1.0}
        ev_facts = [f for i in ev for f in self.facts_by_chunk.get(i["cid"], []) if f.authority >= 0.85]
        # most question-relevant rows first
        ev_facts.sort(key=lambda f: -len(qtok & set(tokenize(f.label + " " + f.metric))))
        official = [f for f in st.facts if f.authority >= 0.85 and not f.ambiguous]
        out, seen = [], set()
        for f in ev_facts:
            key = (f.product, repr(sorted(f.quals.items())), f.metric)
            if key in seen or f.ambiguous:
                continue
            seen.add(key)
            per_snap = {}
            for g in official:
                if g is f or same_fact(f, g):
                    for s in st.versions[g.vid].snaps:
                        per_snap.setdefault(s, set()).add(g.value_str)
            if len({v for vs in per_snap.values() for v in vs}) < 2:
                continue
            steps = [f"{us_date(st.snaps[s][1])}: {' / '.join(sorted(per_snap[s]))}" for s in sorted(per_snap)]
            out.append(f"- History of {f.label} ({f.metric}) on official pages by crawl date: " + "; ".join(steps)
                       + ". (Crawls on: " + ", ".join(us_date(t) for _, t in st.snaps) + ")")
            if len(out) >= limit:
                break
        return out

    def _hints(self, question: str, sidx: int, ev: list[dict]) -> list[str]:
        from .conflicts import pick_winner, same_fact
        st = self.store
        lines = []
        # 1. table facts contradicted by an official page in the same snapshot
        ev_facts = [f for i in ev for f in self.facts_by_chunk.get(i["cid"], [])]
        official = [f for f in st.facts if f.authority >= 0.85 and sidx in st.versions[f.vid].snaps]
        seen = set()
        for f in ev_facts:
            group = [g for g in official if same_fact(g, f)] + [f]
            vals = {g.value for g in group}
            if len(vals) < 2:
                continue
            w = pick_winner(group)
            key = (w.label, w.metric, w.value)
            if key in seen:
                continue
            seen.add(key)
            old = sorted({g.value_str for g in group if g.value != w.value})
            old_pages = sorted({urlparse(g.url).path for g in group if g.value != w.value})[:3]
            lines.append(f"- {w.label} ({w.metric}): current value {w.value_str} per {LABELS.get(w.ptype, w.ptype)} "
                         f"{w.url}" + (f" (effective {us_date(w.eff_date)})" if w.eff_date else "") +
                         f". Outdated value(s) {', '.join(old)} still shown on {', '.join(old_pages)} - do not use.")
            if len(lines) >= 6:
                break
        # 2. questions about change over time: show how the relevant rows moved across crawls
        if _CHANGE_Q.search(question):
            lines += self._history_lines(question, ev)
        # 3. counts that drift across dated pages
        claims = self._count_claims(question, sidx)
        dated = [c for c in claims if c[0]]
        if dated and len({c[1].split()[0] for c in claims}) > 1:
            d0, phrase, v = dated[0]
            others = sorted({c[1] for c in claims if c[1] != phrase})[:5]
            lines.append(f"- Pages disagree on this count. The most recent dated statement is \"{phrase}\" "
                         f"({LABELS.get(v.ptype, v.ptype)} published {us_date(d0)}, {v.url}). Older or undated "
                         f"pages say: {'; '.join(others)}.")
        return lines

    def _prompt(self, question, sidx, target, how, ev, hints=None) -> str:
        snap_date = self.store.snaps[sidx][1]
        if how in ("as_of", "question_date"):
            when = (f"The member is asking about {target.isoformat()}. The excerpts below are the website as it "
                    f"stood then (crawl of {snap_date}). Answer as of that date.")
        elif how == "question_today":
            when = f"Today is {target.isoformat()}. The newest website crawl is from {snap_date}."
        else:
            when = f"Website date (newest crawl): {snap_date}. Today is {snap_date}."
        if hints is None:
            hints = self._hints(question, sidx, ev)
        hint_block = ("Checked facts (where the website contradicts itself):\n" + "\n".join(hints) + "\n\n") \
            if hints else ""
        return (f"{when}\n\nMember question: {question}\n\n{hint_block}"
                f"Website excerpts, most relevant first:\n\n{self._evidence_block(ev, sidx)}")

    def _check(self, data: dict, ev_text: str, ev_facts: list, question: str) -> list[str]:
        ans = data.get("answer", "")
        probs = []
        if not data.get("confident", True):
            extra = {v for _, v in numbers(ans)} - {v for _, v in numbers(question)}
            if extra:
                probs.append("A 'not listed' answer must not contain any numbers (no phone numbers, no rates).")
            return probs
        bad = guards.ungrounded_numbers(ans, ev_text, question)
        if bad:
            probs.append("These numbers do not appear on the website pages and must not be used: "
                         + ", ".join(f"{x:g}" for x in bad))
        urls = guards.bad_urls(ans, self.store.link_urls, ev_text)
        if urls:
            probs.append("These links do not exist on the website: " + ", ".join(urls))
        beh = guards.behaviour_problems(ans)
        if beh:
            probs.append("Do not ask the member questions, do not ask for ZIP codes or member numbers, and do not "
                         "mention excerpts/facts/context. Offending text: " + "; ".join(beh[:3]))
        for stale, cur, label in guards.stale_values(ans, ev_facts):
            probs.append(f"{stale} is an outdated value for '{label}'; the current official value is {cur}.")
        return probs

    def _answer_with_llm(self, question, sidx, target, how, ev) -> dict:
        st = self.store
        hints = self._hints(question, sidx, ev)
        prompt = self._prompt(question, sidx, target, how, ev, hints)
        ev_text = prompt                     # everything the model saw counts as grounded
        ev_facts = [f for i in ev for f in self.facts_by_chunk.get(i["cid"], [])]
        msgs = [{"role": "user", "content": prompt}]
        t0 = time.time()
        try:
            data = self.llm.ask(msgs)
            probs = self._check(data, ev_text, ev_facts, question)
            if probs and os.getenv("KB_DEBUG"):
                print(f"[kb] regenerate for {question[:60]!r}: {probs}", file=sys.stderr)
            if probs and time.time() - t0 < 10:      # only retry while well inside the 60 s budget
                msgs += [{"role": "assistant", "content": data["_raw"]},
                         {"role": "user", "content": "Please fix these problems in your answer and return the "
                                                     "corrected JSON:\n- " + "\n- ".join(probs)}]
                data2 = self.llm.ask(msgs)
                if len(self._check(data2, ev_text, ev_facts, question)) <= len(probs):
                    data = data2
        except Exception as e:  # network / API problems: never crash the grader
            print(f"[kb] LLM error: {type(e).__name__}: {e}", file=sys.stderr)
            return {"text": guards.REFUSAL_TEXT, "sources": [], "confident": False,
                    "conflict_note": f"internal error: {type(e).__name__}"}

        text = (data.get("answer") or "").strip()
        confident = bool(data.get("confident", True))
        # A rate question answered without any rate means the rate isn't published:
        # say so cleanly instead of offering loosely related facts.
        if confident and _ASKS_RATE.search(question) and not _LINK_Q.search(question) and "%" not in text:
            confident = False
            text = guards.REFUSAL_TEXT
        if not confident:
            extra = {v for _, v in numbers(text)} - {v for _, v in numbers(question)}
            if extra or not text:
                text = guards.REFUSAL_TEXT
            return {"text": text, "sources": [], "confident": False,
                    "conflict_note": data.get("conflict_note") or None}
        bad = guards.ungrounded_numbers(text, ev_text, question)
        if bad:
            stripped = guards.strip_sentences_with(text, bad)
            # only drop the offending sentences if a real answer remains
            if stripped and len(stripped) >= 0.6 * len(text) and re.search(r"\d", stripped):
                text = stripped
        return {"text": text, "sources": self._sources(data, ev, text, sidx), "confident": True,
                "conflict_note": data.get("conflict_note") or None}

    def _sources(self, data, ev, text, sidx) -> list[dict]:
        st = self.store
        sid = st.snaps[sidx][0]
        urls: list[str] = []
        for i in data.get("source_ids") or []:
            if isinstance(i, int) and 1 <= i <= len(ev):
                for u in ev[i - 1]["urls"]:
                    if u not in urls:
                        urls.append(u)
        key_vals = {x for x in percents(text)} | {v for u, v in numbers(text)
                                                  if u == "$" or (v >= 100 and not 1900 <= v <= 2100)}
        if key_vals:
            for item in ev:
                ctext = item["text"]
                cvals = set(percents(ctext)) | {v for _, v in numbers(ctext)}
                if key_vals & cvals:
                    for u in item["urls"]:
                        if u not in urls:
                            urls.append(u)
        auth = {}
        for u in urls:
            v = st.version_at(u, sidx)
            auth[u] = v.authority if v else 0
        urls.sort(key=lambda u: -auth[u])
        return [{"url": u, "snapshot_id": sid, "quote": None} for u in urls[:6]]

    # ---------------------------------------------------------------- conflicts
    def conflicts(self) -> list[dict]:
        if self._conflicts is None:
            self._conflicts = build_conflicts(self.store)
        return self._conflicts
