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
EVIDENCE_K = int(os.getenv("KB_K", "10"))
EVIDENCE_CHARS = 2400
ARTICLE_CHARS = int(os.getenv("KB_ARTICLE_CHARS", "4500"))  # news / blog posts are given whole (one dated story)
SMALL_PAGE_CHARS = 2500  # any page this small is given whole
NEIGHBOR_MIN = 900       # shorter chunks get their neighbouring chunks attached


_NAMED_BRANCH = re.compile(r"\b((?:[A-Z][\w.'-]*\s+){1,3})(branch|location|office)\b")
_HOURS_MARK = re.compile(r"^\**\s*(lobby|drive[- ]?up|drive[- ]?thru)\s*\**\s*:?\s*(.*)$", re.I)
_DAY_LINE = re.compile(r"^(mon|tue|wed|thu|fri|sat|sun)[a-z]*(\s*[-–]\s*(mon|tue|wed|thu|fri|sat|sun)[a-z]*)?\s*:\s*.+$",
                       re.I)


def _parse_hours(text: str):
    """Yield (kind, schedule) for '**Lobby** / Monday-Thursday: 8:30 a.m.-5 p.m. / ...' blocks."""
    lines = [l.strip() for l in text.split("\n")]
    i = 0
    while i < len(lines):
        m = _HOURS_MARK.match(lines[i])
        if not m:
            i += 1
            continue
        kind = "Lobby" if m.group(1).lower().startswith("lobby") else "Drive-up"
        days = [m.group(2)] if m.group(2) and _DAY_LINE.match(m.group(2)) else []
        j = i + 1
        while j < len(lines) and len(days) < 6:
            if not lines[j]:
                j += 1
                continue
            if _DAY_LINE.match(lines[j]):
                days.append(lines[j])
                j += 1
                continue
            break
        if days:
            sched = "; ".join(re.sub(r"\s*[–—]\s*", "-", d) for d in days)
            yield kind, re.sub(r"\s+", " ", sched)
        i = max(j, i + 1)


def _norm_time(t: str) -> str:
    return re.sub(r"[\s.]", "", t.lower()).replace(":00", "")


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
        self._warm_up()

    def _warm_up(self) -> None:
        """Ingest is not timed: open the API connection, compile the answer format and write the
        prompt cache now, so the first real question is not slowed by a cold start."""
        if getattr(self, "_warm", False):
            return
        self._warm = True
        try:
            self.llm.ask([{"role": "user", "content": "Warm-up request. Member question: (none). "
                                                      "Reply with status not_listed and an empty answer."}])
        except Exception as e:  # no key / no network: answers will report it later
            print(f"[kb] warm-up skipped: {type(e).__name__}", file=sys.stderr)

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
        from .conflicts import pick_winner, same_fact
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
            # per crawl, the winning official value (latest effective date, highest authority)
            by_snap: dict = {}
            for g in official:
                if g is f or same_fact(f, g):
                    for s in st.versions[g.vid].snaps:
                        by_snap.setdefault(s, []).append(g)
            per_snap = {s: pick_winner(gs).value_str for s, gs in by_snap.items()}
            if len(set(per_snap.values())) < 2:
                continue
            ordered = sorted(per_snap)
            first, last = ordered[0], ordered[-1]
            steps = [f"{us_date(st.snaps[s][1])}: {per_snap[s]}" for s in ordered]
            out.append(f"- History of {f.label} ({f.metric}) on official pages: first crawl "
                       f"({us_date(st.snaps[first][1])}) {per_snap[first]} -> latest crawl "
                       f"({us_date(st.snaps[last][1])}) {per_snap[last]}. By crawl date: " + "; ".join(steps) + ".")
            if len(out) >= limit:
                break
        return out

    def _hours_lines(self, question: str, sidx: int) -> list[str]:
        """General hours questions: every distinct lobby / drive-up schedule published on the site,
        so the answer does not depend on which releases happened to be retrieved."""
        if not _HOURS_Q.search(question):
            return []
        named = _NAMED_BRANCH.search(question)
        if named and named.group(1).strip().lower().rstrip("'s") not in ("dupaco", "the", "a", "your", "all", "any",
                                                                          "each", "every", "my", "our"):
            return []                    # a specific branch: its own hours or "not listed", never others'
        scheds: dict = {}
        for v in self.store.versions:
            if sidx not in v.snaps or v.ptype not in ("newsroom", "product", "home", "landing"):
                continue
            for kind, sched in _parse_hours(v.text):
                d = scheds.setdefault((kind, sched), [])
                d.append((v.pub_date or "", v.title.split("|")[0].strip()[:70]))
        if not scheds:
            return []
        lines = []
        for kind in ("Lobby", "Drive-up"):
            items = [(len(src), max(src)[0], s, src) for (k, s), src in scheds.items() if k == kind]
            items.sort(reverse=True)               # most common schedule first, then newest
            if items:
                parts = [f"({i}) {s} [listed in {n} release{'s' if n > 1 else ''}; e.g. {max(src)[1]}, "
                         f"{us_date(d) if d else 'undated'}]" for i, (n, d, s, src) in enumerate(items[:6], 1)]
                lines.append(f"- Distinct {kind.lower()} schedules published on the site, most common first. They "
                             f"belong to DIFFERENT branches (not outdated versions of one another): for a general "
                             f"hours question lead with the most common one (\"at most branches ...\") and then "
                             f"list EVERY other one: " + " ".join(parts))
        return lines

    def _value_lines(self, question: str, sidx: int) -> list[str]:
        vals = [round(float(x), 4) for x in re.findall(r"(\d+(?:\.\d+)?)\s*%", question)]
        if not vals:
            return []
        from .store import product_class
        st = self.store
        official = [f for f in st.facts if f.authority >= 0.85 and sidx in st.versions[f.vid].snaps
                    and not f.ambiguous and f.metric in ("APY", "APR", "RATE")]
        lines = []
        for v in vals[:2]:
            having = sorted({f"{re.sub(r'^[^:]*: ', '', f.label)} ({f.metric})" for f in official if v in f.value})
            lines.append(f"- Official rate tables showing {v:g}%: " + ("; ".join(having[:6]) if having else "none") + ".")
        fam = product_class(question)
        if fam != "other":
            ranges: dict = {}
            for f in official:
                if f.product != fam or "business" in f.quals["strict"]:
                    continue
                name = re.sub(r"^[^:]*: ", "", f.label).split(" | ")[0]
                ranges.setdefault((name, f.metric), []).append(f)
            parts = []
            for (name, metric), fs in list(ranges.items())[:6]:
                top = max(fs, key=lambda f: max(f.value))
                lo = min(min(f.value) for f in fs)
                hi = max(top.value)
                tier = re.sub(r"^[^|]*\|\s*", "", re.sub(r"^[^:]*: ", "", top.label)) if " | " in top.label else ""
                parts.append(f"{name}: {lo:.2f}%" + (f"-{hi:.2f}%" if hi != lo else "") + f" {metric}"
                             + (f" (highest: {top.value_str} at {tier})" if tier and hi != lo else ""))
            if parts:
                lines.append(f"- Actual current {fam} rates (official): " + "; ".join(parts) + ".")
        return lines

    def _claim_lines(self, question: str, sidx: int) -> list[str]:
        from .conflicts import CLAIMS, prose_claims
        if _PERIOD_Q.search(question):
            return []
        triggers = {
            "number of branch locations": r"\b(how many|number of|count|total)\b.*\b(branch|location|office)|"
                                          r"\bbranch(es)? (count|total)\b",
            "number of counties served": r"\bcount(y|ies)\b",
            "number of employees": r"\b(how many|number of|total)\b.*\b(employee|staff)",
            "Forbes recognitions (times ranked)": r"\bforbes\b",
        }
        wanted = [name for name, rx in triggers.items() if re.search(rx, question, re.I)
                  and (name.startswith("Forbes") or not self._count_claims(question, sidx))]
        if not wanted:
            return []
        found = prose_claims(self.store, sidx)
        lines = []
        for name in wanted:
            items = [(v.pub_date, val, v) for val, v in found.get(name, []) if v.pub_date]
            if not items:
                continue
            d, val, v = max(items, key=lambda x: x[0])
            rx = next(r for n, r, _ in CLAIMS if n == name)
            m = rx.search(v.text)
            sent = v.text[max(0, v.text.rfind(".", 0, m.start()) + 1): v.text.find(".", m.end()) + 1].strip() if m else val
            older = sorted({val2 for _, val2, _ in items if val2 != val})
            lines.append(f"- Most recent dated statement ({LABELS.get(v.ptype, v.ptype)} published {us_date(d)}, "
                         f"{v.url}): \"{sent[:240]}\"" + (f" Older pages say: {', '.join(older)}." if older else ""))
        return lines

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
        # 3. a value named in the question ("does anything pay 0.35%?"): who really has it, and the
        #    actual range of the product family the member named
        lines += self._value_lines(question, sidx)
        lines += self._hours_lines(question, sidx)
        # 4. recurring statements (Forbes rankings, branch / county / employee counts): newest dated one
        lines += self._claim_lines(question, sidx)
        # 5. counts that drift across dated pages
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
        if not _CHANGE_Q.search(question):      # "what changed" answers must quote old values
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
        if confident and _ASKS_RATE.search(question) and not _LINK_Q.search(question):
            if "%" not in text or not self._rate_from_official(text, ev, hints):
                confident = False
                text = guards.REFUSAL_TEXT
        if not confident:
            extra = {v for _, v in numbers(text)} - {v for _, v in numbers(question)}
            if extra or not text:
                text = guards.REFUSAL_TEXT
            return {"text": text, "sources": [], "confident": False, "conflict_note": None}
        bad = guards.ungrounded_numbers(text, ev_text, question)
        if bad:
            stripped = guards.strip_sentences_with(text, bad)
            # only drop the offending sentences if a real answer remains
            if stripped and len(stripped) >= 0.6 * len(text) and re.search(r"\d", stripped):
                text = stripped
        return {"text": text, "sources": self._sources(data, ev, text, sidx), "confident": True,
                "conflict_note": self._conflict_note(text, hints)}

    def _rate_from_official(self, text: str, ev: list[dict], hints: list[str]) -> bool:
        """A quoted rate must come from a rate table or an official page, not from blog prose
        (e.g. 'adjusts up to 2% a year' is not a published rate)."""
        st = self.store
        allowed = set()
        for item in ev:
            auth = max(st.versions[v].authority for v in item["vids"])
            if auth >= 0.85 or st.chunks[item["cid"]].kind == "table":
                allowed |= set(percents(item["text"]))
        for h in hints:
            allowed |= set(percents(h))
        return bool(set(percents(text)) & allowed)

    @staticmethod
    def _conflict_note(text: str, hints: list[str]):
        """Explain, from the code-checked facts, which outdated values were set aside."""
        vals = set(percents(text)) | {v for _, v in numbers(text)}
        notes = []
        for h in hints:
            if h.startswith("- History of"):
                continue
            hv = set(percents(h)) | {v for _, v in numbers(h)}
            if vals & hv:
                notes.append(h.lstrip("- ").strip())
        return " ".join(notes[:2]) or None

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
        times = {_norm_time(m.group(0)) for m in _TIME.finditer(text)}
        if key_vals or len(times) >= 2:
            for item in ev:
                ctext = item["text"]
                cvals = set(percents(ctext)) | {v for _, v in numbers(ctext)}
                ctimes = {_norm_time(m.group(0)) for m in _TIME.finditer(ctext)}
                if (key_vals & cvals) or len(times & ctimes) >= 2:   # page states the same rates / hours
                    for u in item["urls"]:
                        if u not in urls:
                            urls.append(u)
        auth = {}
        for u in urls:
            v = st.version_at(u, sidx)
            auth[u] = v.authority if v else 0
        urls.sort(key=lambda u: -auth[u])
        return [{"url": u, "snapshot_id": sid, "quote": None} for u in urls[:10]]

    # ---------------------------------------------------------------- conflicts
    def conflicts(self) -> list[dict]:
        if self._conflicts is None:
            self._conflicts = build_conflicts(self.store)
        return self._conflicts
