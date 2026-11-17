"""Versioned knowledge store.

* Every URL keeps a history of distinct (cleaned) versions, each tagged with the set of
  snapshots in which it was the live copy.  "What did the site say on date X" is then a
  simple filter.
* Pages are split into chunks (heading sections; tables kept whole together with their
  caption and their "Rates are effective as of" line).
* Every percentage cell of every table becomes a structured Fact (product, qualifiers,
  metric, value, effective date) used for conflict detection and stale-value guards.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .clean import clean_markdown, content_hash, norm_text, percents
from .pages import AUTHORITY, classify, effective_dates, publish_date


@dataclass
class Version:
    vid: int
    url: str
    title: str
    text: str
    hash: str
    ptype: str
    authority: float
    pub_date: Optional[str]
    fetched: str
    snaps: set = field(default_factory=set)
    chunk_ids: list = field(default_factory=list)


@dataclass
class Chunk:
    cid: int
    vid: int
    kind: str              # "table" | "text"
    heading: str
    text: str
    eff_date: Optional[str]
    thash: str


@dataclass
class Fact:
    vid: int
    cid: int
    product: str           # product class, e.g. "auto loan", "certificate"
    label: str             # human readable row label (caption + row cells)
    quals: dict            # normalized qualifiers used for matching
    metric: str            # APY / APR / DIV / RATE
    value: tuple           # tuple of percentages (ranges keep both ends)
    value_str: str
    eff_date: Optional[str]
    url: str = ""
    ptype: str = ""
    authority: float = 0.0
    named: bool = True     # product identified from the row/caption itself (safe to match across pages)
    ambiguous: bool = False  # the same table gives this row several values (an unparsed qualifier)


# ------------------------------------------------------------------ product classes
PRODUCT_RULES = [
    ("hsa", r"health savings|\bhsa\b"),
    ("money market", r"money market|investor.?s choice|\bachieve\b"),
    ("ira certificate", r"\bira\b|\broth\b|traditional:"),
    ("certificate", r"certificate|\bcds?\b|term share"),
    ("auto loan", r"\bauto\b|vehicle|motorcycle|\bcar loan"),
    ("boat/rv loan", r"\bboat|\brv\b|recreational vehicle"),
    ("atv loan", r"\batv\b|snowmobile|jet ski|power ?sports?"),
    ("home equity", r"home equity|heloc|equity line"),
    ("home improvement loan", r"home improvement"),
    ("mortgage", r"fixed conventional|mortgage|year fixed|\barm\b|home loans?|purchase:|refinance:|rate relief"),
    ("credit card", r"\bvisa\b|credit card"),
    ("personal loan", r"personal loan|signature loan|credit coach"),
    ("checking", r"checking"),
    ("savings", r"savings|club account|share account"),
]
_PRODUCT_RES = [(name, re.compile(rx, re.I)) for name, rx in PRODUCT_RULES]


def product_class(*texts: str) -> str:
    for t in texts:
        if not t:
            continue
        for name, rx in _PRODUCT_RES:
            if rx.search(t):
                return name
    return "other"


# ------------------------------------------------------------------ qualifiers
_TERM = re.compile(r"(\d+)\s*[- ]?\s*(month|mo|year|yr)s?\b", re.I)
_BAND = re.compile(r"(?:(\d{4})\s*[-–]\s*(\d{2,4})|(\d{4})\s*(?:\(?and older\)?|or older|\+))", re.I)
_AMT = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)")
STRICT_QUALS = ["jumbo", "special", "roth", "traditional", "business", "intro", "rewards", "platinum",
                "balance transfer", "cash advance", "high-yield", "bright", "launchpad", "investor",
                "achieve", "holiday", "1st rate", "free", "kasasa", "youth", "teen", "student"]
GROUP_QUALS = [("purchase", "refinance"), ("new", "used"), ("fixed", "variable")]


def _amount_key(label: str) -> str:
    s = label.lower().replace("–", "-").replace("—", "-")
    nums = [float(x.replace(",", "")) for x in _AMT.findall(s)]
    if not nums:
        return ""
    fmt = lambda v: f"{v:.2f}".rstrip("0").rstrip(".")
    if len(nums) >= 2:
        return f"{fmt(nums[0])}-{fmt(nums[1])}"
    if "or less" in s or "up to" in s or "under" in s:
        return f"<={fmt(nums[0])}"
    if "+" in s or "or more" in s or "and above" in s or "over" in s:
        return f"{fmt(nums[0])}+"
    return fmt(nums[0])


def qualifiers(label: str, band_rank: Optional[int]) -> dict:
    low = norm_text(label)
    terms = tuple(sorted({f"{int(n)} {'month' if u.lower().startswith('mo') else 'year'}"
                          for n, u in _TERM.findall(low)}))
    strict = tuple(sorted(q for q in STRICT_QUALS if q in low))
    groups = tuple(sorted(next((g for g in grp if g in low), "") for grp in GROUP_QUALS))
    return {"terms": terms, "amount": _amount_key(low), "band": band_rank, "strict": strict,
            "groups": groups}


def quals_match(a: dict, b: dict) -> bool:
    if a["terms"] != b["terms"] or a["amount"] != b["amount"] or a["band"] != b["band"]:
        return False
    if a["strict"] != b["strict"]:
        return False
    for x, y in zip(a["groups"], b["groups"]):
        if x and y and x != y:
            return False
    return bool(a["terms"] or a["amount"] or a["band"] is not None or a["strict"])


# ------------------------------------------------------------------ chunking
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET = re.compile(r"^\s*([-*+]|\d+[.)])\s")


def _metric_of(header: str, cell: str) -> Optional[str]:
    c = cell.upper()
    if re.search(r"%\s*APR\b", c):
        return "APR"
    if re.search(r"%\s*APY\b", c):
        return "APY"
    h = header.lower()
    if any(k in h for k in ("min", "payment", "points", "fee", "open", "earn", "amount")):
        return None
    if "apy" in h or "annual percentage yield" in h:
        return "APY"
    if "apr" in h or "annual percentage rate" in h:
        return "APR"
    if "dividend" in h:
        return "DIV"
    if "rate" in h or "yield" in h:
        return "RATE"
    return None


def _split_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _band_ranks(rows: list[list[str]]) -> dict:
    """Map each model-year band label (e.g. '2024-27 Model Year') to its rank, newest first."""
    starts = {}
    for r in rows:
        for c in r:
            m = _BAND.search(c)
            if m and ("model" in c.lower() or "older" in c.lower()):
                start = int(m.group(1) or m.group(3))
                starts[norm_text(c)] = start
    order = sorted(set(starts.values()), reverse=True)
    return {lab: order.index(st) + 1 for lab, st in starts.items()}


class Chunker:
    TEXT_MAX = 1200

    def __init__(self, title: str):
        self.title = title

    def run(self, text: str):
        """Yield (kind, heading, chunk_text, eff_date, table_info|None)."""
        lines = text.split("\n")
        heads: list[tuple[int, str]] = []
        buf: list[str] = []
        since_head: list[str] = []
        out = []

        def heading_path():
            return " > ".join(h for _, h in heads)

        def flush():
            body = "\n".join(buf).strip()
            buf.clear()
            if not body:
                return
            # split long text into ~TEXT_MAX pieces on paragraph boundaries
            paras, cur = body.split("\n\n"), ""
            for p in paras:
                if len(cur) + len(p) > self.TEXT_MAX and cur:
                    out.append(("text", heading_path(), cur.strip(), None, None))
                    cur = ""
                while len(p) > self.TEXT_MAX * 1.5:
                    cut = p.rfind(". ", 0, self.TEXT_MAX)
                    cut = cut + 1 if cut > 200 else self.TEXT_MAX
                    out.append(("text", heading_path(), (cur + p[:cut]).strip(), None, None))
                    cur, p = "", p[cut:]
                cur += p + "\n\n"
            if cur.strip():
                out.append(("text", heading_path(), cur.strip(), None, None))

        i = 0
        while i < len(lines):
            line = lines[i]
            m = _HEADING.match(line.strip())
            if m:
                flush()
                level, htext = len(m.group(1)), re.sub(r"[*_]", "", m.group(2)).strip()
                heads = [h for h in heads if h[0] < level] + ([(level, htext)] if htext else [])
                since_head = []
                i += 1
                continue
            if line.strip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|[\s:|-]+\|?\s*$", lines[i + 1]):
                # caption = last short, non-bullet line since the last heading
                caption = ""
                for prev in reversed(since_head[-25:]):
                    p = prev.strip()
                    if p and len(p) <= 70 and not _BULLET.match(p) and not p.startswith(("[", "|", "!")) \
                            and not p.endswith((".", ":")):
                        caption = re.sub(r"[*_]", "", p)
                        break
                flush()
                tbl = [line]
                i += 1
                while i < len(lines) and lines[i].strip().startswith("|"):
                    tbl.append(lines[i])
                    i += 1
                # effective-date sentence within the next few non-empty lines
                eff_line, j, seen = "", i, 0
                while j < len(lines) and seen < 4:
                    s = lines[j].strip()
                    if s:
                        seen += 1
                        if s.startswith("|") or _HEADING.match(s):
                            break
                        if effective_dates(s):
                            eff_line = s[:300]
                            break
                    j += 1
                effs = effective_dates(eff_line)
                header = _split_row(tbl[0])
                rows = [_split_row(r) for r in tbl[2:]]
                info = {"caption": caption, "header": header, "rows": rows}
                # split big tables, repeating the header
                body_rows = tbl[2:]
                step = max(1, 2400 // max(1, max((len(r) for r in body_rows), default=1)))
                for k in range(0, max(1, len(body_rows)), step):
                    part = "\n".join(tbl[:2] + body_rows[k:k + step])
                    txt = (f"{caption}\n" if caption else "") + part + (f"\n{eff_line}" if eff_line else "")
                    out.append(("table", heading_path(), txt, effs[0] if effs else None,
                                info if k == 0 else None))
                continue
            buf.append(line)
            since_head.append(line)
            i += 1
        flush()
        return self._merge_small(out)

    def _merge_small(self, out):
        """Pages with many small heading sections produce tiny chunks that lose context.
        Merge consecutive text pieces (headings kept inline) up to TEXT_MAX characters."""
        merged = []
        for kind, heading, text, eff, info in out:
            if kind == "text" and merged and merged[-1][0] == "text" and \
                    len(merged[-1][2]) + len(text) + 40 <= self.TEXT_MAX:
                prev = merged[-1]
                last = heading.split(" > ")[-1] if heading else ""
                joined = prev[2] + ("\n\n" + f"### {last}" if last and heading != prev[1] else "") + "\n\n" + text
                merged[-1] = ("text", prev[1], joined, None, None)
                continue
            if kind == "text" and heading:
                text = f"### {heading.split(' > ')[-1]}\n\n{text}" if len(text) < self.TEXT_MAX else text
            merged.append((kind, heading, text, eff, info))
        return merged


# ------------------------------------------------------------------ the store
class Store:
    def __init__(self):
        self.snaps: list[tuple[str, str]] = []       # (snapshot_id, taken_on)
        self.versions: list[Version] = []
        self.by_url: dict[str, list[int]] = {}
        self.chunks: list[Chunk] = []
        self.facts: list[Fact] = []
        self.link_urls: set[str] = set()             # every URL that appears anywhere (pages + links)
        self.on_new_chunk = None                     # callback(chunk, version) for the index
        self._last_table = (-1, "", "other")         # (vid, heading, product) of the previous table

    def add_snapshot(self, snapshot) -> None:
        sidx = len(self.snaps)
        self.snaps.append((snapshot.snapshot_id, snapshot.taken_on))
        for page in snapshot.pages:
            text = clean_markdown(page.markdown)
            h = content_hash(text)
            hist = self.by_url.setdefault(page.url, [])
            if hist and self.versions[hist[-1]].hash == h:
                self.versions[hist[-1]].snaps.add(sidx)
                continue
            # a page can come back to an earlier text; reuse that version
            same = next((v for v in hist if self.versions[v].hash == h), None)
            if same is not None:
                self.versions[same].snaps.add(sidx)
                hist.remove(same)
                hist.append(same)
                continue
            self._add_version(page, text, h, sidx)

    def _add_version(self, page, text: str, h: str, sidx: int) -> None:
        ptype = classify(page.url, page.title)
        v = Version(vid=len(self.versions), url=page.url, title=page.title or "", text=text, hash=h,
                    ptype=ptype, authority=AUTHORITY[ptype],
                    pub_date=publish_date(text) if ptype in ("newsroom", "article", "listing") else None,
                    fetched=(page.fetched_at or "")[:10], snaps={sidx})
        self.versions.append(v)
        self.by_url[page.url].append(v.vid)
        self.link_urls.add(page.url.rstrip("/"))
        for u in re.findall(r"\]\((https?://[^)\s]+)\)", text):
            self.link_urls.add(u.split("#")[0].rstrip("/"))
        for kind, heading, ctext, eff, info in Chunker(v.title).run(text):
            c = Chunk(cid=len(self.chunks), vid=v.vid, kind=kind, heading=heading, text=ctext,
                      eff_date=eff, thash=content_hash(ctext))
            self.chunks.append(c)
            v.chunk_ids.append(c.cid)
            if info:
                self._table_facts(v, c, info, heading)
            elif kind == "text" and v.ptype in ("rates", "product", "home", "landing", "promo", "calculator"):
                self._prose_facts(v, c)
            if self.on_new_chunk:
                self.on_new_chunk(c, v)

    _PROSE_FACT = re.compile(r"^[-*\s]*(?P<label>[A-Za-z0-9][^:|\n]{2,90}?[A-Za-z)]):\s*(?P<val>\$\s?\d[\d,]*(?:\.\d+)?|"
                             r"\d+(?:\.\d+)?\s*%)\s*(?P<rest>[A-Za-z ()]{0,20})$")

    def _prose_facts(self, v: Version, c: Chunk) -> None:
        """Short 'label: value' lines ("6-month certificate APY: 3.50%", "Replacement fee: $2.50")."""
        effs = effective_dates(c.text)
        for line in c.text.split("\n"):
            m = self._PROSE_FACT.match(line.strip().replace("**", ""))
            if not m:
                continue
            label, raw = m.group("label").strip(), m.group("val")
            low = (label + " " + m.group("rest")).lower()
            if "%" in raw:
                value = tuple(percents(raw))
                metric = "APY" if "apy" in low or "yield" in low else "APR" if "apr" in low else \
                    "DIV" if "dividend" in low else "RATE"
            else:
                value = (float(raw.replace("$", "").replace(",", "").strip()),)
                metric = "FEE" if "fee" in low or "cost" in low or "charge" in low else "AMOUNT"
            prod = product_class(label)
            self.facts.append(Fact(vid=v.vid, cid=c.cid, product=prod, label=label,
                                   quals=qualifiers(label + (" business" if "business" in v.url.lower() else ""),
                                                    None),
                                   metric=metric, value=value, value_str=raw.replace(" ", ""),
                                   eff_date=effs[0] if effs else None, url=v.url, ptype=v.ptype,
                                   authority=v.authority, named=prod != "other"))

    def _table_facts(self, v: Version, c: Chunk, info: dict, heading: str) -> None:
        header, rows, caption = info["header"], info["rows"], info["caption"]
        bands = _band_ranks(rows)
        heads_rev = list(reversed(heading.split(" > "))) if heading else []
        biz_page = "business" in v.url.lower()
        start = len(self.facts)
        table_prod = "other"
        for r in rows:
            if len(r) < 2:
                continue
            label_cells, values = [], []
            for col, cell in enumerate(r):
                hdr = (header[col] if col < len(header) else "").lower()
                metric = _metric_of(hdr, cell)
                if metric and percents(cell):
                    values.append((metric, cell))
                    continue
                if "%" in cell or not cell.strip():
                    continue
                is_tier_col = any(k in hdr for k in ("tier", "balance", "amount", "loan", "term", "program"))
                if any(k in hdr for k in ("min", "payment", "points", "fee", "open", "earn")) and not is_tier_col:
                    continue
                if re.match(r"^\$?\s*[\d,.]+\s*$", cell) and not is_tier_col:
                    continue
                label_cells.append(cell)
            if not values or not label_cells:
                continue
            label = " | ".join(x for x in label_cells if x)
            band = next((bands[norm_text(x)] for x in label_cells if norm_text(x) in bands), None)
            label_prod = product_class(label)
            prod = label_prod if label_prod != "other" else product_class(caption)
            named = prod != "other"
            if not named and self._last_table[:2] == (v.vid, heading) and self._last_table[2] != "other":
                prod, named = self._last_table[2], True       # tabbed rate pages: same product family
            if not named:
                prod = product_class(*heads_rev, v.title)
            table_prod = prod if named else table_prod
            # the caption only qualifies rows that do not name their product themselves
            qtext = label if label_prod != "other" else label + " " + caption
            quals = qualifiers(qtext + (" business" if biz_page else ""), band)
            for metric, cell in values:
                pv = tuple(percents(cell))
                if len(pv) > 2:
                    continue          # several values crammed into one cell: not a clean fact
                vs = " – ".join(re.sub(r"\s+", "", x) for x in re.findall(r"\d+(?:\.\d+)?\s*%", cell))
                self.facts.append(Fact(vid=v.vid, cid=c.cid, product=prod,
                                       label=(caption + ": " if caption else "") + label,
                                       quals=quals, metric=metric, value=pv, value_str=vs,
                                       eff_date=c.eff_date, url=v.url, ptype=v.ptype,
                                       authority=v.authority, named=named))
        # rows of one table that look identical but carry different values (e.g. intro vs
        # standard rate in a column we could not interpret) are ambiguous: never compare them
        seen: dict = {}
        for f in self.facts[start:]:
            k = (f.product, f.metric, repr(sorted(f.quals.items())))
            seen.setdefault(k, set()).add(f.value)
        for f in self.facts[start:]:
            k = (f.product, f.metric, repr(sorted(f.quals.items())))
            if len(seen[k]) > 1:
                f.ambiguous = True
        self._last_table = (v.vid, heading, table_prod)

    # ---------------------------------------------------------------- queries
    def live(self, vid: int, sidx: int) -> bool:
        return sidx in self.versions[vid].snaps

    def carried(self, sidx: int) -> set:
        """Pages missing from snapshot `sidx` but seen earlier: their last version before sidx.
        A crawl can drop a page that still holds the only statement of a fact."""
        cache = self.__dict__.setdefault("_carried_cache", {})
        key = (sidx, len(self.versions))
        if key not in cache:
            out = set()
            for url, hist in self.by_url.items():
                vs = [self.versions[v] for v in hist]
                if any(sidx in v.snaps for v in vs):
                    continue
                before = [v for v in vs if v.snaps and min(v.snaps) < sidx]
                if before:
                    out.add(max(before, key=lambda v: max(s for s in v.snaps if s < sidx)).vid)
            cache[key] = out
        return cache[key]

    def last_seen(self, vid: int) -> str:
        return self.snaps[max(self.versions[vid].snaps)][1]

    def version_at(self, url: str, sidx: int) -> Optional[Version]:
        for vid in reversed(self.by_url.get(url, [])):
            if sidx in self.versions[vid].snaps:
                return self.versions[vid]
        return None

    def latest_index(self) -> int:
        return len(self.snaps) - 1
