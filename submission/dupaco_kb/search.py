"""Small pure-Python BM25 index over chunks, with snapshot filtering and query expansion."""
from __future__ import annotations

import math
import re
from collections import defaultdict

STOP = set("""a an the of to in on for and or is are was were be been being do does did i you your we
our us me my it its this that these those what which who whom whose when where why how can could would
should will shall may might must at by with from as about into than then there here any all some
please tell give get know need want like current currently now today right still dupaco dupaco's
credit union cu s""".split())

_NUMWORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
             "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "fifteen": "15",
             "eighteen": "18", "twenty": "20", "thirty": "30", "forty": "40", "sixty": "60"}

# query-side expansion: user words -> words the site uses
SYNONYMS = {
    "cd": ["certificate"], "cds": ["certificate"], "certificates": ["certificate"],
    "interest": ["rate", "dividend"], "yield": ["apy"], "apy": ["yield"],
    "car": ["auto", "vehicle"], "truck": ["auto", "vehicle"], "vehicle": ["auto"], "auto": ["vehicle"],
    "house": ["home", "mortgage"], "mortgage": ["home", "loan", "fixed", "conventional"],
    "home": ["mortgage"], "heloc": ["home", "equity", "line"],
    "card": ["visa"], "cards": ["visa", "card"], "visa": ["card"],
    "hours": ["lobby", "drive", "open"], "lobby": ["hours"], "drive": ["drive-up"],
    "atm": ["atms", "locator"], "locator": ["branch", "atm", "location"],
    "branches": ["branch", "location"], "branch": ["location", "office"],
    "address": ["location", "street"], "routing": ["aba", "number"],
    "join": ["member", "membership", "become"], "eligible": ["membership", "join", "live", "work"],
    "eligibility": ["membership", "join"], "phone": ["call", "contact"],
    "savings": ["share", "saving"], "checking": ["check"],
    "loan": ["loans"], "loans": ["loan"], "rates": ["rate"], "rate": ["rates", "apr", "apy"],
    "wallet": ["apple", "google", "samsung", "pay"], "student": ["student", "college"],
    "closing": ["close", "closure"], "close": ["closing", "closure"], "open": ["close"],
    "employees": ["employee"], "volunteer": ["volunteering", "vto"],
}

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def _stem(t: str) -> str:
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def tokenize(text: str) -> list[str]:
    text = (text or "").lower().replace("$", " ").replace("’", "'")
    toks = []
    for t in _TOKEN.findall(text):
        t = _NUMWORDS.get(t, t)
        if t in STOP:
            continue
        toks.append(_stem(t))
    return toks


def query_tokens(q: str) -> dict[str, float]:
    """Token -> weight. The member's own words weigh 1.0, expansions 0.4."""
    ql = q.lower()
    out: dict[str, float] = {}
    for t in tokenize(ql):
        out[t] = 1.0
    extra = []
    # "5-year certificate" -> also "60 month"
    for n, unit in re.findall(r"(\d+)[- ]?(year|yr)s?\b", ql):
        if int(n) <= 10:
            extra += [str(int(n) * 12), "month"]
    for t in list(out):
        extra += SYNONYMS.get(t, [])
    for t in tokenize(" ".join(extra)):
        out.setdefault(t, 0.4)
    return out


class BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.6):
        self.k1, self.b = k1, b
        self.post: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.dlen: dict[int, int] = {}
        self.total = 0

    def add(self, doc_id: int, tokens: list[str]) -> None:
        tf = defaultdict(int)
        for t in tokens:
            tf[t] += 1
        for t, n in tf.items():
            self.post[t].append((doc_id, n))
        self.dlen[doc_id] = len(tokens)
        self.total += len(tokens)

    def search(self, qw: dict[str, float], allowed, k: int = 60) -> list[tuple[int, float]]:
        N = len(self.dlen) or 1
        avg = self.total / N
        scores: dict[int, float] = defaultdict(float)
        for t, w in qw.items():
            plist = self.post.get(t)
            if not plist:
                continue
            df = len(plist)
            if N > 200 and df > N * 0.35:      # near-stopword for this corpus (only meaningful on big sites)
                continue
            idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
            for d, tf in plist:
                if not allowed(d):
                    continue
                dl = self.dlen[d]
                scores[d] += w * idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / avg))
        return sorted(scores.items(), key=lambda x: -x[1])[:k]
