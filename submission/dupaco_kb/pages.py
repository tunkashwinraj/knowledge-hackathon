"""Page classification: what kind of page is this, how much should we trust it, and
which dates does it carry (publish date for news/blog, 'effective as of' for rate tables).

The rules are URL/structure based (not tied to any one question) so they also work on
other credit-union sites, e.g. the harness selftest site.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], 1)}
_MON_ABBR = {m[:3]: i for m, i in MONTHS.items()}

# Higher = more trustworthy for "what is the current value".
AUTHORITY = {
    "rates": 1.00,      # the official rate sheet(s)
    "product": 0.85,    # official product / service pages
    "home": 0.80,
    "landing": 0.55,    # root-level marketing pages (often frozen copies)
    "promo": 0.50,      # campaign landing pages (-lp, offers, switch-and-earn)
    "newsroom": 0.45,   # dated press releases
    "article": 0.40,    # blog / learn / resources articles
    "calculator": 0.30, # calculators embed old rate tables
    "document": 0.30,   # PDFs and downloads
    "legal": 0.20,
    "listing": 0.10,    # tag / author / index pages full of teasers
}

LABELS = {
    "rates": "OFFICIAL RATE SHEET",
    "product": "official product page",
    "home": "home page",
    "landing": "marketing landing page (may be an old copy)",
    "promo": "promotional landing page (may be outdated / expired)",
    "newsroom": "news release",
    "article": "blog / article",
    "calculator": "calculator page (rates may be outdated)",
    "document": "downloadable document",
    "legal": "legal / policy page",
    "listing": "index / listing page",
}

_PROMO_HINTS = ("-lp", "promo", "offer", "switch-and-earn", "earn-more", "challenge", "sprint",
                "campaign", "giveaway", "thankuse", "special-", "/lp/")
_PRODUCT_ROOTS = ("/personal", "/business", "/checking", "/savings", "/loans", "/about", "/contact",
                  "/membership", "/online-and-mobile-banking", "/connect", "/mobile", "/disclosures",
                  "/financial-services", "/careers", "/community", "/foundation", "/resources")


def path_of(url: str) -> str:
    p = urlparse(url if "//" in url else "https://" + url)
    return (p.path or "/").lower().rstrip("/") or "/"


def classify(url: str, title: str = "") -> str:
    path = path_of(url)
    query = urlparse(url).query.lower()
    segs = [s for s in path.split("/") if s]
    if path.startswith("/hubfs") or re.search(r"\.(pdf|xlsx?|docx?|pptx?)$", path):
        return "document"
    if any(s in ("tag", "author", "page", "category") for s in segs) or "category=" in query:
        return "listing"
    if "privacy-policy" in path or "terms-of-use" in path:
        return "legal"
    if path == "/":
        return "home"
    if segs and segs[0] == "newsroom":
        return "newsroom" if len(segs) > 1 else "listing"
    if "calculator" in path:
        return "calculator"
    if re.search(r"/20\d\d/\d\d/\d\d/", path + "/"):
        return "article"
    if segs and segs[0] in ("learn", "blog"):
        return "article" if len(segs) >= 3 or (len(segs) == 2 and segs[1] != "blog") else "listing"
    if "resources" in segs[:-1]:          # /business/resources/<article>
        return "article"
    if any(s == "rates" or s.endswith("-rates") for s in segs):
        return "rates"
    if any(h in path for h in _PROMO_HINTS):
        return "promo"
    if path.startswith(_PRODUCT_ROOTS):
        return "product"
    return "landing"


_DATE_LINE = re.compile(r"^(?:published:?\s*)?([A-Za-z]{3,9})\.?\s+(\d{1,2}),\s+(\d{4})$", re.I)


def publish_date(text: str) -> str | None:
    """First 'Month DD, YYYY' line near the top of the page (news / blog datelines)."""
    seen = 0
    for line in text.split("\n"):
        s = line.strip().strip("*_ ")
        if not s:
            continue
        seen += 1
        if seen > 15:
            break
        m = _DATE_LINE.match(s)
        if m:
            mon = MONTHS.get(m.group(1).lower()) or _MON_ABBR.get(m.group(1).lower()[:3])
            if mon:
                return f"{int(m.group(3)):04d}-{mon:02d}-{int(m.group(2)):02d}"
    return None


_EFFECTIVE = re.compile(r"effective\s+(?:as\s+of\s+|on\s+)?(\d{1,2})/(\d{1,2})/(\d{2,4})", re.I)


def effective_dates(text: str) -> list[str]:
    out = []
    for m in _EFFECTIVE.finditer(text or ""):
        y = int(m.group(3))
        y = y + 2000 if y < 100 else y
        out.append(f"{y:04d}-{int(m.group(1)):02d}-{int(m.group(2)):02d}")
    return out


def us_date(iso: str | None) -> str:
    if not iso:
        return ""
    y, m, d = iso.split("-")
    return f"{m}/{d}/{y}"
