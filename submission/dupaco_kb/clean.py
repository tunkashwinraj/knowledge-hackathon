"""Page cleaning and value normalisation.

Crawled pages carry a lot of noise that changes on every crawl (captcha images, chat
widgets, phone-country pickers, "related posts" teasers). We strip it so that
(a) unchanged pages hash identically across snapshots and (b) retrieval and the LLM only
see real content. We also remove footnote digits that the crawler glued onto values
("4.20%1" -> "4.20%", "5.39% APR1" -> "5.39% APR").
"""
from __future__ import annotations

import hashlib
import re

_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_EMPTY_LINK = re.compile(r"\[\s*\]\([^)]*\)")
_FLAG_LINE = re.compile(r"^\s*-\s*[\U0001F1E6-\U0001F1FF]{2}")
_ESC_NEWLINE = re.compile(r"(?:\\\\|\\)r(?:\\\\|\\)n")
_MD_ESCAPE = re.compile(r"\\([*#_\[\]()$!>+-])")
_NOISE_EXACT = {
    "✕", "load more", "#### filter", "###", "# # #", "third party plugins",
    "messenger launcher", "messenger", "* * *", "cancel", "continue", "---", "twitter widget iframe",
}
_NOISE_RE = [re.compile(p, re.I) for p in (
    r"^\d+ min read$",
    r"^sort by:",
    r"^select all images",
    r"recaptcha",
    r"^\[continue\]\(.*\)\s*cancel$",
    r"^\|[\s|]*\|$",                       # table rows emptied by image removal
    r"^(skip to (main )?content|back to top)$",
    r"^\**heads up! this link leads to a different website",
    r"^we only do this when it.s helpful for you",
    r"^#{1,6}\s*\[[^\]]+\]\([^)]+\)\s*$",       # teaser card titles (links to other posts)
    r"^\[(read more|keep reading|learn more|read the full story)\]\([^)]*\)$",
)]
# Sections under these headings are "related content" for OTHER pages (teasers with their
# own dates/numbers) and must not be attributed to this page. We skip them until the next
# heading of the same or a higher level.
_SKIP = re.compile(r"^(#{1,6})\s*\**\s*(handpicked for you|related (posts|articles|stories|resources)|"
                   r"you (might|may) also like|more from|recent (posts|news|articles)|latest (news|posts))\b", re.I)
_HEAD = re.compile(r"^(#{1,6})\s")

# Footnote markers glued to values.  Hex chars are excluded from the look-ahead so
# URL escapes such as "%2C" are left alone.
_FOOT_PCT = re.compile(r"(\d%)(\d)(?![\dA-Fa-f.,%])")
_FOOT_WORD = re.compile(r"\b(APR|APY)(\d)(?![\d.,%])")
_FOOT_PAREN = re.compile(r"\)(\d)(?=\s*(?:\||$))", re.M)
_FOOT_LEAD = re.compile(r"^(\d)(?=[A-Z][a-z])", re.M)


def clean_markdown(md: str) -> str:
    text = _ESC_NEWLINE.sub("\n", md or "")
    text = text.replace("http://www.", "https://www.")
    text = _IMG.sub("", text)
    text = _EMPTY_LINK.sub("", text)
    text = _MD_ESCAPE.sub(r"\1", text)
    out = []
    skip_level = 0
    for raw in text.split("\n"):
        line = raw.rstrip()
        s = line.strip()
        m = _SKIP.match(s)
        if m:
            skip_level = len(m.group(1))
            continue
        if skip_level:
            h = _HEAD.match(s)
            if h and len(h.group(1)) <= skip_level and not re.match(r"^#{1,6}\s*\[", s):
                skip_level = 0
            else:
                continue
        low = s.lower()
        if low in _NOISE_EXACT or _FLAG_LINE.match(s) or any(r.search(s) for r in _NOISE_RE):
            continue
        out.append(line)
    text = "\n".join(out)
    text = _FOOT_PCT.sub(r"\1", text)
    text = _FOOT_WORD.sub(r"\1", text)
    text = _FOOT_PAREN.sub(")", text)
    text = _FOOT_LEAD.sub("", text)
    text = text.replace("\u00a0", " ").replace("\u2011", "-")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def content_hash(text: str) -> str:
    """Whitespace-insensitive hash: re-crawls often differ only in blank lines."""
    return hashlib.sha1(re.sub(r"\s+", " ", text).strip().encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- numbers / values
# Same number grammar as the official scorer, so "3.5%" and "3.50 %" compare equal.
NUM_RE = re.compile(r"(\$)?\s*(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?\s*(%|percent)?", re.I)


def numbers(text: str) -> set[tuple[str, float]]:
    out = set()
    for m in NUM_RE.finditer(text or ""):
        unit = "$" if m.group(1) else ("%" if m.group(4) else "")
        val = float(m.group(2).replace(",", "") + ("." + m.group(3) if m.group(3) else ""))
        out.add((unit, round(val, 4)))
    return out


_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def percents(text: str) -> list[float]:
    return [round(float(x), 4) for x in _PCT.findall(text or "")]


def fmt_pct(v: float) -> str:
    s = f"{v:.3f}".rstrip("0")
    if s.endswith("."):
        s += "00"
    elif len(s.split(".")[1]) == 1:
        s += "0"
    return s + "%"


def norm_text(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("–", "-").replace("—", "-").replace("’", "'")
    s = re.sub(r"\.00\b", "", s)
    s = re.sub(r"[*_`]", "", s)
    return re.sub(r"\s+", " ", s).strip()
