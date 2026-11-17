"""Claude call that turns retrieved evidence into a member-facing answer (JSON)."""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import sys
import threading
from pathlib import Path

MODEL_CHAIN = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]   # step-down order if unavailable
_POOL = cf.ThreadPoolExecutor(max_workers=8, thread_name_prefix="kb-llm")
HEDGE_AFTER = float(os.getenv("KB_HEDGE_AFTER", "3.0"))   # seconds before a duplicate request is sent

DEFAULT_MODEL = "claude-opus-5-5"   # with fast mode: best accuracy AND p95 < 2 s in our A/B (see NOTES.md)

SYSTEM = """You are the member-service assistant of a credit union. You answer members' questions using ONLY the website excerpts in the user message. Answers are graded strictly: a single stale, wrong, or invented number makes the whole answer wrong, and so does leaving out a value that was asked for.

DECIDING WHAT IS TRUE
1. The excerpts show the website as it stood on the "website date". Answer as of that date.
2. When excerpts disagree about the same fact, trust, in this order: the OFFICIAL RATE SHEET and official product pages; then the most recent "Rates are effective as of" date; then everything else. Promotional/landing pages, calculator pages, blog articles, PDFs and old news releases often still show outdated numbers. Never use their numbers when an official page states the same fact.
3. Facts that only appear in news releases or articles (branch counts, awards, closures, events, branch hours): use the MOST RECENT dated release. Older releases describe the past.
4. Expired promotions and closed branches are not current.

READING TABLES
- Match every qualifier of the question exactly: product name (Share Savings, Health Savings Account and money market accounts are different products), term (48 vs 60 months), balance tier, loan-amount tier, vehicle model-year band, jumbo vs regular, special vs standard, personal vs business, purchase vs refinance.
- A vehicle model year belongs to the band that contains it (2025 is in a "2024-27 Model Year" band).
- Distinguish APY from dividend rate and APR from interest rate. Quote the metric asked for. If the member just says "rate": for deposits give the APY with the dividend rate next to it (e.g. "2.25% APY (2.23% dividend rate)"); for loans whose table shows both an interest rate and an APR (e.g. mortgages), give both as "rate / APR"; otherwise give the APR.

"Checked facts" (when present) were verified by comparing pages and crawls: follow them. For questions about how something changed over time, use the "History" lines (values by crawl date) and state the old value, the new value and roughly when it changed (between which crawl dates).

WRITING THE ANSWER
- Answer directly in 1-3 sentences: the current value(s) with their qualifiers (term, tier, metric) and the effective date when shown.
- Be concrete: include the specifics the excerpts give (amounts, product names, phone numbers, exact URLs) rather than generic wording. When the answer refers to an area (charter area, service area, where we operate), also name the places the excerpts give for it (states, cities or counties).
- Broad or list questions ("what certificate rates do you have", "what credit cards do you offer", "what are your rates"): list EVERY matching product/term with its value. Never ask which one they want.
- General questions about a tiered product (e.g. "mortgage rates"): state the smallest loan-amount tier FIRST (e.g. the "$100,000 or less" row), then the other main terms of the same official purchase table (e.g. 20- and 15-year); leave out refinance rows unless asked.
- "Do you have / do you offer" questions: name every option or variant the site lists for it (e.g. new loans and refinancing, personal and business versions).
- Yes/no questions about a value ("does any savings tier pay X?"): answer yes/no for the product the member named, and state that product's actual current values (e.g. its top tier and rate). A generic word like "savings" or "checking" means the credit union's basic account of that type (e.g. Share Savings). Then mention which other products, if any, do have that value.
- How-to questions (join, apply, update contact information, find a branch): give the concrete steps, the exact link URLs (links appear as [text](url) - copy the url of the relevant action link, e.g. an application or sign-up link) and the phone number shown.
- Link questions: give the product's own page as the site links it (e.g. the "See details" / "Compare rates" link next to that product's rates, such as a URL ending in "#rates"), and also the general rates page if there is one.
- General hours questions (no particular branch named): if the excerpts give hours for one or more branches, state them - list EVERY distinct schedule found (with its branch, including Saturday hours) - say hours can vary by branch, and end with an offer such as "Tell me which branch and I can give its exact hours." Do not refuse just because hours differ by location.
- Hours (or any other detail) of a SPECIFIC branch the member names: give them only if the excerpts state them for that branch. Otherwise the status is "not_listed" - never substitute another branch's hours.
- Recurring things (awards, rankings, recognitions, counts that change over time): state the latest status exactly as the most recent dated page words it (e.g. "the most recent release says this was the Nth time ..."). Do not also recite what older pages said (earlier counts, earlier ordinals, older years) - those read as outdated.
- Figures tied to a period ("last year", "in 2023"): list every yearly figure the site reports for that same statistic, each with the year it refers to (most recent first), even if the exact period asked about is not covered. Use status "partial" in that case, not "not_listed".
- Only state numbers that are part of the answer. Never mention outdated values, the values of other products or tiers that were not asked about, or numbers you computed yourself.
- Stay on the product asked about: when the question names one specific product (e.g. "the 10-month jumbo CD special"), do not add its IRA, business or other variants. Broad questions ("what credit cards do you offer", "rates for the credit cards", "what are your certificate rates") are different: list every product of that type the site shows, including business versions.
- Never invent numbers, product names, phone numbers or URLs. Copy them exactly from the excerpts.
- status: "answered" when the excerpts answer the question; "partial" when they give useful but incomplete or not fully current information (hours for some branches, figures for earlier years, a general contact method) - then give that information plainly; "not_listed" when nothing in the excerpts answers the question - in particular when the member asks for a specific rate, fee or term of a product that is not published, or a detail of a specific branch that is not given, even if related general information exists. Never offer other products' rates as a substitute.
- For "not_listed": say briefly that it isn't listed on the website and the member should contact the credit union. That answer must contain NO numbers at all (no phone numbers, no rates, no alternatives from other products).
- Never ask the member for information (ZIP code, member number, which term...). If a question repeats an earlier one, just answer it again.
- Do not mention "excerpts", "sources", "provided information", "checked facts", "these facts" or "context". Speak as the credit union's assistant.
- Never explain conflicting or outdated pages in the answer (that is handled separately).
- Be brief, because members read answers on a phone: one sentence for a single value; for lists use compact "name: value" items separated by semicolons. Do not add minimum balances, payment examples or disclaimers unless the member asks for them.

Return only the JSON object."""

SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["answered", "partial", "not_listed"]},
        "answer": {"type": "string"},
        "source_ids": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["status", "answer", "source_ids"],
    "additionalProperties": False,
}


def _load_dotenv() -> None:
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"):
        return
    here = Path(__file__).resolve()
    for d in [here.parent, *here.parents][:5]:
        f = d / ".env"
        if f.is_file():
            for line in f.read_text(encoding="utf-8").splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return


class LLM:
    def __init__(self):
        _load_dotenv()
        import anthropic  # imported lazily so ingest works even before the key is set
        self.anthropic = anthropic
        self.model = os.getenv("KB_MODEL", DEFAULT_MODEL)
        self.effort = os.getenv("KB_EFFORT", "low")
        if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
            print("[kb] WARNING: ANTHROPIC_API_KEY is not set - answers need the Claude API (see NOTES.md)",
                  file=sys.stderr)
        # the harness allows 60 s per answer: keep every call well inside that budget
        self.client = anthropic.Anthropic(timeout=float(os.getenv("KB_TIMEOUT", "20")), max_retries=1)
        self.use_fast = os.getenv("KB_FAST", "1") == "1"      # ON unless explicitly switched off
        self.use_fallbacks = True
        self._lock = threading.Lock()

    def _kwargs(self) -> dict:
        kw = {"output_config": {"format": {"type": "json_schema", "schema": SCHEMA}}}
        thinking_off = os.getenv("KB_THINKING", "on" if "haiku" not in self.model else "off") == "off"
        if "haiku" in self.model:
            if thinking_off:
                kw["thinking"] = {"type": "disabled"}
            else:
                kw["output_config"]["effort"] = self.effort
        elif "sonnet-5-5" in self.model and thinking_off:
            kw["thinking"] = {"type": "between_tools"}      # Sonnet 5.5's "thinking off"
            kw["output_config"]["effort"] = self.effort
        else:
            kw["output_config"]["effort"] = self.effort
        return kw

    def ask(self, messages: list[dict]) -> dict:
        """Hedged call: if the first request is unusually slow (API latency spike), send an
        identical second request and take whichever good reply arrives first."""
        first = _POOL.submit(self._call, messages)
        try:
            return first.result(timeout=HEDGE_AFTER)
        except cf.TimeoutError:
            pass
        second = _POOL.submit(self._call, messages)
        pending = {first, second}
        error = None
        while pending:
            done, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            for f in done:
                if f.exception() is None:
                    return f.result()
                error = f.exception()
        raise error

    def _request(self, messages: list[dict], allow_fast: bool = True):
        req = dict(model=self.model, max_tokens=2000,
                   system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                   messages=messages, **self._kwargs())
        body, headers = {}, {}
        if self.use_fallbacks and ("opus-5-5" in self.model or "sonnet-5-5" in self.model):
            # server-side refusal fallback: a declined request is retried on a fallback model
            body["fallbacks"] = "default"
            headers["anthropic-beta"] = "server-side-fallback-2026-07-01"
        if allow_fast and self.use_fast and "opus-5-5" in self.model:
            # fast mode (Opus 5.5) is the top-level "speed" field; this SDK version has no named argument for it
            body["speed"] = "fast"
        return req, body, headers

    def _degrade(self, err) -> bool:
        """The key/account cannot use a requested feature (400/403/404): step down once, for the
        rest of the run - fast mode, then the refusal-fallback beta, then a smaller model."""
        with self._lock:
            if self.use_fast and "opus-5-5" in self.model:
                self.use_fast, what = False, "fast mode unavailable -> standard speed"
            elif self.use_fallbacks:
                self.use_fallbacks, what = False, "refusal fallback unavailable -> off"
            elif self.model in MODEL_CHAIN and MODEL_CHAIN.index(self.model) + 1 < len(MODEL_CHAIN):
                old = self.model
                self.model = MODEL_CHAIN[MODEL_CHAIN.index(self.model) + 1]
                what = f"model {old} unavailable -> {self.model}"
            else:
                return False
        print(f"[kb] {what} ({type(err).__name__} {getattr(err, 'status_code', '')})", file=sys.stderr)
        return True

    def _call(self, messages: list[dict]) -> dict:
        for _ in range(len(MODEL_CHAIN) + 3):
            req, body, headers = self._request(messages)
            try:
                resp = self.client.messages.create(**req, extra_body=body or None, extra_headers=headers or None)
                break
            except self.anthropic.RateLimitError:
                if "speed" not in body:
                    raise
                # fast mode has its own capacity limit: answer this one at standard speed
                req, body, headers = self._request(messages, allow_fast=False)
                resp = self.client.messages.create(**req, extra_body=body or None, extra_headers=headers or None)
                break
            except self.anthropic.APIStatusError as e:
                if e.status_code not in (400, 403, 404) or not self._degrade(e):
                    raise
        else:
            raise RuntimeError("no usable model configuration")
        if resp.stop_reason == "refusal":
            raise RuntimeError("model refused")
        text = next((b.text for b in resp.content if b.type == "text"), "")
        data = json.loads(text)
        data["confident"] = data.get("status") != "not_listed"
        data["_raw"] = text
        return data
