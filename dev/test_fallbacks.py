"""Offline test of the API step-down ladder (no network, no cost).

Simulates accounts that cannot use fast mode / Opus 5.5 / the fallback beta, and a fast-mode
rate limit, and checks that answers still come back with the expected configuration.
Usage:  python dev/test_fallbacks.py
"""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")

import anthropic  # noqa: E402
import httpx  # noqa: E402
from dupaco_kb.llm import LLM  # noqa: E402

REQ = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def err(cls, code):
    return cls(f"simulated {code}", response=httpx.Response(code, request=REQ), body=None)


def fake_create(rules, calls):
    def create(**kw):
        body = kw.get("extra_body") or {}
        calls.append((kw["model"], body.get("speed"), "fallbacks" in body))
        for cond, exc in rules:
            if cond(kw["model"], body):
                raise exc
        text = json.dumps({"status": "answered", "answer": "ok", "source_ids": []})
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=text)])
    return create


def run(name, rules, expect_model, expect_fast, n=3):
    llm = LLM()
    calls = []
    llm.client = SimpleNamespace(messages=SimpleNamespace(create=fake_create(rules, calls)))
    for _ in range(n):
        out = llm._call([{"role": "user", "content": "hi"}])
        assert out["answer"] == "ok"
    last = calls[-1]
    ok = last[0] == expect_model and (last[1] == "fast") == expect_fast
    print(f"{'PASS' if ok else 'FAIL'}  {name:42s} final call: model={last[0]} speed={last[1]} "
          f"fallbacks={last[2]}  (total API calls for {n} answers: {len(calls)})")
    return ok


results = [
    run("normal account", [], "claude-opus-5-5", True),
    run("fast mode not enabled (400)",
        [(lambda m, b: b.get("speed") == "fast", err(anthropic.BadRequestError, 400))],
        "claude-opus-5-5", False),
    run("fast mode forbidden (403)",
        [(lambda m, b: b.get("speed") == "fast", err(anthropic.PermissionDeniedError, 403))],
        "claude-opus-5-5", False),
    run("fast mode rate-limited (429)",
        [(lambda m, b: b.get("speed") == "fast", err(anthropic.RateLimitError, 429))],
        "claude-opus-5-5", False),
    run("no Opus 5.5 access (404)",
        [(lambda m, b: m == "claude-opus-5-5", err(anthropic.NotFoundError, 404))],
        "claude-sonnet-5-5", False),
    run("no Opus, no fallback beta (400/404)",
        [(lambda m, b: m == "claude-opus-5-5", err(anthropic.NotFoundError, 404)),
         (lambda m, b: "fallbacks" in b, err(anthropic.BadRequestError, 400))],
        "claude-sonnet-5-5", False),
]
print("ALL PASS" if all(results) else "SOME FAILED")
