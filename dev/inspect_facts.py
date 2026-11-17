"""Dev checks for the data layer and retrieval (no LLM calls).

Usage (from the kit root):  python dev/inspect_facts.py [--recall] [--facts] [--conflicts]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "submission"))
sys.stdout.reconfigure(encoding="utf-8")

from run_eval import load_snapshots  # noqa: E402
from scoring import contains  # noqa: E402
from dupaco_kb.engine import Engine  # noqa: E402
from dupaco_kb.temporal import resolve  # noqa: E402


def build(snap_dir):
    t = time.time()
    eng = Engine()
    for s in load_snapshots(snap_dir):
        eng.ingest(s)
    st = eng.store
    print(f"ingest {time.time()-t:.1f}s  versions={len(st.versions)} chunks={len(st.chunks)} facts={len(st.facts)}")
    for i in range(1, len(st.snaps)):
        changed = sum(1 for url, hist in st.by_url.items()
                      if any(i in st.versions[v].snaps for v in hist) and any(i - 1 in st.versions[v].snaps for v in hist)
                      and st.version_at(url, i).vid != st.version_at(url, i - 1).vid)
        print(f"  snapshot {i-1}->{i}: pages whose cleaned text changed = {changed}")
    return eng


def timeline(eng, needle_label, metric=None, product=None):
    st = eng.store
    rows = {}
    for f in st.facts:
        if needle_label.lower() not in f.label.lower():
            continue
        if metric and f.metric != metric:
            continue
        if product and f.product != product:
            continue
        v = st.versions[f.vid]
        for s in sorted(v.snaps):
            rows.setdefault((f.label, f.metric, v.url.replace("https://www.dupaco.com", "")), {})[s] = f.value_str
    for (label, metric, url), per in sorted(rows.items()):
        line = "  ".join(f"S{s+1}:{per.get(s, '-'):>8}" for s in range(len(st.snaps)))
        print(f"  {label[:60]:60s} {metric:4s} {url[:38]:38s} {line}")


def recall(eng, eval_path, k):
    items = [json.loads(l) for l in Path(eval_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    ok = 0
    for it in items:
        sidx, _, _ = resolve(eng.store.snaps, it.get("as_of"), it["question"])
        ev = eng.retrieve(it["question"], sidx, k=k)
        text = "\n".join(e["text"] for e in ev)
        must = it.get("must_include", [])
        hit = [m for m in must if contains(text, m)]
        good = len(hit) == len(must)
        ok += good
        if not good:
            print(f"  MISS {it['id']:8s} {it['question'][:70]:70s} missing={[m for m in must if m not in hit]}")
            for e in ev[:4]:
                print(f"        - {e['urls'][0][:100]}  score={e['score']:.1f}")
    print(f"retrieval recall@{k}: {ok}/{len(items)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", default=str(ROOT / "snapshots"))
    ap.add_argument("--eval", default=str(ROOT / "eval" / "public_eval.jsonl"))
    ap.add_argument("--recall", action="store_true")
    ap.add_argument("--facts", action="store_true")
    ap.add_argument("--conflicts", action="store_true")
    ap.add_argument("-k", type=int, default=8)
    a = ap.parse_args()
    eng = build(Path(a.snapshots))
    if a.facts:
        for lab, metric, prod in [("30-Year Fixed Conventional $832,750", "RATE", None),
                                  ("10 Month Fixed Jumbo", "APY", None),
                                  ("1st Rate Checking", "APY", None), ("$5,000", "APY", "checking"),
                                  ("60 Months", "APR", "auto loan"),
                                  ("Share Savings", "APY", None), ("Bright Savings", "APY", None)]:
            print(f"--- {lab} {metric} {prod or ''}")
            timeline(eng, lab, metric, prod)
    if a.recall:
        recall(eng, a.eval, a.k)
    if a.conflicts:
        cs = eng.conflicts()
        print(f"conflicts: {len(cs)}")
        for c in cs:
            print(f"  [{c['_score']:3d}] {c['topic'][:90]:90s} values={c['values']} -> {c['resolution']}")


if __name__ == "__main__":
    main()
