"""Ask single questions end to end and show evidence summary + raw model output."""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness")); sys.path.insert(0, str(ROOT / "submission"))
sys.stdout.reconfigure(encoding="utf-8")
from run_eval import load_snapshots
from dupaco_kb.engine import Engine
from dupaco_kb.temporal import resolve

eng = Engine()
for s in load_snapshots(ROOT / "snapshots"):
    eng.ingest(s)
orig = eng.llm.ask
def spy(msgs):
    out = orig(msgs)
    print("   RAW:", out["_raw"][:700])
    return out
eng.llm.ask = spy
for q in sys.argv[1:]:
    as_of = None
    if "||" in q:
        q, as_of = q.split("||")
    sidx, target, how = resolve(eng.store.snaps, as_of, q)
    ev = eng.retrieve(q, sidx)
    print("=" * 90, "\nQ:", q, as_of or "")
    for i, e in enumerate(ev, 1):
        v = eng.store.versions[eng.store.chunks[e["cid"]].vid]
        print(f"   [{i}] {v.ptype:10s} {v.pub_date or '':10s} {e['urls'][0][:95]} len={len(e['text'])}")
    t = time.time()
    r = eng.answer(q, as_of)
    print(f"   FINAL ({time.time()-t:.1f}s):", r["text"][:600], "| confident:", r["confident"])
