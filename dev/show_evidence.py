"""Print the evidence the engine would give Claude for a question (no LLM call)."""
import sys
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
full = "--full" in sys.argv
for q in [a for a in sys.argv[1:] if a != "--full"]:
    sidx, target, how = resolve(eng.store.snaps, None, q)
    ev = eng.retrieve(q, sidx)
    print("=" * 100, "\nQ:", q, "| snapshot", sidx, how)
    if full:
        print(eng._prompt(q, sidx, target, how, ev))
    else:
        for i, e in enumerate(ev, 1):
            v = eng.store.versions[eng.store.chunks[e["cid"]].vid]
            print(f"[{i}] {e['score']:.1f} {v.ptype:10s} {v.pub_date or '':10s} {e['urls'][0][:90]}  whole={e['whole']} len={len(e['text'])}")
