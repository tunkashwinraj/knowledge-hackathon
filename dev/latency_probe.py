"""Where does answer time go?  Local work (retrieve + hints + prompt) vs the Claude call.
Usage: python dev/latency_probe.py [--llm N]   (N = number of questions to also send to Claude)"""
import json, sys, time, statistics
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
items = [json.loads(l) for f in ("eval/public_eval.jsonl", "dev/paraphrase_eval.jsonl")
         for l in open(ROOT / f, encoding="utf-8") if l.strip()]
local, sizes = [], []
prompts = []
for it in items:
    t = time.time()
    sidx, target, how = resolve(eng.store.snaps, it.get("as_of"), it["question"])
    ev = eng.retrieve(it["question"], sidx)
    hints = eng._hints(it["question"], sidx, ev)
    p = eng._prompt(it["question"], sidx, target, how, ev, hints)
    local.append(time.time() - t)
    sizes.append(len(p))
    prompts.append((it["id"], p))
q = sorted(local)
print(f"local work per question: p50 {statistics.median(local)*1000:.0f} ms, p95 {q[int(.95*len(q))]*1000:.0f} ms, "
      f"max {max(local)*1000:.0f} ms;  prompt chars p50 {statistics.median(sizes):.0f}, max {max(sizes)}")
if "--llm" in sys.argv:
    n = int(sys.argv[sys.argv.index("--llm") + 1])
    llm = eng.llm
    rows = []
    for qid, p in prompts[:n]:
        t = time.time()
        llm._call([{"role": "user", "content": p}])
        rows.append(time.time() - t)
    d = sorted(rows)
    print(f"LLM {llm.model} fast={__import__('os').getenv('KB_FAST','0')} thinking={__import__('os').getenv('KB_THINKING','default')}: "
          f"p50 {statistics.median(d):.2f}s p95 {d[min(len(d)-1,int(.95*len(d)))]:.2f}s max {d[-1]:.2f}s")
