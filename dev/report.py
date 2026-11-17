"""Summarize a results folder: headline numbers + every question that lost points."""
import json, sys
sys.stdout.reconfigure(encoding="utf-8")
out, evalf = sys.argv[1], sys.argv[2]
rep = json.load(open(f"{out}/report.json"))
print(out, {k: rep[k] for k in ("qa_score", "stale_trap_hits", "errors", "final_score")},
      "speed", rep["speed"], "consistency", rep["consistency"]["score"])
for k, v in rep["by_category"].items():
    print(f"  {k:22s} {v['score']:.2f} n={v['n']}")
items = {json.loads(l)["id"]: json.loads(l) for l in open(evalf, encoding="utf-8") if l.strip()}
for l in open(f"{out}/per_question.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r["correctness"] < 0.99:
        it = items[r["id"]]
        print(f"== {r['id']} [{r['category']}] score={r['score']} missed={r['missed']} trap={r['trap_hits']} lat={r['latency_s']}")
        print("   Q:", it["question"], "| as_of:", it.get("as_of"))
        print("   A:", r["answer"]["text"][:450])
