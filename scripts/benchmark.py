"""Replay eval questions + paraphrases (realistic repeat traffic) through baseline and optimized.
Usage: python -m scripts.benchmark [--repeats 2]"""
import argparse, json, random
import numpy as np
from app.config import ROOT, load_config
from app.service import Service


def workload(repeats: int, seed=7):
    ev = json.load(open(ROOT / "data/eval_set.json"))
    rnd = random.Random(seed)
    items = [(e["question"], e["expected"]) for e in ev]
    paras = [(p, e["expected"]) for e in ev for p in e["paraphrases"]]
    out = list(items)                       # first sight of each question (cold)
    for _ in range(repeats):                # then repeat traffic: paraphrases and exact repeats
        batch = paras + items
        rnd.shuffle(batch)
        out += batch
    return out


def stats(db, mode):
    rows = db.query("SELECT * FROM requests WHERE mode=? ORDER BY id", (mode,))
    lat = np.array([r["latency_ms"] for r in rows]); q = [r["quality"] for r in rows if r["quality"] is not None]
    routes = {}
    for r in rows: routes[r["route"]] = routes.get(r["route"], 0) + 1
    return dict(n=len(rows), cost_per_req=float(np.mean([r["cost"] for r in rows])),
                total_cost=float(sum(r["cost"] for r in rows)),
                hit_rate=float(np.mean([r["cache_hit"] for r in rows])),
                escalation_rate=float(np.mean([r["escalated"] for r in rows])),
                p50=float(np.percentile(lat, 50)), p95=float(np.percentile(lat, 95)),
                quality=float(np.mean(q)) if q else float("nan"), routes=routes)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repeats", type=int, default=2); a = ap.parse_args()
    svc = Service(load_config())
    svc.db.execute("DELETE FROM requests WHERE mode IN ('baseline','optimized')")
    svc.cache.clear()
    wl = workload(a.repeats)
    for q, exp in wl: svc.answer(q, mode="baseline", expected=exp)
    for q, exp in wl: svc.answer(q, mode="optimized", expected=exp)
    b, o = stats(svc.db, "baseline"), stats(svc.db, "optimized")
    red = 1 - o["total_cost"] / b["total_cost"]
    md = f"""# Benchmark ({b['n']} requests per run, threshold={svc.cfg.threshold}, embedder={getattr(svc.embedder,'name','?')}, llm={svc.cfg.llm_mode})

| Metric | Baseline (always large) | Optimized | Change |
|---|---|---|---|
| Cost / request | ${b['cost_per_req']:.5f} | ${o['cost_per_req']:.5f} | **{red:.1%} lower** |
| Cache hit rate | - | {o['hit_rate']:.1%} | |
| Escalation rate | - | {o['escalation_rate']:.1%} | |
| p50 latency | {b['p50']:.1f} ms | {o['p50']:.1f} ms | {o['p50']/b['p50']-1:+.1%} |
| p95 latency | {b['p95']:.1f} ms | {o['p95']:.1f} ms | {o['p95']/b['p95']-1:+.1%} |
| Quality (key-fact recall) | {b['quality']:.3f} | {o['quality']:.3f} | {o['quality']-b['quality']:+.3f} |

Route split (optimized): {o['routes']}
"""
    print(md); (ROOT / "results/benchmark.md").write_text(md)


if __name__ == "__main__":
    main()
