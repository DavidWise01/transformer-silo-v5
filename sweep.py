#!/usr/bin/env python3
"""Robustness check: is the router's edge a seed-0 fluke, or does it hold?
Runs the full train (router + both baselines, to convergence) across several
seeds and reports the spread. Honest artifacts state the range, not one lucky
number. Run: python sweep.py"""
from __future__ import annotations
import json
from train import run

SEEDS = [0, 1, 2, 3, 4]
rows = []
for s in SEEDS:
    r = run(seed=s)
    rows.append({
        "seed": s,
        "silo": r["silo_only"]["acc"],
        "plain": r["plain_only"]["acc"],
        "router": r["router"]["acc"],
        "router_minus_plain": round(r["router"]["acc"] - r["plain_only"]["acc"], 4),
        "router_pairs": r["router"]["avg_pairs"],
        "plain_pairs": r["plain_only"]["avg_pairs"],
        "compute_saved": round(1 - r["router"]["avg_pairs"] / r["plain_only"]["avg_pairs"], 4),
        "bag_to_silo": r["router"]["route_bag_to_silo"],
        "order_to_plain": round(1 - r["router"]["route_order_to_silo"], 4),
    })
    print(f"seed {s}: silo={rows[-1]['silo']:.3f} plain={rows[-1]['plain']:.3f} "
          f"router={rows[-1]['router']:.3f}  (r-p={rows[-1]['router_minus_plain']:+.3f})  "
          f"pairs {rows[-1]['router_pairs']} vs {rows[-1]['plain_pairs']} "
          f"({rows[-1]['compute_saved']*100:.0f}% less)  route bag->silo {rows[-1]['bag_to_silo']:.2f} order->plain {rows[-1]['order_to_plain']:.2f}")

rp = [x["router_minus_plain"] for x in rows]
cs = [x["compute_saved"] for x in rows]
summary = {
    "seeds": SEEDS,
    "rows": rows,
    "router_minus_plain": {"min": min(rp), "max": max(rp), "mean": round(sum(rp)/len(rp), 4)},
    "compute_saved": {"min": min(cs), "max": max(cs), "mean": round(sum(cs)/len(cs), 4)},
    "router_beats_plain_every_seed": all(x > 0 for x in rp),
    "router_ge_plain_every_seed": all(x >= 0 for x in rp),
}
with open("sweep.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nrouter - plain: min {min(rp):+.3f}  mean {sum(rp)/len(rp):+.3f}  max {max(rp):+.3f}")
print(f"compute saved:  {min(cs)*100:.0f}-{max(cs)*100:.0f}%  (mean {sum(cs)/len(cs)*100:.0f}%)")
print(f"router beats plain on every seed: {summary['router_beats_plain_every_seed']}  "
      f"(>= plain: {summary['router_ge_plain_every_seed']})")
