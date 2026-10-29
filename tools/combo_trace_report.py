#!/usr/bin/env python3
"""Writes a Markdown report of the trace-driven combination runs. usage: combo_trace_report.py OUT.md   (about 1 minute)"""
import collections, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import combo_trace_sim as cs
def main():
    out = sys.argv[1]; rows = cs.collect(); L = []; w = L.append
    w("# Trace-driven combinations (measured cache behaviour, projected speed)\n")
    w(f"{len(rows)} runs = 3 recorded Qwen3-30B router traces x 3 cache policies x 4 expert-selection rules x 3 bit widths x 3 speculation settings x 2 layer-skip settings x overlap on/off x 3 RAM sizes. Hit rates are simulated on the real traces; speeds apply the time model of combo_model.py (scenario numbers there stay estimates). Quality: only measured OLMoE perplexity losses are used; `?` means unmeasured.\n")
    for ram in cs.RAMS:
        m = cs.table(rows); base = m[("lru", "full", 4.3, 0, 0.0, False, ram)]
        w(f"\n## {ram} GB machine - base path {base:.2f} tok/s (likely scenario, mean of 3 traces)\n")
        g, pairs = cs.interactions(rows, ram)
        w("Single levers (x base): " + ", ".join(f"{k} {v:.2f}" for k, v in g.items()) + "\n")
        sup = sorted([p for p in pairs if p[4] > 1.03], key=lambda p: -p[4]); sub = sorted([p for p in pairs if p[4] < 0.97], key=lambda p: p[4])
        w(f"Pairs: {len(sup)} better than the product of their parts (>3%), {len(sub)} worse (<-3%), {len(pairs)-len(sup)-len(sub)} about equal.\n")
        if sup: w("Best synergies: " + "; ".join(f"{a}+{b} {c:.2f} vs product {p:.2f}" for a, b, c, p, r in sup[:6]) + "\n")
        if sub: w("Worst interference: " + "; ".join(f"{a}+{b} {c:.2f} vs product {p:.2f}" for a, b, c, p, r in sub[:6]) + "\n")
        agg = collections.defaultdict(list)
        for r in rows:
            if r["ram"] == ram: agg[cs.label(r)].append(r)
        tab = {k: dict(likely=sum(x["likely"] for x in v) / len(v), cons=sum(x["cons"] for x in v) / len(v), opt=sum(x["opt"] for x in v) / len(v), ppl=v[0]["ppl"], hit=sum(x["hit"] for x in v) / len(v)) for k, v in agg.items()}
        for name, f in (("Quality-safe (no lossy lever)", lambda v: v["ppl"] == 0.0), ("Measured perplexity loss <= 12%", lambda v: v["ppl"] is not None and v["ppl"] <= 0.12), ("Any (loss unmeasured or large)", lambda v: True)):
            w(f"\n**{name}** (top 8)\n\n| combination | cons | likely | opt | x base | hit | ppl |\n|---|---|---|---|---|---|---|")
            for k, v in sorted(((k, v) for k, v in tab.items() if f(v)), key=lambda kv: -kv[1]["likely"])[:8]: w(f"| {k} | {v['cons']:.2f} | {v['likely']:.2f} | {v['opt']:.2f} | {v['likely']/base:.2f} | {v['hit']:.2f} | {'?' if v['ppl'] is None else '+%d%%' % round(v['ppl']*100)} |")
        best_safe = max((v["likely"] for v in tab.values() if v["ppl"] == 0.0)); w(f"\nBest quality-safe gain over base: x{best_safe/base:.2f}.")
    w("\n## Reading guide\n- A gain here is model output on measured hit rates, not a measured tok/s on the 125B.\n- Speculation acceptance, overlap share and batch efficiency are estimates (combo_model.py); the cache-aware 6-of-8 rule and layer skipping have no measured quality cost.\n- Policy choice (LRU vs segmented LRU vs held-out pinning) moves throughput by under 3%.\n")
    open(out, "w").write("\n".join(L) + "\n"); print("wrote", out)
if __name__ == '__main__': main()
