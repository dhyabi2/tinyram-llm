#!/usr/bin/env python3
"""100x feasibility map. 100x is a ratio, so it depends on the baseline as much as on the system: F / B >= 100  <=>  B <= F / 100.
For each published stack (techniques stacked in the order published work introduced them) this prints the decode speed F it reaches on 8/16/32 GB CPU-only machines
(conservative / likely / optimistic parameters from combo_model.py) and the baseline speed B* below which that stack is >= 100x faster.
Then it lists which naive baselines (modelled or measured) fall under B*. Parameters beyond the measured ones are estimates; quality is NOT in F (see the risk score)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import combo_model as cm
BASE = dict(K=10, bits=4.3, lowrank=False, route=False, overlap=False, spec=0, gpu=0, lut=False, exit=1.0)
STACKS = [   # name, settings, published work behind each addition
 ("explicit large reads + LRU (our base path)", {}, "Flash-MoE, BigMoeOnEdge, SSD-LLaMA, moe-offload, LLM in a flash"),
 ("+ overlap with next-layer prediction", dict(overlap=True), "ProMoE, ExpertFlow, Pre-gated MoE"),
 ("+ cache-aware routing", dict(overlap=True, route=True), "Skliar et al. cache-conditional experts, ReMoE"),
 ("+ exact speculative decoding k=4", dict(overlap=True, route=True, spec=4), "MoE-Spec, EcoSpec, SpecMoEOff, SP-MoE"),
 ("+ 3-bit experts (mixed precision)", dict(overlap=True, route=True, spec=4, bits=3.0), "HOBBIT, MC-MoE, SliceMoE"),
 ("+ lookup-table low-bit kernels", dict(overlap=True, route=True, spec=4, bits=3.0, lut=True), "T-MAC, AQLM (gain over Q4_K unmeasured)"),
 ("+ top-6 experts", dict(overlap=True, route=True, spec=4, bits=3.0, lut=True, K=6), "NAEE, LExI, REAP"),
 ("+ top-4 experts", dict(overlap=True, route=True, spec=4, bits=3.0, lut=True, K=4), "NAEE, MC-MoE"),
 ("+ skip 25% of layers", dict(overlap=True, route=True, spec=4, bits=3.0, lut=True, K=4, exit=0.75), "LayerSkip, ShortGPT, Dr.LLM"),
 ("+ 12 GB GPU tier (hardware)", dict(overlap=True, route=True, spec=4, bits=3.0, lut=True, K=4, exit=0.75, gpu=12), "KTransformers, Fiddler, HybriMoE"),
 ("+ 24 GB GPU tier (hardware)", dict(overlap=True, route=True, spec=4, bits=3.0, lut=True, K=4, exit=0.75, gpu=24), "KTransformers, Fiddler, HybriMoE"),
]
# naive baselines (tok/s): modelled plain mmap per machine, plus sensitivity values for a thrashing baseline (swap, or mmap with a much smaller cache than the footprint)
def baselines(m):
    sc = cm.sc_at(1); return [("modelled plain mmap", cm.mmap_path(m, sc)), ("thrashing baseline 0.10", 0.10), ("thrashing baseline 0.05", 0.05), ("swap-class baseline 0.02", 0.02), ("swap-class baseline 0.01", 0.01)]
if __name__ == "__main__":
    for m in cm.CACHE_GIB:
        print(f"\n=== {m} machine (CPU-only unless marked hardware); compute ceiling {cm.G_CPU/cm.FLOP_TOT:.2f} tok/s")
        print(f"{'stack':<46}{'risk':>5}{'F cons':>8}{'F likely':>9}{'F opt':>7}{'B* for 100x (likely)':>22}   published work")
        for name, st, pub in STACKS:
            c = {**BASE, **st}; f = [cm.tok_s(c, m, cm.sc_at(i)) for i in (0, 1, 2)]
            print(f"{name:<46}{cm.risk(c):>5}{f[0]:>8.2f}{f[1]:>9.2f}{f[2]:>7.2f}{f[1]/100:>18.4f} tok/s   {pub}")
        print("  baselines:", " | ".join(f"{n} {b:.3f}" for n, b in baselines(m)))
        print("  stacks that reach >= 100x on each baseline (likely F):")
        for n, b in baselines(m):
            ok = [s[0] for s in STACKS if cm.tok_s({**BASE, **s[1]}, m, cm.sc_at(1)) / b >= 100]
            print(f"    vs {n:<28} {b:6.3f}: need F >= {100*b:5.1f} tok/s -> {len(ok)} stack(s)" + (f", first: {ok[0]}" if ok else ""))
