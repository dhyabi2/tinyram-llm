#!/usr/bin/env python3
"""How much does more RAM raise the maximum tokens/s? Sweeps RAM from 4 to 128 GB (CPU-only, likely parameters).
RAM enters only through the cache hit rate (measured LRU curve in token-footprints). Two corrections to the measured curve are shown:
  'trace'  = the curve as measured on a 299-token trace (its plateau at 0.96 is the cold-start misses of a short run);
  'steady' = long-run hit rate: the 3.3% cold misses vanish once the cache holds the working set (>= 7.5 footprints) [E].
The compute ceiling depends on the CPU, not on RAM: the sweep is repeated for 55 (measured laptop), 110 and 220 GFLOP/s decode rates."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import combo_model as cm
BASE = dict(K=10, bits=4.3, lowrank=False, route=False, overlap=False, spec=0, gpu=0, lut=False, exit=1.0)
RAMS = (4, 6, 8, 12, 16, 24, 32, 48, 64, 80, 96, 128); COLD = 0.033
sc = cm.sc_at(1)
def hit_steady(k): h = cm.lru(k); return min(0.999, h + COLD) if k >= 7.5 else h
orig_lru = cm.lru
def run(G, steady, stack, rate=cm.R_EXPL):
    cm.G_CPU = G; cm.lru = (lambda k: hit_steady_wrap(k)) if steady else orig_lru
    out = [cm.tok_s({**BASE, **stack}, cm.cache_for_ram(r), sc, rate=rate) for r in RAMS]
    cm.lru = orig_lru; return out
def hit_steady_wrap(k):
    h = orig_lru(k); return min(0.999, h + COLD) if k >= 7.5 else h
if __name__ == "__main__":
    print(f"expert bytes per token {cm.EXP_GB:.2f} GB; explicit read rate {cm.R_EXPL} GB/s; mmap {cm.R_MMAP} GB/s; file 69 GB\n")
    for G in (55.0, 110.0, 220.0):
        ceil = G / cm.FLOP_TOT
        tr = run(G, False, {}); ss = run(G, True, {}); ov = run(G, True, dict(overlap=True)); mm = run(G, True, {}, rate=cm.R_MMAP)
        print(f"=== CPU decode rate {G:.0f} GFLOP/s -> compute ceiling {ceil:.2f} tok/s")
        print(f"{'RAM GB':>7}{'cache GiB':>10}{'hit(trace)':>11}{'base (trace)':>14}{'base (steady)':>15}{'+overlap (steady)':>19}{'plain mmap (steady)':>21}{'base / mmap':>13}{'% of ceiling':>14}")
        for i, r in enumerate(RAMS):
            cache = cm.cache_for_ram(r); k = cache * 1.074 / cm.EXP_GB
            print(f"{r:>7}{cache:>10.1f}{orig_lru(k):>11.2f}{tr[i]:>14.2f}{ss[i]:>15.2f}{ov[i]:>19.2f}{mm[i]:>21.2f}{ss[i]/mm[i]:>12.1f}x{100*ss[i]/ceil:>13.0f}%")
        knee = next((r for r, v in zip(RAMS, ss) if v >= 0.9 * ceil), None)
        print(f"  base path (steady) reaches 90% of the ceiling at {knee} GB; value of RAM between sizes (tok/s per extra GB): " + ", ".join(f"{a}->{b}: {(y-x)/(b-a):.3f}" for (a, x), (b, y) in zip(zip(RAMS, ss), zip(RAMS[1:], ss[1:])) if (y - x) / (b - a) > 0.002))
        print()
