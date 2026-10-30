#!/usr/bin/env python3
"""Projection for a 552B sparse MoE (384 routed experts + 1 shared, top-6, 16B active at decode) streamed from disk on one PC.
Reuses the measured LRU hit curve (30B traces) from combo_model.py. [M] measured here, [E] estimated. The output is a projection, not a result.
Measured levers only: overlap hides OVERLAP_HIDE of the I/O time ([M] 2.28 -> 2.645 tok/s = 1.16x), cache bias moves BETA of the misses to hits
([M] misses -60..-91% at bias 1..3), fewer experts, lower bits, speculative decoding. Cross-layer prefetch is left out: measured 1.0-1.075x.
usage: v41_model.py [--ram 16] [--target 14] [--top 10]"""
import argparse, itertools, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import combo_model as cm

ACT, ROUTED_ACT = 16e9, 8.4e9          # [source: model card] active at decode; [E] routed share of it (rest is resident dense + shared expert)
BPW0, G_CPU = 4.3, 55.0                # [M] CPU GFLOP/s on the 4-core Mac
R_MMAP, R_EXPL = cm.R_MMAP, cm.R_EXPL  # [M] GB/s mmap paging vs explicit 2.7 MB reads
OVERLAP_HIDE = 0.25                    # [M] pool 2.28 -> overlap 2.645 tok/s, as a share of the I/O time hidden
BETA = {"low": 0.4, "mid": 0.6, "high": 0.8}   # [M] share of misses removed by cache bias 1.0..3.0 on Qwen3 (-60..-91% at bias 1..3, partly offset on other models)
ALPHA = {"low": 0.5, "mid": 0.7, "high": 0.85}  # [E] speculative acceptance per draft token (copy-heavy text measured ~0.77, open text lower)
B_EFF = 1.6                            # [E] batch-4 matmul efficiency vs batch 1
CURVES = {"lru": dict(cm.LRU), "real": dict(cm.LRU_REAL)}

def cache_gb(ram, bits): return max(0.0, ram - 3.5 - (ACT - ROUTED_ACT) * bits / 8 / 1e9 - 0.7)   # OS, resident dense+shared, KV

def tok_s(ram, K=6, bits=BPW0, route=False, overlap=False, spec=0, sc="mid", rate=R_EXPL, curve="lru"):
    cm.LRU = CURVES[curve]; k = spec or 1
    foot = ROUTED_ACT * (K / 6.0) * bits / 8 / 1e9 * (cm.union_factor(k) if k > 1 else 1.0)   # routed bytes one pass touches
    h = cm.lru(cache_gb(ram, bits) / foot)
    if route: h = h + (1 - h) * BETA[sc]
    comp = (ACT - ROUTED_ACT * (1 - K / 6.0)) * 2 / 1e9 / G_CPU * (k / B_EFF if k > 1 else 1.0)   # s per pass
    io = foot * (1 - h) / rate
    t = comp + io * ((1 - OVERLAP_HIDE) if overlap else 1.0)
    tokens = (1 - ALPHA[sc] ** k) / (1 - ALPHA[sc]) if k > 1 else 1.0   # expected tokens per pass
    return tokens / t, h

def plain(ram, curve="lru"): return tok_s(ram, rate=R_MMAP, curve=curve)[0]
def risk(c): return {6: 0, 5: 2, 4: 4}[c["K"]] + {4.3: 0, 3.0: 3, 2.5: 4}[c["bits"]] + (1 if c["route"] else 0) + (1 if c["spec"] else 0)
def grid():
    for K, bits, route, ov, sp in itertools.product((6, 5, 4), (4.3, 3.0, 2.5), (False, True), (False, True), (0, 4)):
        yield dict(K=K, bits=bits, route=route, overlap=ov, spec=sp)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ram", type=float, default=16); ap.add_argument("--target", type=float, default=14); ap.add_argument("--top", type=int, default=6)
    a = ap.parse_args()
    for curve in ("lru", "real"):
        for sc in ("low", "mid", "high"):
            base = plain(a.ram, curve); rows = [(tok_s(a.ram, sc=sc, curve=curve, **c), c) for c in grid()]
            rows = [(t / base, t, h, risk(c), c) for (t, h), c in rows]
            ok = sorted([r for r in rows if r[0] >= a.target], key=lambda r: (r[3], -r[0]))
            print(f"== {a.ram:g} GB, hit curve {curve}, scenario {sc}: plain mmap {base:.3f} tok/s, expert cache {cache_gb(a.ram, 4.3):.1f} GB; best {max(r[0] for r in rows):.1f}x; combos >= {a.target:g}x: {len(ok)}")
            for r in ok[:a.top if sc == "mid" else 1]:
                c = r[4]; print(f"   {r[0]:5.1f}x {r[1]:5.2f} tok/s hit {r[2]:.2f} risk {r[3]} K={c['K']} bits={c['bits']} bias={c['route']} overlap={c['overlap']} spec={c['spec']}")
