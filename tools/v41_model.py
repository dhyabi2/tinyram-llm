#!/usr/bin/env python3
"""Projection for sparse MoE models streamed from disk on one PC, calibrated on the Qwen3-30B Mac runs.
Geometry is a parameter: GEOM['qwen3'] reproduces the Mac measurements (validate), GEOM['v41'] is a 552B model with 384 routed experts, top-6, 16B active at decode.
[M] measured here, [E] estimated. Output for v41 is a projection, not a result.
Calibration (see validate()): compute 55 GFLOP/s [M]; effective read rate of the pool path under memory pressure 1.1-1.7 GB/s, median 1.4 [M, derived], NOT the 3.1 GB/s idle microbenchmark;
plain mmap behaves like an LRU with 0.7x the hit rate at 1.5 footprints and pages at 0.34 GB/s [M]. Overlap is inside the effective rate. Cross-layer prefetch left out (measured 1.0-1.075x).
usage: v41_model.py validate | sweep [--ram 16] [--target 14] [--rate 1.4] [--top 8]"""
import argparse, itertools, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import combo_model as cm

GEOM = {   # act = active params at decode, routed = routed share of them, foot0 = routed GB one token touches at the base k, k0 = base experts/token, res_gb = resident GB, cache = expert cache GB
 "qwen3": dict(act=3.3e9, routed=1.8e9, k0=8, foot0=1.02, cache=1.5 / 1.074, ram=None),
 "v41":   dict(act=16e9, routed=8.4e9, k0=6, foot0=8.4e9 * 4.3 / 8 / 1e9, cache=None, ram=16),   # routed share is [E]; model card gives only totals
}
G_CPU, R_MMAP, R_EFF, MMAP_HIT = 55.0, 0.34, 1.4, 0.7
# share of misses removed by bias 1.0 / 3.0 [M]: Qwen3 pool hit 59->72% / 81% (8 experts), 52->75% / 85% (6 experts)
BETA = {1: 0.40, 3: 0.60}; SPREAD = {"low": -0.10, "mid": 0.0, "high": 0.10}
ALPHA = {"low": 0.5, "mid": 0.7, "high": 0.85}; B_EFF = 1.6
CURVES = {"lru": dict(cm.LRU), "real": dict(cm.LRU_REAL)}

def cache_gb(g, ram, bits=4.3):
    if g["cache"] is not None: return g["cache"]
    return max(0.0, ram - 3.5 - (g["act"] - g["routed"]) * bits / 8 / 1e9 - 0.7)

def tok_s(geom, K=None, bits=4.3, route=False, spec=0, sc="mid", rate=None, curve="lru", plain=False, ram=None, hit_override=None):
    g = GEOM[geom]; K = K or g["k0"]; k = spec or 1; rate = rate or (R_MMAP if plain else R_EFF)
    cm.LRU = CURVES[curve]
    foot = g["foot0"] * (K / g["k0"]) * (bits / 4.3) * (cm.union_factor(k) if k > 1 else 1.0)
    h = hit_override if hit_override is not None else cm.lru(cache_gb(g, ram or g["ram"], bits) / foot) * (MMAP_HIT if plain else 1.0)
    if route and not plain and hit_override is None: h = h + (1 - h) * (BETA[route] + SPREAD[sc])
    comp = (g["act"] - g["routed"] * (1 - K / g["k0"])) * 2 / 1e9 / G_CPU * (k / B_EFF if k > 1 else 1.0)
    io = foot * (1 - h) / rate
    tokens = (1 - ALPHA[sc] ** k) / (1 - ALPHA[sc]) if k > 1 else 1.0
    return tokens / (comp + io), h

def validate():
    rows = [("plain mmap 8", dict(plain=True), 0.447), ("pool+overlap 8", {}, 2.00), ("anchored 6", dict(K=6), 3.05),
            ("6 + bias1", dict(K=6, hit_override=0.75), 4.06), ("6 + bias3", dict(K=6, hit_override=0.85), 5.84), ("8 + bias1", dict(hit_override=0.72), 2.95), ("8 + bias3", dict(hit_override=0.81), 4.02)]
    print("validation on Qwen3-30B, 9 GB ballast Mac (model uses measured hit rate where one exists):")
    worst = 0
    for name, kw, meas in rows:
        t, h = tok_s("qwen3", **kw); err = t / meas - 1; worst = max(worst, abs(err)); print(f"  {name:15s} measured {meas:5.2f}  model {t:5.2f}  hit {h:.2f}  error {err:+.0%}")
    print(f"  worst error {worst:.0%}"); return worst

RISK = {6: 0, 5: 2, 4: 4}
def grid():
    for K, bits, route, sp in itertools.product((6, 5, 4), (4.3, 3.0, 2.5), (0, 1, 3), (0, 4)):
        yield dict(K=K, bits=bits, route=route, spec=sp)
def risk(c): return RISK[c["K"]] + {4.3: 0, 3.0: 3, 2.5: 4}[c["bits"]] + {0: 0, 1: 1, 3: 3}[c["route"]] + (1 if c["spec"] else 0)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["validate", "sweep"]); ap.add_argument("--ram", type=float, default=16); ap.add_argument("--target", type=float, default=14)
    ap.add_argument("--rate", type=float, default=R_EFF); ap.add_argument("--top", type=int, default=8); a = ap.parse_args()
    if a.cmd == "validate": validate(); sys.exit()
    for curve in ("lru", "real"):
        base = tok_s("v41", plain=True, curve=curve, ram=a.ram)[0]
        for sc in ("low", "mid", "high"):
            rows = []
            for c in grid():
                t, h = tok_s("v41", sc=sc, curve=curve, ram=a.ram, rate=a.rate, **c); rows.append((t / base, t, h, risk(c), c))
            ok = sorted([r for r in rows if r[0] >= a.target], key=lambda r: (r[3], -r[0]))
            print(f"== {a.ram:g} GB, read rate {a.rate} GB/s, curve {curve}, {sc}: plain mmap {base:.3f} tok/s; best {max(r[0] for r in rows):.1f}x ({max(r[1] for r in rows):.2f} tok/s); combos >= {a.target:g}x: {len(ok)}")
            for r in ok[:a.top if sc == "mid" else 1]:
                c = r[4]; print(f"   {r[0]:5.1f}x {r[1]:5.2f} tok/s hit {r[2]:.2f} risk {r[3]} K={c['K']} bits={c['bits']} bias={c['route']} spec={c['spec']}")
