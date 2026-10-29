#!/usr/bin/env python3
"""Combination model: decode tokens/s of a ~125B MoE (6B active, 10 of 512 experts, 48 layers) when several speed levers are combined.
Every combination of lever settings is evaluated for 8/16/32 GB machines under three parameter scenarios (conservative / likely / optimistic).
Time per token: serial t = io + compute; overlap-with-prediction t = max(compute, prefetched io) + demand io. Cache hit comes from the MEASURED LRU curve in token-footprints.
Parameters are tagged [M] measured here or [E] estimated; a combination that depends on [E] parameters is a projection, not a result.
usage: combo_model.py [--top N]"""
import itertools, math, multiprocessing as mp, statistics as st, sys, json

# ---------------- measured / fixed ----------------
LRU = {0.0: 0.0, 0.9: 0.000, 1.0: 0.306, 1.5: 0.454, 2.0: 0.549, 3.0: 0.686, 4.0: 0.793, 7.5: 0.950, 10: 0.962, 20.6: 0.962}   # [M] mean of 3 prompts, 30B traces, cold start included
# Alternative hit curve (COMBO_HIT=real): points read from public benchmarks on the 125B-class model itself (llama.cpp PR threads, single-user and unverified):
# hit 0.72 / 0.77 / 0.89 at 13 / 16 / 28 % of the experts, and an offline per-layer LRU study 0.621 / 0.759 / 0.876 / 0.912 / 0.974 at 7.8 / 14 / 25 / 31 / 62.5 %.
# 1 footprint = 480 of the 24,576 expert slices = 1.95 % of the experts, so k = fraction x 51.2. Points below k = 4 are NOT measured: the 30B curve is scaled by 0.78 (its ratio at k = 4).
LRU_REAL = {0.0: 0.0, 0.9: 0.0, 1.0: 0.24, 2.0: 0.43, 3.0: 0.54, 4.0: 0.621, 6.66: 0.72, 7.17: 0.759, 8.2: 0.77, 12.8: 0.876, 14.3: 0.89, 15.9: 0.912, 32.0: 0.974}
import os as _os
if _os.environ.get("COMBO_HIT") == "real": LRU = LRU_REAL
UNION = {1: 8.0, 2: 12.6, 3: 16.3, 4: 19.6, 6: 24.9, 8: 29.1, 12: 35.2, 16: 39.2}                                              # [M] unique experts per layer for k consecutive tokens (of 8 per token)
EXP_GB = 480 * 2.7e-3            # [E] expert bytes per token at top-10, 4.3 bpw = 1.296 GB
FLOP_TOT, DENSE_FRAC = 12.0, 3.4 / 6.0   # [E] 12 GFLOP per token; dense share of the active parameters
G_CPU = 55.0                     # [M] GFLOP/s decode on the 4-core laptop CPU (cached 30B decode)
R_MMAP, R_EXPL = 0.34, 3.11      # [M] GB/s: mmap paging; explicit 2.7 MB extents with 4 reads in flight
CACHE_GIB = {"8 GB": 2.5, "16 GB": 9.15, "32 GB": 25.15}    # [E] expert cache left after OS, KV and the resident dense part (plan F)
SCEN = {   # name: (conservative, likely, optimistic)     [E] unless noted
 "alpha":   (0.60, 0.80, 0.90),    # speculative acceptance per draft token
 "beta":    (0.20, 0.40, 0.50),    # share of misses turned into hits by cache-aware routing
 "phi":     (0.30, 0.45, 0.60),    # share of misses prefetched early (cross-layer predictor precision 0.55 in-prompt but 0.31-0.43 held out [M])
 "kappa":   (1.0, 1.2, 2.0),       # compute speed-up of table-lookup kernels at <=3 bit. MEASURED existing low-bit kernels on OLMoE decode: Q3_K_S x0.86, Q2_K x1.12 vs Q4_K_M; a lookup kernel beating them is unmeasured
 "gpu_bw":  (200., 300., 500.),    # GB/s of the GPU for weights resident in VRAM
 "b4":      (1.3, 1.6, 2.2),       # batch-4 matmul efficiency vs batch 1
 "b8":      (1.5, 2.2, 3.2),       # batch-8
 "lowrank": (0.60, 0.50, 0.35),    # expert byte factor of a shared-basis + residual layout
}
RISK = {"topk6": 3, "topk4": 4, "bits3": 3, "bits25": 4, "lowrank": 3, "route": 1, "exit": 4}   # none 0, low 1, medium 2, high 3, very high 4. MEASURED on OLMoE (perplexity vs Q4_K_M/top-8): top-6 +8.8%, top-4 +45.6%, Q3_K_S +10.4% and Q2_K +26% (both requantised, so upper bounds); route/lowrank/exit are [E]

def lru(k):
    ks = sorted(LRU)
    if k <= ks[0]: return LRU[ks[0]]
    for a, b in zip(ks, ks[1:]):
        if a <= k <= b: return LRU[a] + (LRU[b] - LRU[a]) * (k - a) / (b - a)
    return LRU[ks[-1]]

def union_factor(k):          # experts per token-pass relative to one token
    ks = sorted(UNION)
    if k in UNION: return UNION[k] / 8.0
    for a, b in zip(ks, ks[1:]):
        if a <= k <= b: return (UNION[a] + (UNION[b] - UNION[a]) * (k - a) / (b - a)) / 8.0
    return UNION[ks[-1]] / 8.0

def tok_s(c, machine, sc, rate=R_EXPL, base_model=False):
    """c = dict(K, bits, lowrank, route, overlap, spec, gpu, lut, exit). sc = scenario dict of parameter values."""
    K = c["K"]; bits = c["bits"]; ex = c["exit"]
    byte_f = (K / 10.0) * (bits / 4.3) * ex * (sc["lowrank"] if c["lowrank"] else 1.0)
    exp_gb = EXP_GB * byte_f                                            # expert bytes touched per token
    flop_exp = FLOP_TOT * (1 - DENSE_FRAC) * (K / 10.0) * ex * (0.8 if c["lowrank"] else 1.0)
    flop_dense = FLOP_TOT * DENSE_FRAC * ex
    k = c["spec"] or 1
    cache_gb = (machine if isinstance(machine, (int, float)) else CACHE_GIB[machine]) * 1.074 + (max(0.0, c["gpu"] - 1.9 - 1.0) if c["gpu"] else 0.0)   # RAM cache + VRAM left after dense and a working margin
    foot = exp_gb * (union_factor(k) if k > 1 else 1.0)                 # expert bytes one pass touches
    h = lru(cache_gb / foot)
    if c["route"]: h = h + (1 - h) * sc["beta"]
    io_pass = (1 - h) * foot / rate                                     # seconds of expert I/O per pass
    gfl = G_CPU * (sc["kappa"] if (c["lut"] and bits <= 3.0) else 1.0)
    be = {1: 1.0, 4: sc["b4"], 8: sc["b8"]}.get(k, 1.0)
    # work per pass: k positions of dense+expert FLOPs on the CPU, except the part served from VRAM
    vr_share = 0.0
    if c["gpu"]:
        vr_share = min(1.0, max(0.0, c["gpu"] - 1.9 - 1.0) / max(cache_gb, 1e-9)) * h
    cpu_flop = k * (flop_dense * (0.0 if c["gpu"] else 1.0) + flop_exp * (1 - vr_share))
    comp = cpu_flop / (gfl * be)
    if c["gpu"]: comp += (k * (exp_gb * vr_share + 1.9)) / sc["gpu_bw"] / (be if k > 1 else 1.0)
    if c["overlap"]:
        phi = sc["phi"]; pre = phi * io_pass / 0.43                      # prefetched traffic incl. wasted bytes (held-out precision 0.43 [M])
        t = max(comp, pre) + (1 - phi) * io_pass
    else:
        t = comp + io_pass
    E = (1 - sc["alpha"] ** k) / (1 - sc["alpha"]) if k > 1 else 1.0
    if k > 1: t *= 1.04                                                  # draft cost (n-gram or MTP head) [E]
    return E / t

def cache_for_ram(ram_gb):
    """Expert cache (GiB) left on a machine with `ram_gb` of RAM: least-squares line through plan F's three points (8 GB -> 2.5, 16 -> 9.15, 32 -> 25.15 GiB) [E]."""
    return max(0.3, 0.944 * ram_gb - 5.05)

def best_path(machine, sc): return tok_s(dict(K=10, bits=4.3, lowrank=False, route=False, overlap=False, spec=0, gpu=0, lut=False, exit=1.0), machine, sc)
def mmap_path(machine, sc): return tok_s(dict(K=10, bits=4.3, lowrank=False, route=False, overlap=False, spec=0, gpu=0, lut=False, exit=1.0), machine, sc, rate=R_MMAP)

GRID = dict(K=(10, 6, 4), bits=(4.3, 3.0, 2.5), lowrank=(False, True), route=(False, True), overlap=(False, True), spec=(0, 4, 8), gpu=(0, 12, 24), lut=(False, True), exit=(1.0, 0.75))
def combos():
    for vals in itertools.product(*GRID.values()):
        c = dict(zip(GRID, vals))
        if c["lowrank"] and c["bits"] != 4.3: continue          # low-rank layout replaces the bit setting (it already includes its own residual bits)
        if c["lut"] and c["bits"] > 3.0 and not c["lowrank"]: continue   # lookup kernels need low-bit weights
        yield c

def name(c):
    p = []
    if c["K"] != 10: p.append(f"top{c['K']}")
    if c["lowrank"]: p.append("lowrank")
    elif c["bits"] != 4.3: p.append(f"{c['bits']}bit")
    if c["route"]: p.append("cacherouting")
    if c["overlap"]: p.append("overlap")
    if c["spec"]: p.append(f"spec{c['spec']}")
    if c["gpu"]: p.append(f"gpu{c['gpu']}")
    if c["lut"]: p.append("lut")
    if c["exit"] != 1.0: p.append("exit75")
    return "+".join(p) or "best path"

def risk(c):
    r = 0
    if c["K"] == 6: r += RISK["topk6"]
    if c["K"] == 4: r += RISK["topk4"]
    if c["bits"] == 3.0: r += RISK["bits3"]
    if c["bits"] == 2.5: r += RISK["bits25"]
    if c["lowrank"]: r += RISK["lowrank"]
    if c["route"]: r += RISK["route"]
    if c["exit"] != 1.0: r += RISK["exit"]
    return r

def sc_at(i): return {k: v[i] for k, v in SCEN.items()}
def evaluate(args):
    c, machine = args
    return {"name": name(c), "machine": machine, "risk": risk(c), **{s: tok_s(c, machine, sc_at(i)) for s, i in (("cons", 0), ("likely", 1), ("opt", 2))}, "c": c}

if __name__ == "__main__":
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 8
    cs = list(combos()); jobs = [(c, m) for c in cs for m in CACHE_GIB]
    with mp.Pool() as pool: rows = pool.map(evaluate, jobs, chunksize=200)
    print(f"{len(cs)} combinations x {len(CACHE_GIB)} machines = {len(rows)} evaluations x 3 scenarios")
    for m in CACHE_GIB:
        bp = {s: best_path(m, sc_at(i)) for s, i in (("cons", 0), ("likely", 1), ("opt", 2))}; mm = mmap_path(m, sc_at(1))
        print(f"\n=== {m}: plain mmap {mm:.2f} tok/s | best path (explicit reads + LRU) cons/likely/opt {bp['cons']:.2f}/{bp['likely']:.2f}/{bp['opt']:.2f} tok/s | compute ceiling {G_CPU/FLOP_TOT:.2f}")
        R = [r for r in rows if r["machine"] == m]
        for cap, label in ((0, "NO quality risk"), (2, "quality risk <= 2 (low/medium)"), (99, "any risk")):
            sel = sorted([r for r in R if r["risk"] <= cap], key=lambda r: -r["likely"])[:top]
            print(f"  -- best combinations, {label}:")
            for r in sel:
                print(f"     {r['name']:<46} risk {r['risk']:>2} | cons {r['cons']:5.2f} likely {r['likely']:5.2f} opt {r['opt']:5.2f} | x best path (likely) {r['likely']/bp['likely']:4.2f} | x mmap {r['likely']/mm:5.1f}")
    json.dump(rows, open("combo_rows.json", "w"))
