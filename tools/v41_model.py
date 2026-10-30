#!/usr/bin/env python3
"""Projection for sparse MoE models streamed from disk on one PC, calibrated on the Qwen3-30B Mac runs.
Geometry is a parameter: GEOM['qwen3'] reproduces the Mac measurements (validate), GEOM['v41'] is a 552B model with 384 routed experts, top-6, 16B active at decode.
[M] measured here, [E] estimated. Output for v41 is a projection, not a result.
Calibration (see validate()): compute 55 GFLOP/s [M]; effective read rate of the pool path under memory pressure 1.1-1.7 GB/s, median 1.4 [M, derived], NOT the 3.1 GB/s idle microbenchmark;
plain mmap behaves like an LRU with 0.7x the hit rate at 1.5 footprints and pages at 0.34 GB/s [M]. Overlap is inside the effective rate. Cross-layer prefetch left out (measured 1.0-1.075x).
Desktop profile (24-core CPU, 64 GB, 16 GB GPU): hw_tok_s() adds a GPU for the dense part, a VRAM expert tier, DRAM-bound CPU expert math and an SSD-bound miss term. No machine of that class was measured; every desktop number is a projection from published specs and the Mac calibration.
usage: v41_model.py validate | sweep [--ram 16] [--target 14] [--rate 1.4] [--top 8] | desk [--ssd 7] [--target 14] | desksweep [--ssd 7] [--top 8] | tornado [--ssd 7]"""
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
CURVES = {"lru": dict(cm.LRU_30B), "real": dict(cm.LRU_REAL)}
def _lru(n, curve):   # hit rate on a named curve without leaving combo_model's global curve changed
    saved = cm.LRU; cm.LRU = CURVES[curve]
    try: return cm.lru(n)
    finally: cm.LRU = saved

def cache_gb(g, ram, bits=4.3):
    if g["cache"] is not None: return g["cache"]
    return max(0.0, ram - 3.5 - (g["act"] - g["routed"]) * bits / 8 / 1e9 - 0.7)

def tok_s(geom, K=None, bits=4.3, route=False, spec=0, sc="mid", rate=None, curve="lru", plain=False, ram=None, hit_override=None):
    g = GEOM[geom]; K = K or g["k0"]; k = spec or 1; rate = rate or (R_MMAP if plain else R_EFF)
    foot = g["foot0"] * (K / g["k0"]) * (bits / 4.3) * (cm.union_factor(k) if k > 1 else 1.0)
    h = hit_override if hit_override is not None else _lru(cache_gb(g, ram or g["ram"], bits) / foot, curve) * (MMAP_HIT if plain else 1.0)
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

# ---- desktop profile: many cores, 64 GB, GPU with a VRAM expert tier. [E] = estimated from specs or published llama.cpp reports, not measured here ----
HW = {"desk": dict(ram=64, os=8, g=200.0, bw=50.0, ssd=7.0, eff=0.75, vram=16, vres=2.4, gbw=530.0, t_dense=0.018, r_mmap=1.0, peak=89.6)}
# ram/os GB, g = CPU GFLOP/s [E: ~3.5x the Mac], bw = DRAM GB/s the CPU expert math reaches [E: 50 of 89.6 peak, DDR5-5600 x2], ssd = raw sequential GB/s (7 = PCIe 4, 12 = PCIe 5),
# eff = share of it the pool path keeps under load [Mac: 0.45, idle-machine guess 0.75], vram GB minus vres GB (display, context, KV, buffers), gbw = achieved VRAM GB/s of 960 peak [E],
# peak = DRAM GB/s on paper (2 x 8 B x 5600 MT/s), stream = GB/s of RAM-resident experts the GPU pulls over PCIe in parallel with the CPU math (needs new code),
# t_dense = GPU time for attention + dense + shared per token incl. per-layer syncs [E], r_mmap = plain mmap fault rate GB/s [Mac 0.34 with 8 threads; PC guess 1.0]
NT = {"lru": 16.0, "real": 51.2}   # token-footprints that cover the whole expert set in the traces behind each hit curve (128/8, 512/10)
MMAP_MISS = 1.21                   # plain mmap misses vs an ideal LRU of the same size: 1 - 1.21 * (1 - 0.415) = 0.29 reproduces the Mac plain run [M]
# share of accesses a fixed set of the hottest (layer, expert) pairs covers at 1/2/3/5/10/20 % of all pairs, Qwen3-30B traces, 3 prompts, leave-one-out [M]: set learned on the other two prompts / from the first 32 tokens of the same prompt / oracle on the same prompt
TIER_X = (0.0, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 0.30)
TIER = {"low": (0.0, 0.019, 0.041, 0.064, 0.111, 0.232, 0.450, 0.611), "mid": (0.0, 0.085, 0.146, 0.189, 0.262, 0.407, 0.628, 0.779), "high": (0.0, 0.110, 0.186, 0.246, 0.341, 0.518, 0.757, 0.892)}
def tier_hit(x, lvl):
    ys = TIER[lvl]
    for i in range(1, len(TIER_X)):
        if x <= TIER_X[i]: return ys[i - 1] + (ys[i] - ys[i - 1]) * (x - TIER_X[i - 1]) / (TIER_X[i] - TIER_X[i - 1])
    return ys[-1]
DENSE_BPW, T_ENGRAM = 5.5, 0.003   # dense weights kept at higher precision; Engram table rows read per token (random, few) [E]
SCEN = {   # low / mid / high for every uncertain input (kc: share of the fewer-experts saving that also counts as extra hits, 0 = bytes only, 1 = all cache units rescale; sc: bias strength spread; curve: hit-curve source; mapping: footprints scale with cache/foot [foot] or with the share of all experts [frac])
 "low":  dict(curve="lru",  mapping="frac", kc=0.0, sc="low",  o=0.0,  tier_lvl="low",  eff=0.60, bw=40.0, g=140.0, r_mmap=2.0),
 "mid":  dict(curve="real", mapping="frac", kc=0.5, sc="mid",  o=0.25, tier_lvl="mid",  eff=0.75, bw=50.0, g=200.0, r_mmap=1.0),
 "high": dict(curve="real", mapping="foot", kc=1.0, sc="high", o=0.5,  tier_lvl="high", eff=0.90, bw=65.0, g=260.0, r_mmap=0.6)}

def hw_tok_s(K=6, bits=4.3, route=0, o=0.25, ssd=None, eff=None, curve="real", mapping="frac", gpu=True, tier=True, plain=False, tier_lvl="mid", sc="mid", kc=0.5, ram=None, bw=None, g=None, r_mmap=None, stream=0.0, hw="desk"):
    """tokens/s and the parts of one decode token for GEOM['v41'] on the desktop profile. plain=True is llama.cpp mmap (gpu=True: dense on the GPU, experts still paged)."""
    G = GEOM["v41"]; p = dict(HW[hw]); p.update({k: v for k, v in dict(bw=bw, g=g, r_mmap=r_mmap).items() if v is not None})
    foot = G["foot0"] * bits / 4.3 * K / G["k0"]; dense_gb = (G["act"] - G["routed"]) * DENSE_BPW / 8 / 1e9
    fl_r = G["routed"] * 2 * K / G["k0"] / 1e9; fl_d = (G["act"] - G["routed"]) * 2 / 1e9
    ratio = (NT[curve] / 64.0 if mapping == "frac" else 1.0) * (K / G["k0"]) ** (1 - kc)   # foot already shrinks with K (full credit); this factor takes back the share (1 - kc) of that
    cur = lambda n: _lru(n, curve)
    cache = max(0.0, (p["ram"] if ram is None else ram) - p["os"] - (0 if gpu else dense_gb))
    if plain:
        h = max(0.0, 1 - MMAP_MISS * (1 - cur(cache / foot * ratio)))
        tc = (p["t_dense"] if gpu else 0.0) + max(((0 if gpu else dense_gb) + foot * (2 - h)) / p["bw"], ((0 if gpu else fl_d) + fl_r) / p["g"]) + T_ENGRAM
        tio = foot * (1 - h) / p["r_mmap"]; return dict(tok_s=1 / (tc + tio), h=h, hv=0.0, tc=tc, tio=tio, foot=foot)
    vexp = max(0.0, p["vram"] - p["vres"] - dense_gb) if (gpu and tier) else 0.0
    h = cur((cache + vexp) / foot * ratio); hv = tier_hit(vexp / (G["foot0"] * bits / 4.3 * 64), tier_lvl) if vexp else 0.0   # share of ALL routed bytes held in VRAM
    if route:
        h2 = h + (1 - h) * (BETA[route] + SPREAD[sc]); hv = hv * h2 / h if h > 0 else hv; h = h2
    hv = min(hv, h); rate = (p["ssd"] if ssd is None else ssd) * (p["eff"] if eff is None else eff)
    cpu_gb = foot * (h - hv) + 2 * foot * (1 - h)          # DRAM traffic: resident reads + (DMA write, read) of every fetched expert
    sb = min(stream, max(0.0, p["peak"] * 0.9 - p["bw"])) if gpu else 0.0   # extra DRAM->GPU bytes/s over PCIe, only the DRAM bandwidth the CPU path leaves free
    if gpu: tc = p["t_dense"] + foot * hv / p["gbw"] + max(cpu_gb / (p["bw"] + sb), fl_r * (1 - hv) / p["g"]) + T_ENGRAM
    else:   tc = max((dense_gb + cpu_gb) / p["bw"], (fl_d + fl_r) / p["g"]) + T_ENGRAM
    tio = foot * (1 - h) / rate; t = tc + tio - o * min(tc, tio)   # o = share of the smaller of compute / reads that is hidden behind the other
    return dict(tok_s=1 / t, h=h, hv=hv, tc=tc, tio=tio, foot=foot, rate=rate, vexp=vexp)

def scen_kw(name, ssd):
    s = dict(SCEN[name]); s["ssd"] = ssd; return s
def desk_base(name, gpu=False):
    s = scen_kw(name, None); return hw_tok_s(plain=True, gpu=gpu, **{k: s[k] for k in ("curve", "mapping", "bw", "g", "r_mmap")})["tok_s"]
def desk_run(name, ssd, **kw):
    s = scen_kw(name, ssd); s.update(kw); return hw_tok_s(**s)

LADDER = [("pool, CPU only", dict(gpu=False)), ("+ GPU for dense (stock hybrid)", dict(tier=False)), ("+ VRAM hot-expert tier", {}), ("+ bias 1.0", dict(route=1)), ("+ 5 of 6 experts", dict(route=1, K=5)),
          ("6 experts + bias 3.0", dict(route=3)), ("5 of 6 + bias 3.0", dict(route=3, K=5)), ("bias 1.0 + 2.6 bpw", dict(route=1, bits=2.6)), ("bias 1.0 + 2.0 bpw", dict(route=1, bits=2.0)),
          ("5 of 6 + bias 3.0 + 2.6 bpw", dict(route=3, K=5, bits=2.6)), ("4 of 6 + bias 3.0 + 3.0 bpw", dict(route=3, K=4, bits=3.0)),
          ("bias 1.0 + overlap 0.75", dict(route=1, o=0.75)), ("bias 1.0 + PCIe stream 20 GB/s", dict(route=1, stream=20.0)), ("bias 1.0 + overlap 0.75 + stream 20", dict(route=1, o=0.75, stream=20.0)),
          ("5 of 6 + bias 1.0 + overlap + stream", dict(route=1, K=5, o=0.75, stream=20.0))]
def desk_report(ssd, target):
    print(f"desktop profile, SSD raw {ssd:g} GB/s; ratios are against plain llama.cpp mmap on the same machine: CPU only, routed experts 4.3 bpw, dense 5.5 bpw, 6 experts per token; plain mmap speed does not rise with the SSD (page-fault latency bound)")
    for name in ("low", "mid", "high"):
        b0, b1 = desk_base(name), desk_base(name, gpu=True)
        print(f"== {name}: plain mmap {b0:.2f} tok/s; plain mmap + GPU dense {b1:.2f} tok/s")
        for lab, kw in LADDER:
            r = desk_run(name, ssd, **kw); mark = "  >= target" if r["tok_s"] / b0 >= target else ""
            print(f"   {lab:30s} {r['tok_s']:6.2f} tok/s {r['tok_s'] / b0:5.1f}x  hit {r['h']:.2f} (VRAM {r['hv']:.2f}) compute {r['tc'] * 1e3:4.0f} ms reads {r['tio'] * 1e3:4.0f} ms{mark}")

BITS_RISK = {4.3: 0, 3.0: 3, 2.6: 4, 2.5: 4, 2.0: 6}
def desk_sweep(ssd, target):
    """every lever combination, lowest quality risk first; also how many of the low / mid / high scenarios reach the target"""
    rows = []
    for K, bits, route, ov, st in itertools.product((6, 5, 4), (4.3, 3.0, 2.6, 2.0), (0, 1, 3), (0, 1), (0.0, 20.0)):
        res = {}
        for name in ("low", "mid", "high"):
            kw = dict(K=K, bits=bits, route=route, stream=st)
            if ov: kw["o"] = max(SCEN[name]["o"], 0.75)
            r = desk_run(name, ssd, **kw); res[name] = r["tok_s"] / desk_base(name)
        q = RISK[K] + BITS_RISK[bits] + {0: 0, 1: 1, 3: 3}[route]; rows.append((q, ov + (st > 0), res, dict(K=K, bits=bits, route=route, overlap=ov, stream=st)))
    return sorted(rows, key=lambda r: (r[0], r[1], -r[2]["mid"]))

def tornado(ssd, **kw):
    """one input at a time from mid to low and high; returns [(input, tok/s low, tok/s high)] sorted by spread"""
    base = dict(scen_kw("mid", ssd)); base.update(kw); out = []
    for k in ("curve+mapping", "o", "tier_lvl", "eff", "bw", "g", "sc", "kc"):
        vals = []
        for lvl in ("low", "high"):
            a = dict(base)
            for kk in k.split("+"): a[kk] = SCEN[lvl][kk]
            a.pop("r_mmap"); vals.append(hw_tok_s(**a)["tok_s"])
        out.append((k, vals[0], vals[1]))
    return sorted(out, key=lambda r: -abs(r[2] - r[1]))

RISK = {6: 0, 5: 2, 4: 4}
def grid():
    for K, bits, route, sp in itertools.product((6, 5, 4), (4.3, 3.0, 2.5), (0, 1, 3), (0, 4)):
        yield dict(K=K, bits=bits, route=route, spec=sp)
def risk(c): return RISK[c["K"]] + {4.3: 0, 3.0: 3, 2.5: 4}[c["bits"]] + {0: 0, 1: 1, 3: 3}[c["route"]] + (1 if c["spec"] else 0)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["validate", "sweep", "desk", "tornado", "desksweep"]); ap.add_argument("--ram", type=float, default=16); ap.add_argument("--target", type=float, default=14)
    ap.add_argument("--ssd", type=float, default=7.0); ap.add_argument("--rate", type=float, default=R_EFF); ap.add_argument("--top", type=int, default=8); a = ap.parse_args()
    if a.cmd == "validate": validate(); sys.exit()
    if a.cmd == "desk": desk_report(a.ssd, a.target); sys.exit()
    if a.cmd == "desksweep":
        print(f"desktop profile, SSD {a.ssd:g} GB/s: lever combinations by quality risk (K cut + bits + bias), x vs plain mmap CPU-only in the low / mid / high scenario; engineering = overlap 0.75 and/or PCIe stream")
        seen = 0
        for q, eng, res, c in desk_sweep(a.ssd, a.target):
            if res["mid"] >= a.target and seen < a.top: seen += 1; print(f"   risk {q:2d} eng {eng}  {res['low']:5.1f} / {res['mid']:5.1f} / {res['high']:5.1f}x  K={c['K']} bits={c['bits']} bias={c['route']} overlap={c['overlap']} stream={c['stream']:g}")
        sys.exit()
    if a.cmd == "tornado":
        for lab, kw in (("6 experts, no bias", {}), ("6 experts + bias 1.0", dict(route=1)), ("5 of 6 + bias 3.0", dict(route=3, K=5))):
            print(f"== {lab}, SSD {a.ssd:g} GB/s, mid scenario = {desk_run('mid', a.ssd, **kw)['tok_s']:.2f} tok/s; one input moved to low / high:")
            for k, lo, hi in tornado(a.ssd, **kw): print(f"   {k:14s} {lo:6.2f} .. {hi:6.2f}")
        sys.exit()
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
