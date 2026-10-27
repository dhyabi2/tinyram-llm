#!/usr/bin/env python3
"""Expert-cache hit-rate simulation + throughput model for the 8 GB MoE-streaming design.

SYNTHETIC trace: 48 layers x 512 experts, top-10 routed experts per layer per token (the shared expert is
resident and not simulated). Popularity is Zipf(s) per layer (random permutation per layer); between
consecutive tokens each selected expert is kept with probability `persist` (temporal locality), otherwise
redrawn from the layer's popularity. The numbers are therefore only as real as those two knobs: replace the
generator with real router traces (profile_experts.py) before quoting any of them as a result of the model.

Policies: LRU, SLRU 80/20, SLRU + pinning the hottest experts found in a SEPARATE warm-up profile (no peeking),
the static-optimal bound (best possible fixed set), and Belady MIN on a prefix (the true ceiling for any policy).
"""
import numpy as np, collections, heapq, time, itertools, sys

L, E, K = 48, 512, 10
SEED = 20260101
t0 = time.time()

def make_trace(n_tokens, s, persist, seed):
    """Per layer: Zipf(s) popularity (random permutation per layer). Each token keeps every previous expert with
    probability `persist`; vacated slots are refilled by sampling WITHOUT replacement from the popularity
    (Gumbel top-k), never with an expert already selected this token (no duplicate = no fake hits)."""
    rng = np.random.default_rng(seed)
    logp = np.empty((L, E))
    for l in range(L):
        w = 1.0 / np.arange(1, E + 1) ** s
        w = w[rng.permutation(E)]
        logp[l] = np.log(w / w.sum())
    sel = np.argsort(-(logp + rng.gumbel(size=(L, E))), axis=1)[:, :K]          # initial token
    out = np.empty((n_tokens, L, K), dtype=np.int16)
    for t in range(n_tokens):
        keep = rng.random((L, K)) < persist
        g = logp + rng.gumbel(size=(L, E))
        for l in range(L):
            kept = sel[l][keep[l]]
            need = K - len(kept)
            if need:
                g[l, kept] = -np.inf                                              # cannot be drawn again
                fresh = np.argpartition(-g[l], need)[:need]
                sel[l] = np.concatenate([kept, fresh])
        out[t] = sel
    return out, logp

def accesses(trace):
    for t in range(trace.shape[0]):
        for l in range(L):
            for e in trace[t, l]:
                yield l * E + int(e)

class LRU:
    def __init__(self, cap): self.cap = cap; self.d = collections.OrderedDict()
    def access(self, k):
        if k in self.d: self.d.move_to_end(k); return True
        self.d[k] = 1
        if len(self.d) > self.cap: self.d.popitem(last=False)
        return False

class SLRU:
    """Segmented LRU: new items enter probation; a hit promotes to protected; pinned items never evicted."""
    def __init__(self, cap, protected=0.8, pinned=()):
        self.pinned = set(pinned); self.cap = max(cap - len(self.pinned), 1)
        self.pcap = int(self.cap * protected); self.prob = collections.OrderedDict(); self.prot = collections.OrderedDict()
    def access(self, k):
        if k in self.pinned: return True
        if k in self.prot: self.prot.move_to_end(k); return True
        if k in self.prob:
            del self.prob[k]; self.prot[k] = 1
            if len(self.prot) > self.pcap:
                old, _ = self.prot.popitem(last=False); self.prob[old] = 1
                self._trim()
            return True
        self.prob[k] = 1; self._trim(); return False
    def _trim(self):
        while len(self.prob) + len(self.prot) > self.cap and self.prob: self.prob.popitem(last=False)

def hit_rate(policy, trace, warm):
    h = n = 0
    for i, k in enumerate(accesses(trace)):
        r = policy.access(k)
        if i >= warm: h += r; n += 1
    return h / n

def belady(trace, cap, limit_tokens=250):
    seq = list(accesses(trace[:limit_tokens]))
    nxt = {}; nexts = [0] * len(seq)
    for i in range(len(seq) - 1, -1, -1):
        nexts[i] = nxt.get(seq[i], 1 << 60); nxt[seq[i]] = i
    cache = {}; heap = []; hits = 0
    warm = len(seq) // 5; counted = 0
    for i, k in enumerate(seq):
        if k in cache:
            if i >= warm: hits += 1
        else:
            if len(cache) >= cap:
                while True:
                    negn, victim = heapq.heappop(heap)
                    if cache.get(victim) == -negn: del cache[victim]; break
        if i >= warm: counted += 1
        cache[k] = nexts[i]; heapq.heappush(heap, (-nexts[i], k))
    return hits / counted

def static_bound(trace, cap):
    cnt = collections.Counter(accesses(trace))
    top = sum(c for _, c in cnt.most_common(cap))
    return top / (trace.shape[0] * L * K)

print("=" * 86)
print("EXPERT-CACHE SIMULATION  (synthetic trace; seeded; L=48 E=512 top-10 routed/token)")
print("=" * 86)
N_PROFILE, N_RUN = 200, 500
CAPS = [600, 1200, 2400, 4800]          # experts in the pool  (~0.8, 1.7, 3.4, 6.7 GB at ~1.4 MB each)
RESULTS = {}
SKEWS = [float(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [0.6, 0.8, 1.0]
PERSISTS = [float(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else (0.5, 0.7, 0.85)
for s in SKEWS:
    for persist in PERSISTS:
        prof, _ = make_trace(N_PROFILE, s, persist, SEED + 1)           # profiling run (separate seed)
        trace, _ = make_trace(N_RUN, s, persist, SEED)
        freq = collections.Counter(accesses(prof))
        warm = 150 * L * K
        row = {}
        for cap in CAPS:
            pins = [k for k, _ in freq.most_common(max(int(0.05 * cap), 1))]
            row[cap] = dict(
                lru=hit_rate(LRU(cap), trace, warm),
                slru80=hit_rate(SLRU(cap), trace, warm),                         # the 80/20 split a stage recommended
                slru50=hit_rate(SLRU(cap, protected=0.5), trace, warm),          # probation (50%) >= one token's 480 accesses
                slru50_pin=hit_rate(SLRU(cap, protected=0.5, pinned=pins), trace, warm),
                lru_pin=hit_rate(SLRU(cap, protected=0.0, pinned=pins), trace, warm),
                static=static_bound(trace, cap),
                belady=belady(trace, cap) if cap in (1200,) else None,
            )
        RESULTS[(s, persist)] = row
        r = row[1200]
        print(f"zipf s={s:.1f} persist={persist:.2f} | pool=1200: LRU {r['lru']:.3f} | SLRU80/20 {r['slru80']:.3f} | SLRU50/50 {r['slru50']:.3f} "
              f"| SLRU50+pin {r['slru50_pin']:.3f} | LRU+pin {r['lru_pin']:.3f} | static-opt {r['static']:.3f} | Belady {r['belady']:.3f}  [{time.time()-t0:.0f}s]")
        sys.stdout.flush()

if (0.8, 0.7) in RESULTS:
    print("\nPool-size sweep, SLRU+pin hit rate (mid skew s=0.8, persist=0.7):")
    for cap in CAPS: print(f"  pool {cap:5d} experts (~{cap*1.4/1000:.1f} GB): hit rate {RESULTS[(0.8, 0.7)][cap]['lru']:.3f} (LRU)  {RESULTS[(0.8, 0.7)][cap]['slru50_pin']:.3f} (SLRU50+pin)")
if len(SKEWS) < 3 or '--model' not in sys.argv:
    print(f"\n(total {time.time()-t0:.0f}s)"); sys.exit(0)

# --------------------------------------------------------------------------------------------------
print("\n" + "=" * 86)
print("THROUGHPUT MODEL  tok/s = 1 / max(t_ssd, t_ram, t_compute)   (all inputs are ASSUMPTIONS, stated)")
print("=" * 86)
D = 3.387e9; E_ACT = 2.613e9                   # dense-always-on / active-expert params from '125B total, 6B active, 11 of 512'
EXPERT_PARAMS = (125e9 - D) / (L * E)          # ~4.95M params per expert
BPW_DENSE = 6.9                                # attention Q6_K / shared Q8_0 mix
EXPERT_MB = {"cold-tier misses (1.1 MB)": 1.1, "param-weighted avg (1.4 MB)": EXPERT_PARAMS * 2.256 / 8 / 1e6}
DRIVES = {"SATA SSD": 0.45, "NVMe PCIe3": 2.0, "NVMe PCIe4": 3.5}      # effective random-read GB/s
RAM_GBS = {"DDR4 dual-ch": 25.0, "DDR5 dual-ch": 50.0}
GFLOPS = 60.0
MTP_BYTES_SPEEDUP = 1.59
print(f"expert = {EXPERT_PARAMS/1e6:.2f}M params; dense active = {D/1e9:.2f}B ({D*BPW_DENSE/8/1e9:.2f} GB read from RAM per token); "
      f"routed reads per token = {L*K}")
print(f"compute ceiling = 6e9*2 FLOP / {GFLOPS:.0f} GFLOP/s -> {1/(12e9/(GFLOPS*1e9)):.1f} tok/s (speculation does NOT reduce compute per accepted token)")

def tps(h, mb, ssd, ram, mtp):
    miss_bytes = L * K * (1 - h) * mb * 1e6
    ram_bytes = D * BPW_DENSE / 8 + L * K * h * mb * 1e6
    t_ssd, t_ram, t_c = miss_bytes / (ssd * 1e9), ram_bytes / (ram * 1e9), 12e9 / (GFLOPS * 1e9)
    if mtp: t_ssd, t_ram = t_ssd / MTP_BYTES_SPEEDUP, t_ram / MTP_BYTES_SPEEDUP
    t = max(t_ssd, t_ram, t_c)
    return 1 / t, ("compute" if t == t_c else "SSD" if t == t_ssd else "RAM")

print("\n tok/s (bound), no MTP | with MTP     rows: hit rate; expert size 1.4 MB; DDR4")
print(f"{'hit rate':>9} | " + " | ".join(f"{d:>22}" for d in DRIVES))
for h in (0.30, 0.40, 0.55, 0.70, 0.85):
    cells = []
    for d, g in DRIVES.items():
        a, wa = tps(h, EXPERT_MB["param-weighted avg (1.4 MB)"], g, 25.0, False)
        b, wb = tps(h, EXPERT_MB["param-weighted avg (1.4 MB)"], g, 25.0, True)
        cells.append(f"{a:5.2f} {wa:>3}| {b:5.2f} {wb:>3}")
    print(f"{h:9.2f} | " + " | ".join(f"{c:>22}" for c in cells))

print("\nHit rate the SIMULATION gives at pool=1200 (SLRU+pin) -> predicted tok/s on NVMe PCIe3, DDR4, no MTP / MTP:")
for (s, p), row in sorted(RESULTS.items()):
    h = row[1200]["lru"]
    a, wa = tps(h, 1.4, 2.0, 25.0, False); b, wb = tps(h, 1.4, 2.0, 25.0, True)
    print(f"  s={s:.1f} persist={p:.2f}: h={h:.3f} -> {a:4.2f} tok/s ({wa}) | {b:4.2f} tok/s ({wb})")

print("\n100x condition:  F/B >= 100 AND F >= 2 tok/s.  Required F for a few baselines:")
for B in (0.01, 0.03, 0.05, 0.10): print(f"  B = {B:4.2f} tok/s -> F >= {max(100*B, 2.0):.1f} tok/s")
print(f"\n(total {time.time()-t0:.0f}s)")
