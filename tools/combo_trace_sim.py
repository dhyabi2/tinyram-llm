#!/usr/bin/env python3
"""Trace-driven combination simulator: cache policy x expert selection x bit width x speculative grouping x layer skipping x overlap x RAM, replayed on REAL router traces.
Measured here: cache hits/misses for every combination are simulated on the three recorded Qwen3-30B traces (top-8 of 128 experts, 48 layers); nothing is read from a table.
Mapped to the 125B target (480 slices per token) by token-footprints, as in throughput_model_v2.py. Scenario numbers (acceptance, overlap share, batch efficiency) come from combo_model.py and stay tagged [E].
Quality: only the OLMoE perplexity losses measured earlier are used (top-6 +8.8%, top-4 +45.6%, Q3_K_S +10.4%, Q2_K +26%, requantised so upper bounds); everything else is marked unmeasured.
usage: combo_trace_sim.py [--out rows.json] [--top N]"""
import collections, heapq, itertools, json, math, multiprocessing as mp, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import combo_model as cm

TRACES = ["qwen3-30b_db", "qwen3-30b_story", "qwen3-30b_code"]
LAYERS, TOP, SLICES_125 = 48, 8, 480
FOOT_ENTRIES = LAYERS * TOP                                   # accesses of one token at full top-8
CACHE_GIB = {8: 2.5, 16: 9.15, 32: 25.15}                     # [E] expert cache GiB left on an 8/16/32 GB machine (plan F)
PPL = {"sel": {"full": 0.0, "top6": 0.088, "top4": 0.456, "aware6": None}, "bits": {4.3: 0.0, 3.5: None, 3.0: 0.104, 2.5: 0.26}}   # OLMoE measured relative perplexity loss; None = unmeasured
def load(name, root=None):
    root = root or os.path.join(HERE, "..", "data", "traces"); L = [l.split() for l in open(os.path.join(root, name + ".txt")) if l.startswith("blk.")]
    while L and not L[0][0].startswith("blk.0."): L = L[1:]
    L = L[: len(L) // LAYERS * LAYERS]
    return [[[int(x) for x in L[t * LAYERS + l][1:]] for l in range(LAYERS)] for t in range(len(L) // LAYERS)]   # tokens[t][layer] = ranked expert ids

def skipped(skip):                                            # layers dropped: spread evenly, never layer 0 or the last
    if not skip: return frozenset()
    n = round(LAYERS * skip); step = LAYERS / n; return frozenset(min(LAYERS - 2, max(1, int(step * (i + 0.5)))) for i in range(n))

def select(ranked, mode, cached):
    if mode == "full": return ranked
    if mode == "top6": return ranked[:6]
    if mode == "top4": return ranked[:4]
    if mode == "aware6":                                      # first 3 by rank are mandatory, then cached ones first, then uncached by rank
        keep = ranked[:3]; rest = ranked[3:]; keep += [e for e in rest if e in cached][: 3]
        keep += [e for e in rest if e not in cached and e not in keep][: 6 - len(keep)]; return keep
    raise ValueError(mode)

class Cache:
    def __init__(self, cap, policy, pinned=()):
        self.cap, self.policy, self.pin = max(1, cap), policy, set(pinned)
        npin = len(self.pin) if policy == "pin" else 0; self.free = max(1, self.cap - npin)
        self.a = collections.OrderedDict(); self.b = collections.OrderedDict()                       # slru: b = protected
        self.protect = int(self.free * 0.7)
    def __contains__(self, k): return (self.policy == "pin" and k in self.pin) or k in self.a or k in self.b
    def touch(self, k):
        if self.policy == "pin" and k in self.pin: return True
        if self.policy == "slru":
            if k in self.b: self.b.move_to_end(k); return True
            if k in self.a:
                del self.a[k]; self.b[k] = 1
                while len(self.b) > self.protect: old, _ = self.b.popitem(last=False); self.a[old] = 1
                return True
            self.a[k] = 1
            while len(self.a) + len(self.b) > self.free: (self.a if self.a else self.b).popitem(last=False)
            return False
        if k in self.a: self.a.move_to_end(k); return True
        self.a[k] = 1
        while len(self.a) > self.free: self.a.popitem(last=False)
        return False

def belady(stream, cap):
    nxt = [0] * len(stream); last = {}
    for i in range(len(stream) - 1, -1, -1): nxt[i] = last.get(stream[i], 1 << 60); last[stream[i]] = i
    inc = {}; heap = []; miss = 0
    for i, k in enumerate(stream):
        if k in inc: inc[k] = nxt[i]; heapq.heappush(heap, (-nxt[i], k)); continue
        miss += 1
        if len(inc) >= cap:
            while True:
                n, v = heapq.heappop(heap)
                if v in inc and inc[v] == -n: del inc[v]; break
        inc[k] = nxt[i]; heapq.heappush(heap, (-nxt[i], k))
    return miss

def hot_pins(names, n, exclude):                              # held-out: most-used (layer, expert) pairs of the OTHER prompts
    c = collections.Counter()
    for nm in names:
        if nm == exclude: continue
        for tok in load(nm):
            for l, ids in enumerate(tok):
                for e in ids: c[(l, e)] += 1
    return [k for k, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]

_TR = {}; _PIN = {}
def tr(name):
    if name not in _TR: _TR[name] = load(name)
    return _TR[name]
def pins(name, n):
    if (name, n) not in _PIN: _PIN[(name, n)] = hot_pins(TRACES, n, name)
    return _PIN[(name, n)]

def simulate(trace, policy, sel, bits, ram, spec, skip):
    """Returns dict(tokens, passes, accesses, misses). cap in entries = footprints x 384, with more entries when experts are smaller."""
    fp = CACHE_GIB[ram] * 1.074e9 / (2.7e6 * (bits / 4.3)) / SLICES_125                            # footprints of the 125B at this bit width
    cap = int(fp * FOOT_ENTRIES); sk = skipped(skip); T = tr(trace); k = max(1, spec)
    if policy == "belady":
        stream = []
        for g in range(0, len(T), k):
            for l in range(LAYERS):
                if l in sk: continue
                seen = []
                for tok in T[g:g + k]:
                    for e in select(tok[l], sel if sel != "aware6" else "top6", ()): 
                        if (l, e) not in seen: seen.append((l, e))
                stream += seen
        return dict(tokens=len(T), passes=-(-len(T) // k), accesses=len(stream), misses=belady(stream, cap))
    c = Cache(cap, policy, pins(trace, int(0.10 * cap)) if policy == "pin" else ()); acc = mis = 0
    for g in range(0, len(T), k):
        for l in range(LAYERS):
            if l in sk: continue
            seen = []
            for tok in T[g:g + k]:
                cached = {e for e in tok[l] if (l, e) in c}
                for e in select(tok[l], sel, cached):
                    if (l, e) not in seen: seen.append((l, e))
            for key in seen:
                acc += 1
                if not c.touch(key): mis += 1
    return dict(tokens=len(T), passes=-(-len(T) // k), accesses=acc, misses=mis)

def toks(sim, sel, bits, spec, skip, overlap, scen, ram_rate=cm.R_EXPL, ceiling_cpu=cm.G_CPU):
    """Tokens/s for a simulated combination. Miss bytes scale from the 30B trace (384 slices/token) to the 125B (480 x selected fraction)."""
    f = {"full": 1.0, "top6": 0.75, "top4": 0.5, "aware6": 0.75}[sel]; sc = cm.sc_at(scen); k = max(1, spec); ex = 1 - skip
    miss_per_pass = sim["misses"] / sim["passes"]; bytes_ = miss_per_pass * (SLICES_125 / FOOT_ENTRIES) * 2.7e6 * (bits / 4.3)
    io = bytes_ / 1e9 / ram_rate
    flop_exp = cm.FLOP_TOT * (1 - cm.DENSE_FRAC) * f * ex; flop_dense = cm.FLOP_TOT * cm.DENSE_FRAC * ex
    be = {1: 1.0, 2: (1 + sc["b4"]) / 2 if False else 1.15, 4: sc["b4"], 8: sc["b8"]}.get(k, 1.0)
    comp = k * (flop_exp + flop_dense) / (ceiling_cpu * be)
    if overlap:
        phi = sc["phi"]; t = min(comp + io, max(comp, phi * io / 0.43) + (1 - phi) * io)   # an adaptive system stops prefetching when wasted bytes would cost more than they hide
    else: t = comp + io
    E = (1 - sc["alpha"] ** k) / (1 - sc["alpha"]) if k > 1 else 1.0
    if k > 1: t *= 1.04
    return E / t

def ppl_cost(sel, bits, skip=0.0):
    a, b = PPL["sel"][sel], PPL["bits"][bits]
    return None if a is None or b is None or skip else (1 + a) * (1 + b) - 1

POLICIES = ("lru", "slru", "pin"); SELS = ("full", "top6", "top4", "aware6"); BITS = (4.3, 3.5, 3.0); SPECS = (0, 2, 4); SKIPS = (0.0, 0.25); OVERLAPS = (False, True); RAMS = (8, 16, 32)
def grid():
    return itertools.product(TRACES, POLICIES, SELS, BITS, SPECS, SKIPS, RAMS)
def sim_job(a):
    trace, pol, sel, bits, spec, skip, ram = a; return a, simulate(trace, pol, sel, bits, ram, spec, skip)

def collect():
    with mp.Pool() as pool: sims = dict(pool.map(sim_job, list(grid()), chunksize=8))
    rows = []
    for (trace, pol, sel, bits, spec, skip, ram), s in sims.items():
        for ov in OVERLAPS:
            rows.append(dict(trace=trace, policy=pol, sel=sel, bits=bits, spec=spec, skip=skip, ram=ram, overlap=ov, hit=1 - s["misses"] / max(1, s["accesses"]), misses_per_pass=s["misses"] / s["passes"],
                             **{n: toks(s, sel, bits, spec, skip, ov, i) for n, i in (("cons", 0), ("likely", 1), ("opt", 2))}, ppl=ppl_cost(sel, bits, skip)))
    return rows

def label(r):
    p = [r["policy"]]
    if r["sel"] != "full": p.append(r["sel"])
    if r["bits"] != 4.3: p.append(f"{r['bits']}bit")
    if r["spec"]: p.append(f"spec{r['spec']}")
    if r["skip"]: p.append(f"skip{int(r['skip']*100)}")
    if r["overlap"]: p.append("overlap")
    return "+".join(p)

if __name__ == "__main__":
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None; top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 6
    rows = collect(); print(f"{len(rows)} combination runs ({len(rows)//len(TRACES)} per trace) replayed on {len(TRACES)} real traces")
    if out: json.dump(rows, open(out, "w"))
    agg = collections.defaultdict(list)
    for r in rows: agg[(label(r), r["ram"])].append(r)
    mean = lambda v: sum(v) / len(v)
    tab = {k: dict(likely=mean([x["likely"] for x in v]), cons=mean([x["cons"] for x in v]), opt=mean([x["opt"] for x in v]), hit=mean([x["hit"] for x in v]), ppl=v[0]["ppl"]) for k, v in agg.items()}
    for ram in RAMS:
        base = tab[("lru", ram)]["likely"]; print(f"\n=== {ram} GB (mean of 3 traces): base path (LRU, full top-k, 4.3 bit) = {base:.2f} tok/s likely")
        for name, flt in (("quality-safe (no lossy lever)", lambda k, v: v["ppl"] == 0.0), ("measured loss <= 12% perplexity", lambda k, v: v["ppl"] is not None and v["ppl"] <= 0.12), ("any, loss measured or unmeasured", lambda k, v: True)):
            sel = sorted([(k, v) for k, v in tab.items() if k[1] == ram and flt(k, v)], key=lambda kv: -kv[1]["likely"])[:top]
            print(f"  -- {name}:")
            for (lab, _), v in sel: print(f"     {lab:<42} cons {v['cons']:5.2f} likely {v['likely']:5.2f} opt {v['opt']:5.2f} | x base {v['likely']/base:4.2f} | hit {v['hit']:.2f} | ppl {'?' if v['ppl'] is None else f'+{v[chr(112)+chr(112)+chr(108)]*100:.0f}%'}")

LEVERS = {"slru": ("policy", "slru"), "pin": ("policy", "pin"), "top6": ("sel", "top6"), "top4": ("sel", "top4"), "aware6": ("sel", "aware6"), "bits3.5": ("bits", 3.5), "bits3": ("bits", 3.0),
          "spec2": ("spec", 2), "spec4": ("spec", 4), "skip25": ("skip", 0.25), "overlap": ("overlap", True)}
BASE = dict(policy="lru", sel="full", bits=4.3, spec=0, skip=0.0, overlap=False)
def table(rows, field="likely"):
    agg = collections.defaultdict(list)
    for r in rows: agg[(r["policy"], r["sel"], r["bits"], r["spec"], r["skip"], r["overlap"], r["ram"])].append(r[field])
    return {k: sum(v) / len(v) for k, v in agg.items()}
def interactions(rows, ram, field="likely", tol=1.03):
    """For every pair of levers from different families: combined gain vs product of the two single gains. Returns (single gains, list of (a, b, combined, product, ratio))."""
    m = table(rows, field); val = lambda cfg: m[(cfg["policy"], cfg["sel"], cfg["bits"], cfg["spec"], cfg["skip"], cfg["overlap"], ram)]
    b = val(BASE); g = {l: val({**BASE, k: v}) / b for l, (k, v) in LEVERS.items()}; out = []
    for (l1, (k1, v1)), (l2, (k2, v2)) in itertools.combinations(LEVERS.items(), 2):
        if k1 == k2: continue
        c = val({**BASE, k1: v1, k2: v2}) / b; out.append((l1, l2, c, g[l1] * g[l2], c / (g[l1] * g[l2])))
    return g, out
