#!/usr/bin/env python3
"""Published-technique family tests, group C: screen 1 #16..#21 and screen 2 #0, #1.

Each fam_* function recomputes the decisive free test of one family on the three real Qwen3-30B-A3B router traces
(data/traces/qwen3-30b_{db,story,code}.txt: 128 experts, top-8, 48 layers, 250-300 decode tokens each) and returns a dict of numbers.
Caveats that apply to every family: the 30B traces stand in for the 125B model (512 experts, top-10, ~2.7 MB per expert at 4.3 bpw);
cache sizes are given in token-footprints (cache entries = footprints x 384 accesses per token at top-8); machines with 8/16/32 GB have about
2.5/9.15/25.15 GiB of expert cache = 2.1/7.6/20.8 footprints of the 125B. Disk and compute rates are the earlier Mac measurements named below.
Disk micro-benchmarks (real reads of a large local file, not deterministic) live in bench_* functions and are run by the command line, never by fam_*.

usage: f_C.py famname [famname ...]      run families and print JSON
       f_C.py bench-pread GGUF [tokens] [threads]    replay a trace as cache-bypassing preads of the real expert slices
       f_C.py bench-layout GGUF [pairs]               3 x one-third-size scattered reads vs one contiguous read per expert
       f_C.py bench-threads FILE [size_mb]            read rate against threads in flight
"""
import collections, json, math, os, random, struct, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); TOOLS = os.path.join(HERE, ".."); sys.path.insert(0, TOOLS)
import cachesim, combo_trace_sim as cs, trace_tools

TRACES = ["qwen3-30b_db", "qwen3-30b_story", "qwen3-30b_code"]
NL, E30, K30 = 48, 128, 8                          # 30B trace geometry
E125, K125, MB125 = 512, 10, 2.7                   # 125B target geometry and expert size (MB)
SLICES30, SLICES125 = NL * K30, NL * K125          # accesses per token: 384 / 480
GB_TOK = SLICES125 * MB125 / 1000                  # 1.296 GB of expert bytes per token at hit 0
T_COMP = 12.0 / 55.0                               # [M] seconds of compute per token: 12 GFLOP at the 55 GFLOP/s measured decode rate = 0.218 s (4.58 tok/s)
RATE_4MIB, RATE_1MIB, RATE_MMAP, RATE_EXPL4 = 2.38, 1.56, 0.34, 3.11   # [M] GB/s on the Mac NVMe (read_bench.py): 4 MiB / 1 MiB reads at queue depth 1, mmap paging, 2.7 MB x 4 in flight
RATE_POINTS = [(0.9, 1.5), (1.048576, 1.56), (4.194304, 2.38)]       # (read size MB, GB/s) used to turn a read size into a rate
FOOT = {8: 2.1, 16: 7.6, 32: 20.8}                 # token-footprints of expert cache on 8/16/32 GB machines
RECORDED = dict(pread_layer_batch_8threads=2.48, pread_layer_batch_unmerged=2.72, mmap_cold=0.277, scattered_triple_3x0p885mb=2.9, contiguous_triple_2p85mb=3.8,
                threads_2p7mb={1: 1.63, 4: 3.04, 8: 3.70, 16: 4.14})   # GB/s, bench_* output on this Mac's internal NVMe (shared machine, run-to-run spread about 15 %), recorded when the report was written
PREFILL_GFLOPS = 311.0                             # [M] batched prefill GFLOP/s on OLMoE (earlier llama-bench measurement)

# ---------------------------------------------------------------- common helpers
_TOK = {}
def tokens(name):
    """tokens[t][layer] = ranked expert ids (list of 8) for one trace, cached."""
    if name not in _TOK: _TOK[name] = cs.load(name)
    return _TOK[name]

def popcount(x): return bin(x).count("1")

_MASK = {}
def masks(name):
    """masks[t][layer] = bitmask of the experts of that token and layer."""
    if name not in _MASK:
        _MASK[name] = [[sum(1 << e for e in ids) for ids in tok] for tok in tokens(name)]
    return _MASK[name]

_LRU = {}
def lru_hit(name, foot):
    """LRU hit rate of one trace with a cache of `foot` token-footprints (cold start included)."""
    k = (name, round(foot, 6))
    if k not in _LRU:
        seq = [(l, e) for tok in tokens(name) for l in range(NL) for e in tok[l]]
        _LRU[k] = cachesim.lru_hits(seq, int(foot * SLICES30)) / len(seq)
    return _LRU[k]

def mean_hit(foot): return sum(lru_hit(n, foot) for n in TRACES) / len(TRACES)

def rate_for_size(mb):
    """Disk rate (GB/s) for reads of `mb` MB: log-linear through RATE_POINTS, clamped at the ends."""
    pts = RATE_POINTS
    if mb <= pts[0][0]: return pts[0][1]
    if mb >= pts[-1][0]: return pts[-1][1]
    for (a, ra), (b, rb) in zip(pts, pts[1:]):
        if a <= mb <= b: return ra + (rb - ra) * (math.log(mb) - math.log(a)) / (math.log(b) - math.log(a))

def tok_s(hit, rate, comp=T_COMP, gb_tok=GB_TOK):
    """Return (serial, overlap) tokens/s: serial 1/(io+compute), overlap min(1/compute, rate/bytes)."""
    io = (1 - hit) * gb_tok / rate
    return 1.0 / (io + comp), (1.0 / max(io, comp) if io > 0 else 1.0 / comp)

def unique_random(E, k, n):
    """Expected distinct experts when n tokens each pick k of E uniformly at random (independent)."""
    return E * (1.0 - (1.0 - k / E) ** n)

def r3(x): return round(x, 3)

# ---------------------------------------------------------------- batching of independent streams (#16, #17)
def union_mean(specs, nsteps, mode="offset"):
    """Mean distinct experts per layer when the tokens of several streams are served together.
    specs: list of (trace name, start offset); at step t stream i contributes token (offset+t) mod len(trace). Returns (mean unique per layer, per-stream k)."""
    tot = 0; n = 0
    for t in range(nsteps):
        ms = [masks(nm)[(off + t) % len(masks(nm))] for nm, off in specs]
        for l in range(NL):
            u = 0
            for m in ms: u |= m[l]
            tot += popcount(u); n += 1
    return tot / n

def sample_specs(K, kind, rng):
    """K streams: 'same' = all from one trace at distinct random offsets (optimistic: one prompt), 'mixed' = round-robin over the three traces, distinct offsets inside each."""
    if kind == "same":
        nm = TRACES[rng.randrange(3)]; offs = rng.sample(range(len(tokens(nm))), K)
        return [(nm, o) for o in offs]
    out = []; used = {n: set() for n in TRACES}
    for i in range(K):
        nm = TRACES[i % 3]; o = rng.randrange(len(tokens(nm)))
        while o in used[nm]: o = rng.randrange(len(tokens(nm)))
        used[nm].add(o); out.append((nm, o))
    return out

def sharing_curve(Ks, kind, draws, nsteps, seed):
    """sharing[K] = K*8 / mean unique experts per layer, averaged over `draws` random stream sets."""
    rng = random.Random(seed); out = {}
    for K in Ks:
        s = []
        for _ in range(draws):
            specs = sample_specs(K, kind, rng); s.append(K * K30 / union_mean(specs, nsteps))
        out[K] = sum(s) / len(s)
    return out

def aligned_streams(names, nsteps):
    """The screen's first check: unrelated traces served at the same step index (offset 0). Returns sharing = 8*B / unique per layer."""
    return len(names) * K30 / union_mean([(n, 0) for n in names], nsteps)

def fam_s1_16():
    """Serve K independent prompts in lock-step: does one read per expert serve all K? Measured sharing on real traces, extrapolated to 512 experts / top-10."""
    n = min(len(tokens(t)) for t in TRACES)
    Ks = (2, 4, 8, 16, 32)
    same = sharing_curve(Ks, "same", 12, 100, 16001); mixed = sharing_curve(Ks, "mixed", 12, 100, 16002)
    rand128 = {K: K * K30 / unique_random(E30, K30, K) for K in Ks}
    two = aligned_streams(["qwen3-30b_db", "qwen3-30b_story"], n); three = aligned_streams(TRACES, n)
    ext = {}                                     # 125B at hit 0 with explicit 2.4 GB/s reads, per layer unique = 512(1-(1-10/512)^K)
    base = GB_TOK / RATE_4MIB                    # seconds per token alone
    for K in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        u = unique_random(E125, K125, K); step_gb = u * NL * MB125 / 1000; step_s = step_gb / RATE_4MIB
        ext[K] = dict(unique_per_layer=r3(u), step_gb=r3(step_gb), agg_tok_s=r3(K / step_s), per_user_tok_s=r3(1 / step_s), sharing=r3(K * K125 / u),
                      agg_gain_vs_single_stream_same_disk=r3((K / step_s) / (1 / base)))
    return dict(streams_aligned_2_unrelated=r3(two), streams_aligned_3_unrelated=r3(three), K=list(Ks),
                sharing_same_domain_offsets={K: r3(v) for K, v in same.items()}, sharing_mixed_domains={K: r3(v) for K, v in mixed.items()},
                sharing_random_128={K: r3(v) for K, v in rand128.items()}, extrapolation_125B=ext,
                kill_same_domain_k8_below_1p5=bool(same[8] < 1.5), best_path_aggregate_gain_k8=ext[8]["agg_gain_vs_single_stream_same_disk"],
                best_path_aggregate_gain_k32=ext[32]["agg_gain_vs_single_stream_same_disk"], per_user_fraction_k32=r3(ext[32]["per_user_tok_s"] * base))

def random_positions_factor(B, mode, draws, seed):
    """Mean over traces/draws/layers of distinct experts / (8B) when B tokens are processed together.
    mode 'same': B distinct random positions of one trace; 'mixed': one position in each of B traces drawn from the three (round robin); 'consec': B consecutive tokens."""
    rng = random.Random(seed); fac = []
    for nm in TRACES:
        M = masks(nm); T = len(M)
        for _ in range(draws):
            if mode == "same": pos = [(nm, p) for p in rng.sample(range(T), B)]
            elif mode == "consec": s = rng.randrange(T - B + 1); pos = [(nm, s + i) for i in range(B)]
            else:
                pos = []
                for i in range(B):
                    n2 = TRACES[(TRACES.index(nm) + i) % 3]; pos.append((n2, rng.randrange(len(masks(n2)))))
            tot = 0
            for l in range(NL):
                u = 0
                for n2, p in pos: u |= masks(n2)[p][l]
                tot += popcount(u)
            fac.append(tot / NL / (K30 * B))
    return sum(fac) / len(fac)

def batch_lru_misses(specs, nsteps, foot):
    """B streams served together with ONE shared LRU cache of `foot` single-token footprints: per step and layer the union of the streams' experts is touched once.
    Returns (misses per step, distinct accesses per step)."""
    cap = int(foot * SLICES30); c = collections.OrderedDict(); mis = acc = 0
    for t in range(nsteps):
        ms = [masks(nm)[(off + t) % len(masks(nm))] for nm, off in specs]
        for l in range(NL):
            u = 0
            for m in ms: u |= m[l]
            e = 0
            while u:
                if u & 1:
                    k = (l, e); acc += 1
                    if k in c: c.move_to_end(k)
                    else:
                        mis += 1; c[k] = None
                        if len(c) > cap: c.popitem(last=False)
                u >>= 1; e += 1
    return mis / nsteps, acc / nsteps

def cached_batch_gain(B, kind, foot, draws, seed, nsteps=100):
    """Aggregate tok/s gain of serving B streams together over serving ONE stream with the same cache, both at 2.4 GB/s, flat compute (serial and overlap). Mean over random stream sets."""
    rng = random.Random(seed); gs = []; go = []
    for _ in range(draws):
        specs = sample_specs(B, kind, rng); m1, _ = batch_lru_misses(specs[:1], nsteps, foot); mb, _ = batch_lru_misses(specs, nsteps, foot)
        sc = SLICES125 / SLICES30 * MB125 / 1000 / RATE_4MIB              # seconds of io per missed 30B-entry mapped to the 125B
        t1 = m1 * sc; tb = mb * sc
        s1 = 1 / (t1 + T_COMP); sb = B / (tb + B * T_COMP); o1 = 1 / max(t1, T_COMP); ob = B / max(tb, B * T_COMP)
        gs.append(sb / s1); go.append(ob / o1)
    return sum(gs) / len(gs), sum(go) / len(go)

def batch_throughput(B, f, hit_best):
    """125B aggregate tokens/s at hit 0 with explicit 4 MiB reads for B tokens served together (union factor f); compute does not shrink with batching (flat, conservative).
    Returns (serial, overlap, gain vs best path serial, gain vs best path overlap); best path = the same machine's LRU hit with single-token decode."""
    io = B * GB_TOK * f / RATE_4MIB; comp = B * T_COMP
    ser, ovl = B / (io + comp), B / max(io, comp)
    bs, bo = tok_s(hit_best, RATE_4MIB)
    return ser, ovl, ser / bs, ovl / bo

def fam_s1_17():
    """Process B tokens together layer by layer: measured union factor (distinct experts / (8B)) on the real traces, same-domain vs mixed, and the aggregate speedup it implies."""
    Bs = (2, 4, 8, 16, 32)
    same = {B: random_positions_factor(B, "same", 30, 17001 + B) for B in Bs}
    mixed = {B: random_positions_factor(B, "mixed", 30, 17101 + B) for B in Bs}
    consec = {B: random_positions_factor(B, "consec", 30, 17201 + B) for B in (4, 8, 16)}
    rnd128 = {B: unique_random(E30, K30, B) / (K30 * B) for B in Bs}
    rnd512 = {B: unique_random(E125, K125, B) / (K125 * B) for B in Bs}
    h8 = mean_hit(FOOT[8]); rows = {}
    for B in (2, 4, 8):
        for label, f in (("same_domain_real128", same[B]), ("mixed_real128", mixed[B]), ("random512", rnd512[B])):
            s, o, gs, go = batch_throughput(B, f, h8); rows[f"B{B}_{label}"] = dict(factor=r3(f), agg_serial=r3(s), agg_overlap=r3(o), gain_vs_best_serial=r3(gs), gain_vs_best_overlap=r3(go), per_user_serial=r3(s / B))
    bs, bo = tok_s(h8, RATE_4MIB); cg = {}
    for B in (2, 4, 8):
        for kind in ("same", "mixed"):
            gs, go = cached_batch_gain(B, kind, FOOT[8], 5, 17300 + B, 80)
            cg[f"B{B}_{kind}"] = dict(serial_gain_same_cache=r3(gs), overlap_gain_same_cache=r3(go))
    return dict(B=list(Bs), cached_gain_8gb=cg, factor_same_domain={B: r3(v) for B, v in same.items()}, factor_mixed={B: r3(v) for B, v in mixed.items()}, factor_consecutive={B: r3(v) for B, v in consec.items()},
                factor_random_128={B: r3(v) for B, v in rnd128.items()}, factor_random_512={B: r3(v) for B, v in rnd512.items()},
                best_path_8gb_hit=r3(h8), best_path_8gb_serial=r3(bs), best_path_8gb_overlap=r3(bo), throughput=rows,
                killed_by_b4_same_above_0p85=bool(same[4] > 0.85))

# ---------------------------------------------------------------- GGUF header, read plans, replays (#18, #19, s2 #1)
def read_gguf_tensors(path):
    """Minimal GGUF v2/v3 header reader. Returns {name: dict(ne=[...], type=int, offset=absolute byte offset, nbytes=bytes up to the next tensor)}."""
    def rd(f, fmt): sz = struct.calcsize(fmt); return struct.unpack("<" + fmt, f.read(sz))[0]
    def rstr(f): n = rd(f, "Q"); return f.read(n).decode("utf-8", "replace")
    def rval(f, t):
        if t == 8: return rstr(f)
        if t == 9:
            et = rd(f, "I"); n = rd(f, "Q"); return [rval(f, et) for _ in range(n)]
        fmt = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}[t]; return rd(f, fmt)
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF": raise ValueError("not a GGUF file")
        ver = rd(f, "I"); nt = rd(f, "Q"); nkv = rd(f, "Q"); align = 32
        if ver < 2: raise ValueError("unsupported GGUF version")
        for _ in range(nkv):
            k = rstr(f); t = rd(f, "I"); v = rval(f, t)
            if k == "general.alignment": align = int(v)
        infos = []
        for _ in range(nt):
            nm = rstr(f); nd = rd(f, "I"); ne = [rd(f, "Q") for _ in range(nd)]; ty = rd(f, "I"); off = rd(f, "Q"); infos.append((nm, ne, ty, off))
        pos = f.tell(); data0 = (pos + align - 1) // align * align; fsize = os.path.getsize(path)
    order = sorted(range(len(infos)), key=lambda i: infos[i][3]); out = {}
    for j, i in enumerate(order):
        nm, ne, ty, off = infos[i]; end = infos[order[j + 1]][3] if j + 1 < len(order) else fsize - data0
        out[nm] = dict(ne=ne, type=ty, offset=data0 + off, nbytes=end - off)
    return out

def expert_table(tensors, layers=NL):
    """{(layer, tensor kind): (absolute offset, bytes per expert slice, n_experts)} for ffn_gate/up/down_exps; the slice of expert e starts at offset + e*slice."""
    t = {}
    for l in range(layers):
        for kind in ("gate", "up", "down"):
            d = tensors.get(f"blk.{l}.ffn_{kind}_exps.weight")
            if d: ne = d["ne"][2] if len(d["ne"]) > 2 else 1; t[(l, kind)] = (d["offset"], d["nbytes"] // ne, ne)
    return t

def runs_of(ids, gap=0):
    """Group sorted slot numbers into maximal runs; two slots join when they differ by at most gap+1. Returns list of (first, last)."""
    out = []
    for p in sorted(ids):
        if out and p - out[-1][1] <= gap + 1: out[-1][1] = p
        else: out.append([p, p])
    return [tuple(r) for r in out]

def plan_reads(toks, table, merge=True, kinds=("gate", "up", "down")):
    """One list per (token, layer): the (offset, length) reads that fetch the selected expert slices. merge=True joins adjacent expert ids of one tensor into one read."""
    plan = []
    for tok in toks:
        for l in range(len(tok)):
            reads = []
            for kind in kinds:
                off, sl, ne = table[(l, kind)]
                rr = runs_of(tok[l]) if merge else [(e, e) for e in sorted(tok[l])]
                for a, b in rr: reads.append((off + a * sl, (b - a + 1) * sl))
            plan.append(reads)
    return plan

def plan_stats(plan):
    reads = sum(len(g) for g in plan); byt = sum(n for g in plan for _, n in g); return dict(groups=len(plan), reads=reads, bytes=byt)

def make_synthetic_gguf(path, layers=2, experts=8, slice_bytes=4096, align=32):
    """Write a tiny valid GGUF with blk.N.ffn_{gate,up,down}_exps.weight tensors (uint8 payload, pattern = byte offset mod 251) for tests and benchmark dry runs."""
    def s(x): b = x.encode(); return struct.pack("<Q", len(b)) + b
    infos = b""; off = 0; layout = []
    for l in range(layers):
        for kind in ("gate", "up", "down"):
            nb = experts * slice_bytes
            infos += s(f"blk.{l}.ffn_{kind}_exps.weight") + struct.pack("<I", 3) + struct.pack("<QQQ", slice_bytes, 1, experts) + struct.pack("<IQ", 24, off)   # type 24 = I8, size = elements
            layout.append(nb); off += (nb + align - 1) // align * align
    head = b"GGUF" + struct.pack("<IQQ", 3, layers * 3, 1) + s("general.alignment") + struct.pack("<II", 4, align)
    pos = len(head) + len(infos); pad = (-pos) % align
    with open(path, "wb") as f:
        f.write(head + infos + b"\0" * pad); o = 0
        for nb in layout:
            f.write(bytes((o + i) % 251 for i in range(nb))); o += nb
            f.write(b"\0" * ((-nb) % align)); o += (-nb) % align
    return path

def _open_nocache(path):
    fd = os.open(path, os.O_RDONLY)
    if sys.platform == "darwin":
        import fcntl; fcntl.fcntl(fd, 48, 1)                  # F_NOCACHE
    else:
        pass                                                   # Linux would need O_DIRECT with aligned buffers; the benchmark is documented for the Mac
    return fd

def replay_pread(path, plan, threads=8):
    """Replay a read plan against a real file: for each group (one token-layer) all its reads are issued concurrently on `threads` workers, then the group is awaited.
    Returns dict(bytes, reads, seconds, gb_s). Cache-bypassing (F_NOCACHE). Offsets and lengths are padded to 4 KiB for the syscall; `bytes` counts requested bytes."""
    from concurrent.futures import ThreadPoolExecutor
    fd = _open_nocache(path); tot = 0; nr = 0
    def one(r):
        off, n = r; a = off // 4096 * 4096; b = (off + n + 4095) // 4096 * 4096; d = os.pread(fd, b - a, a); return min(len(d), n)
    t0 = time.perf_counter()
    try:
        with ThreadPoolExecutor(threads) as ex:
            for g in plan:
                for got in ex.map(one, g): tot += got
                nr += len(g)
    finally: os.close(fd)
    dt = time.perf_counter() - t0
    return dict(bytes=tot, reads=nr, seconds=dt, gb_s=tot / dt / 1e9 if dt > 0 else 0.0)

def replay_mmap(path, plan):
    """Same plan through a read-only mmap, touching one byte per 16 KiB page (demand paging). Returns dict(bytes, seconds, gb_s); bytes = requested bytes."""
    import mmap
    fd = os.open(path, os.O_RDONLY); mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ); tot = 0; s = 0; t0 = time.perf_counter()
    try:
        for g in plan:
            for off, n in g:
                for p in range(off, off + n, 16384): s += mm[p]
                tot += n
    finally: mm.close(); os.close(fd)
    dt = time.perf_counter() - t0
    return dict(bytes=tot, seconds=dt, gb_s=tot / dt / 1e9 if dt > 0 else 0.0)

def bench_pread(gguf, ntok=12, threads=8, trace="qwen3-30b_db", start=100, with_mmap=True, mmap_start=200):
    """Decisive test of #18: replay real routing decisions as explicit preads of the real expert slices of the Qwen3-30B file."""
    tab = expert_table(read_gguf_tensors(gguf)); T = tokens(trace)
    merged = plan_reads(T[start:start + ntok], tab, merge=True); single = plan_reads(T[start:start + ntok], tab, merge=False)
    out = dict(merged_plan=plan_stats(merged), unmerged_plan=plan_stats(single), merge_saving_reads=1 - plan_stats(merged)["reads"] / plan_stats(single)["reads"])
    out["pread_merged"] = replay_pread(gguf, merged, threads); out["pread_unmerged"] = replay_pread(gguf, single, threads)
    if with_mmap: out["mmap_cold"] = replay_mmap(gguf, plan_reads(T[mmap_start:mmap_start + max(2, ntok // 4)], tab, merge=True))
    return out

def bench_layout(gguf, pairs=300, threads=4, scratch_dir=None, seed=1, reps=3):
    """Decisive test of #19 part 2: random (layer, expert) triples read as three scattered reads vs the same bytes as ONE contiguous read from a scratch copy of the triples
    (emulating the repack without rewriting the model). The scratch file holds `pairs` triples and is removed afterwards. Returns GB/s of both."""
    import tempfile
    tab = expert_table(read_gguf_tensors(gguf)); rng = random.Random(seed); ne = tab[(0, "gate")][2]
    picks = [(rng.randrange(NL), rng.randrange(ne)) for _ in range(pairs)]
    scat = [[(tab[(l, k)][0] + e * tab[(l, k)][1], tab[(l, k)][1]) for k in ("gate", "up", "down")] for l, e in picks]
    flat = [r for g in scat for r in g]
    fd = _open_nocache(gguf); scratch = tempfile.mkdtemp(dir=scratch_dir); sp = os.path.join(scratch, "triples.bin"); sizes = []
    try:
        wfd = os.open(sp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            if sys.platform == "darwin":
                import fcntl; fcntl.fcntl(wfd, 48, 1)              # F_NOCACHE on the writer too, so the scratch copy does not sit in the page cache
            for g in scat:
                n = 0
                for off, ln in g: d = os.pread(fd, ln, off); os.write(wfd, d); n += len(d)
                sizes.append(n)
            os.fsync(wfd)
        finally: os.close(wfd)
        offs = []; o = 0
        for n in sizes: offs.append((o, n)); o += n
        order = list(range(pairs)); rng.shuffle(order)
        res = dict(pairs=pairs, triple_mb=r3(sum(sizes) / pairs / 1e6), slice_mb=r3(scat[0][0][1] / 1e6))
        # scattered: the three reads of `threads` pairs in flight together (a layer batch); contiguous: one read per pair, `threads` pairs in flight. Interleaved repeats, medians reported.
        sc_runs = []; ct_runs = []; sc4 = []
        for rep in range(reps):
            rng.shuffle(order)
            sc_runs.append(replay_pread(gguf, [flat[3 * j:3 * (j + threads)] for j in range(0, pairs, threads)], 3 * threads)["gb_s"])     # `threads` pairs in flight = 3*threads requests
            sc4.append(replay_pread(gguf, [flat[3 * j:3 * (j + threads)] for j in range(0, pairs, threads)], threads)["gb_s"])          # only `threads` requests in flight
            ct_runs.append(replay_pread(sp, [[offs[i] for i in order[j:j + threads]] for j in range(0, pairs, threads)], threads)["gb_s"])
        med = lambda v: sorted(v)[len(v) // 2]
        res.update(scattered_3reads_gb_s=r3(med(sc_runs)), contiguous_1read_gb_s=r3(med(ct_runs)), scattered_runs=[r3(x) for x in sc_runs], scattered_same_request_count_gb_s=r3(med(sc4)), contiguous_runs=[r3(x) for x in ct_runs],
                   ratio_contiguous_over_scattered=r3(med(ct_runs) / med(sc_runs)))
    finally:
        os.close(fd)
        try: os.remove(sp); os.rmdir(scratch)
        except OSError: pass
    return res

def bench_threads(path, size_mb=2.7, threads_list=(1, 4, 8, 16), seconds=3.0, seed=2):
    """Read rate against reads in flight: random 4 KiB-aligned offsets of `size_mb` MB, F_NOCACHE. Returns {threads: GB/s}."""
    from concurrent.futures import ThreadPoolExecutor
    fd = _open_nocache(path); size = os.path.getsize(path); n = int(size_mb * 1e6) // 4096 * 4096; rng = random.Random(seed); out = {}
    try:
        for th in threads_list:
            got = 0; t0 = time.perf_counter()
            def one(_):
                return len(os.pread(fd, n, rng.randrange((size - n) // 4096) * 4096))
            with ThreadPoolExecutor(th) as ex:
                while time.perf_counter() - t0 < seconds:
                    got += sum(ex.map(one, range(th * 2)))
            out[th] = r3(got / (time.perf_counter() - t0) / 1e9)
    finally: os.close(fd)
    return out

def merge_stats(name):
    """Per token-layer: fraction of the 8 expert slices that are NOT the start of a new read when adjacent ids merge (reads / slices)."""
    rs = 0; sl = 0
    for tok in tokens(name):
        for ids in tok: rs += len(runs_of(ids)); sl += len(ids)
    return rs / sl

def _gguf_default():
    p = os.path.expanduser("~/Downloads/Qwen3-30B-A3B-Q4_K_M.gguf"); return p if os.path.exists(p) else None

def fam_s1_18():
    """Explicit per-layer batched pread instead of mmap, no cross-token cache: bytes per token, merging gain, hit rate lost, and what that does to tok/s versus mmap-as-LRU.
    Two read rates are used: the screen's assumption (1.5 GB/s for 0.9 MB reads) and the rate recorded by bench_pread (per-layer batches, 8 threads)."""
    merge = {n: r3(merge_stats(n)) for n in TRACES}
    foots = (0.9, 1.0, 2.0, 3.0, 4.0, 7.5)
    lru = {f: r3(mean_hit(f)) for f in foots}
    slice_mb = None; src = "assumed 0.9 MB (header not read)"; g = _gguf_default()
    if g:
        try:
            tab = expert_table(read_gguf_tensors(g)); slice_mb = sum(tab[(0, k)][1] for k in ("gate", "up", "down")) / 3e6; src = "mean of gate/up/down slices, read from the GGUF header"
        except Exception: slice_mb = None
    if slice_mb is None: slice_mb = 0.9
    rates = dict(screen_assumed=1.5, recorded_pread=RECORDED["pread_layer_batch_8threads"]); out = {}
    for rn, rate in rates.items():
        rows = {}
        for gb, fp in FOOT.items():
            h = mean_hit(fp)
            m_ser, m_ovl = tok_s(h, RATE_MMAP)                         # plain mmap acts as an LRU of the page cache, 0.34 GB/s paging (earlier measurement)
            e_ser, e_ovl = tok_s(0.0, rate)                             # per-layer buffer: no cross-token cache, hit 0
            b_ser, b_ovl = tok_s(h, RATE_4MIB)                          # our best path: explicit large reads + LRU
            rows[f"{gb}GB"] = dict(footprints=fp, lru_hit=r3(h), mmap_serial=r3(m_ser), mmap_overlap=r3(m_ovl), nocache_serial=r3(e_ser), nocache_overlap=r3(e_ovl),
                                   best_path_serial=r3(b_ser), best_path_overlap=r3(b_ovl), nocache_over_mmap_serial=r3(e_ser / m_ser), nocache_over_mmap_overlap=r3(e_ovl / m_ovl),
                                   nocache_over_best_serial=r3(e_ser / b_ser))
        out[rn] = dict(rate_gb_s=rate, io_s_per_token=r3(GB_TOK / rate), machines=rows)
    return dict(reads_per_slice_after_merge=merge, merge_saving=r3(1 - sum(merge.values()) / 3), slice_mb=r3(slice_mb), slice_source=src,
                lru_hit_by_footprints=lru, hit_below_one_footprint=lru[0.9], disk_gb_per_token_nocache=r3(GB_TOK), by_rate=out, recorded=dict(RECORDED),
                unmeasured="mmap replay of the real slices is recorded (cold, 0.277 GB/s) but the 8/16 GB page-cache hit behaviour of mmap is modelled by LRU, not measured; the 125B file does not exist here")

# ---------------------------------------------------------------- layout: co-activation chain (#19, #20)
def coact_chain(train, layer, E=E30):
    """Greedy chain order of the experts of one layer. train = list of token lists (tokens[t][layer] = ids). Start at the most frequent expert; repeatedly append the unplaced
    expert with the highest co-activation count with the last placed (ties: more frequent, then lower id); if no co-activation remains take the most frequent unplaced.
    Returns pos[e] = slot of expert e."""
    freq = [0] * E; co = collections.defaultdict(int)
    for toks in train:
        for tok in toks:
            ids = tok[layer]
            for a in ids: freq[a] += 1
            for i, a in enumerate(ids):
                for b in ids[i + 1:]: co[(min(a, b), max(a, b))] += 1
    left = set(range(E)); order = []
    cur = max(left, key=lambda e: (freq[e], -e)); order.append(cur); left.discard(cur)
    while left:
        best = max(left, key=lambda e: (co.get((min(cur, e), max(cur, e)), 0), freq[e], -e))
        cur = best; order.append(cur); left.discard(cur)
    pos = [0] * E
    for s, e in enumerate(order): pos[e] = s
    return pos

def layout_reads(toks, pos_by_layer, gap=0):
    """Mean reads per token-layer (maximal runs of adjacent slots, joining gaps <= gap) and mean slots read incl. gap fill."""
    rs = 0; sl = 0; n = 0
    for tok in toks:
        for l, ids in enumerate(tok):
            rr = runs_of([pos_by_layer[l][e] for e in ids] if pos_by_layer else ids, gap)
            rs += len(rr); sl += sum(b - a + 1 for a, b in rr); n += 1
    return rs / n, sl / n

_CHAIN = {}; _MISS = {}
def chains(train_names, spans=None, E=E30):
    """Per-layer chains trained on whole traces `train_names`, or on spans {name: (start, stop)}."""
    key = (tuple(train_names), tuple(sorted((spans or {}).items())), E)
    if key not in _CHAIN:
        tr = []
        for nm in train_names:
            T = tokens(nm); a, b = (spans or {}).get(nm, (0, len(T))); tr.append(T[a:b])
        _CHAIN[key] = [coact_chain(tr, l, E) for l in range(NL)]
    return _CHAIN[key]

def layout_study():
    """All layout scenarios of #19/#20 on the three traces. Returns dict scenario -> {trace: (reads per layer)}."""
    res = collections.defaultdict(dict)
    for i, nm in enumerate(TRACES):
        T = tokens(nm); half = len(T) // 2; prev = TRACES[(i - 1) % 3]; others = [x for x in TRACES if x != nm]
        res["identity"][nm] = layout_reads(T, None)[0]
        res["same_prompt_first_half_to_second"][nm] = layout_reads(T[half:], chains([nm], {nm: (0, half)}))[0]
        res["cross_prompt_one_other"][nm] = layout_reads(T, chains([prev]))[0]
        res["held_out_two_others"][nm] = layout_reads(T, chains(others))[0]
        res["in_sample_not_valid"][nm] = layout_reads(T, chains([nm]))[0]
    return res

def size_power(r_small, r_big, mb_small, mb_big):
    """Exponent p of a rate ~ size^p law through two measured (size, rate) points."""
    return math.log(r_big / r_small) / math.log(mb_big / mb_small)

def fam_s1_19():
    """Repack each expert's gate/up/down into one contiguous run (3 reads -> 1) and optionally cluster co-firing experts: reads per layer, read sizes, and the tok/s the layout adds.
    Disk rates: the screen's expectation (scattered 1.7, contiguous 2.38 GB/s) and the rates recorded by bench_layout on this Mac (2.9 scattered, 3.8 contiguous)."""
    st = layout_study(); mean = lambda d: sum(d.values()) / len(d)
    sc = {k: dict(per_trace={n: r3(v) for n, v in d.items()}, mean=r3(mean(d))) for k, d in st.items()}
    ident = mean(st["identity"]); cross = mean(st["cross_prompt_one_other"]); held = mean(st["held_out_two_others"])
    tri = 2.85                                                       # MB of one expert's gate+up+down in the 30B file (0.885 + 0.885 + 1.29)
    sizes = dict(stock_slice_mb=0.95, repack_triple_mb=tri, repack_triple_cluster_mb=tri * ident / cross)
    # miss-level run length with a cross-prompt chain (train on ONE other prompt), which is what remains to merge after the cache
    miss_run = {}
    for gb, fp in FOOT.items():
        rr = mm = 0
        for i, nm in enumerate(TRACES):
            ch = chains([TRACES[(i - 1) % 3]]); ms, _, _ = lru_miss_sets(nm, fp); a, m, _ = reads_of_misses(ms, ch); rr += a; mm += m
        miss_run[gb] = mm / rr
    machines = {}
    for rn, (r_s, r_c) in dict(screen_expectation=(1.7, RATE_4MIB), recorded=(RECORDED["scattered_triple_3x0p885mb"], RECORDED["contiguous_triple_2p85mb"])).items():
        p = size_power(r_s, r_c, 0.885, tri); per = {}
        for gb, fp in FOOT.items():
            h = mean_hit(fp); s0, o0 = tok_s(h, r_s); s1, o1 = tok_s(h, r_c)
            r_cl = r_c * miss_run[gb] ** p; s2, o2 = tok_s(h, r_cl)
            per[f"{gb}GB"] = dict(hit=r3(h), scatter_serial=r3(s0), scatter_overlap=r3(o0), repack_serial=r3(s1), repack_overlap=r3(o1), repack_gain_serial=r3(s1 / s0), repack_gain_overlap=r3(o1 / o0),
                                  miss_run_length_cross_prompt=r3(miss_run[gb]), cluster_rate=r3(r_cl), cluster_extra_serial=r3(s2 / s1), cluster_extra_overlap=r3(o2 / o1))
        machines[rn] = dict(rates=(r_s, r_c), size_exponent=r3(p), machines=per)
    return dict(scenarios=sc, identity_reads=r3(ident), cross_prompt_reads=r3(cross), held_out_reads=r3(held), read_reduction_cross=r3(1 - cross / ident),
                kill_cross_prompt_reads_above_6=bool(cross > 6.0), sizes_mb={k: r3(v) for k, v in sizes.items()}, by_rates=machines, recorded=dict(RECORDED),
                unmeasured="a real repacked file (the scratch-copy emulation in bench_layout stands in); 125B geometry; effect of 3 tensors interleaved on the cache granularity")

def lru_miss_sets(name, foot, pos_by_layer=None):
    """Replay the trace through an LRU cache of `foot` footprints. Returns (list per token-layer of the experts that missed, accesses, misses)."""
    if (name, foot) in _MISS: return _MISS[(name, foot)]
    cap = int(foot * SLICES30); c = collections.OrderedDict(); out = []; acc = mis = 0
    for tok in tokens(name):
        for l in range(NL):
            miss = []
            for e in tok[l]:
                k = (l, e); acc += 1
                if k in c: c.move_to_end(k)
                else:
                    mis += 1; miss.append(e); c[k] = None
                    if len(c) > cap: c.popitem(last=False)
            out.append((l, miss))
    _MISS[(name, foot)] = (out, acc, mis)
    return _MISS[(name, foot)]

def reads_of_misses(msets, pos_by_layer, gap=0):
    rs = 0; n = 0; fill = 0; mis = 0
    for l, miss in msets:
        if not miss: continue
        ids = [pos_by_layer[l][e] for e in miss] if pos_by_layer else miss
        rr = runs_of(ids, gap); rs += len(rr); n += 1; fill += sum(b - a + 1 for a, b in rr) - len(miss); mis += len(miss)
    return rs, mis, fill

def fam_s1_20():
    """Reorder experts on disk so co-firing ones are adjacent: held-out run length, gap merging without reorder, and the effect on the reads that remain after an LRU cache."""
    st = layout_study(); mean = lambda d: sum(d.values()) / len(d)
    ident = {n: st["identity"][n] for n in TRACES}; held = st["held_out_two_others"]; ins = st["in_sample_not_valid"]
    runlen = lambda reads: K30 / reads
    gapres = {}
    for g in (0, 1, 3):
        r = {nm: layout_reads(tokens(nm), None, g) for nm in TRACES}
        gapres[g] = dict(reads=r3(mean({n: v[0] for n, v in r.items()})), run_length=r3(K30 / mean({n: v[0] for n, v in r.items()})), slots_per_layer=r3(mean({n: v[1] for n, v in r.items()})))
    after = {}
    for gb, fp in FOOT.items():
        h = mean_hit(fp); sl = 0.9; stock_reads = held_reads = mis = 0
        for i, nm in enumerate(TRACES):
            others = [x for x in TRACES if x != nm]; ch = chains(others); ms, acc, m = lru_miss_sets(nm, fp)
            a, mm, _ = reads_of_misses(ms, None); b, _, _ = reads_of_misses(ms, ch)
            stock_reads += a; held_reads += b; mis += mm
        run_stock = mis / stock_reads; run_held = mis / held_reads
        r0 = rate_for_size(sl * run_stock); r1 = rate_for_size(sl * run_held)
        s0, o0 = tok_s(h, r0); s1, o1 = tok_s(h, r1)
        after[f"{gb}GB"] = dict(hit=r3(h), miss_run_length_stock=r3(run_stock), miss_run_length_reordered=r3(run_held), rate_stock=r3(r0), rate_reordered=r3(r1), disk_rate_gain=r3(r1 / r0),
                                serial_gain=r3(s1 / s0), overlap_gain=r3(o1 / o0))
    return dict(reads_identity=r3(mean(ident)), reads_held_out=r3(mean(held)), reads_in_sample=r3(mean(ins)), run_length_identity=r3(runlen(mean(ident))), run_length_held_out=r3(runlen(mean(held))),
                run_length_in_sample=r3(runlen(mean(ins))), per_trace_reads_held_out={n: r3(v) for n, v in held.items()}, per_trace_reads_identity={n: r3(v) for n, v in ident.items()},
                gap_merge_no_reorder=gapres, after_cache=after, kill_held_out_run_length_below_1p5=bool(runlen(mean(held)) < 1.5),
                unmeasured="OLMoE (64 experts) trace for the same study is measured by the command-line run recorded in the report; the 125B geometry is not measured")

# ---------------------------------------------------------------- speculative / n-gram verify (#21)
def spec_cycle_sim(name, foot, K, a, seed):
    """Replay a verify-cycle decoder on a real trace. Each cycle serves a window of K+1 consecutive trace tokens in one pass (distinct experts per layer, through an LRU cache of `foot`
    footprints); the number of tokens accepted is 1 + the run of successes of a geometric acceptance of probability a, capped at K; the window start advances by that number.
    Returns dict(tokens advanced, passes, misses, accesses)."""
    rng = random.Random(seed); T = tokens(name); n = len(T); cap = int(foot * SLICES30); c = collections.OrderedDict(); i = 0; passes = mis = acc = 0
    while i < n:
        win = T[i:i + K + 1]; passes += 1
        for l in range(NL):
            seen = []
            for tok in win:
                for e in tok[l]:
                    if e not in seen: seen.append(e)
            for e in seen:
                k = (l, e); acc += 1
                if k in c: c.move_to_end(k)
                else:
                    mis += 1; c[k] = None
                    if len(c) > cap: c.popitem(last=False)
        m = 1
        while m <= K and rng.random() < a: m += 1
        i += min(m, len(win)) if K else 1
    return dict(tokens=n, passes=passes, misses=mis, accesses=acc)

def spec_time(plain_miss_per_tok, sim, K, small_batch_gflops):
    """Seconds per accepted token with serial IO at 2.4 GB/s plus compute; miss entries scaled from 30B (384/token) to 125B (480/token at 2.7 MB).
    small_batch_gflops=None -> flat: a pass of K+1 positions costs (K+1) decode computes; else the 12 GFLOP per position at that rate."""
    byt = lambda misses: misses * (SLICES125 / SLICES30) * MB125 / 1000
    io = byt(sim["misses"]) / RATE_4MIB / sim["tokens"]
    comp_pass = (K + 1) * (T_COMP if small_batch_gflops is None else 12.0 / small_batch_gflops)
    comp = comp_pass * sim["passes"] / sim["tokens"]
    plain = byt(plain_miss_per_tok) / RATE_4MIB + T_COMP
    return plain / (io + comp), io + comp, plain

def fam_s1_21():
    """Tiny draft / n-gram self-draft, verify K tokens in one pass: extra bytes read (the IO kill condition) and the speedup under flat compute vs the unproven fast-small-batch compute."""
    Ks = (1, 2, 3, 7); As = (0.5, 0.7, 0.9); Fs = (2.0, 4.0, 7.5); rows = {}
    for f in Fs:
        plain = {nm: lru_miss_sets_count(nm, f) for nm in TRACES}
        for K in Ks:
            for a in As:
                br = []; sf = []; sb = []
                for j, nm in enumerate(TRACES):
                    s = spec_cycle_sim(nm, f, K, a, 21000 + 100 * j + K)
                    ratio = (s["misses"] / s["tokens"]) / plain[nm]; br.append(ratio)
                    sf.append(spec_time(plain[nm] * 1.0, s, K, None)[0]); sb.append(spec_time(plain[nm] * 1.0, s, K, PREFILL_GFLOPS)[0])
                rows[f"fp{f}_K{K}_a{a}"] = dict(bytes_ratio=r3(sum(br) / 3), bytes_ratio_min=r3(min(br)), bytes_ratio_max=r3(max(br)), speedup_flat=r3(sum(sf) / 3), speedup_fast_small_batch=r3(sum(sb) / 3))
    k1 = [v["bytes_ratio"] for k, v in rows.items() if "_K1_" in k]; hi = [v["bytes_ratio"] for k, v in rows.items() if "_K1_" not in k]
    return dict(rows=rows, bytes_ratio_k1_range=[min(k1), max(k1)], bytes_ratio_k_ge2_range=[min(hi), max(hi)],
                speedup_flat_max=max(v["speedup_flat"] for v in rows.values()), speedup_flat_min=min(v["speedup_flat"] for v in rows.values()),
                speedup_fast_max=max(v["speedup_fast_small_batch"] for v in rows.values()), speedup_fast_min=min(v["speedup_fast_small_batch"] for v in rows.values()),
                unmeasured="real draft acceptance (a is a scenario value, not measured), the extra experts of REJECTED positions (trace tokens stand in for them), and small-batch compute: see the llama-bench record in the report")

def lru_miss_sets_count(name, foot):
    """Misses per token of plain one-token decode with an LRU of `foot` footprints."""
    _, acc, mis = lru_miss_sets(name, foot); return mis / len(tokens(name))

def bench_ratio(rows):
    """From llama-bench JSON rows (list of dicts with n_prompt, n_gen, avg_ts): per-token cost of batch p relative to decode, {p: cost ratio} (cost ratio = decode_ts / (p_ts / 1) = time per token at batch p over time per token at decode)."""
    dec = [r["avg_ts"] for r in rows if r.get("n_gen", 0) > 0 and r.get("n_prompt", 0) == 0]
    if not dec: return {}
    d = sum(dec) / len(dec); out = {}
    for r in rows:
        if r.get("n_prompt", 0) > 0 and r.get("n_gen", 0) == 0: out[r["n_prompt"]] = r3(d / r["avg_ts"])
    return out

# ---------------------------------------------------------------- screen 2 #0: NPU prefill
def touched_fraction(name, n):
    """Mean over layers and windows of the share of the 128 experts touched by n consecutive tokens of a real trace."""
    M = masks(name); T = len(M); tot = c = 0
    for s in range(0, T - n + 1, max(1, n // 4)):
        for l in range(NL):
            u = 0
            for t in range(s, s + n): u |= M[t][l]
            tot += popcount(u) / E30; c += 1
    return tot / c

def ttft(n_tok, gflops, read_gb, rate, overlap=True):
    comp = n_tok * 12e9 / (gflops * 1e9); io = read_gb / rate
    return (max(comp, io) if overlap else comp + io), comp, io

def fam_s2_0():
    """NPU/FPGA for prefill GEMMs: compute vs I/O time of a 500-token prefill, how many experts a prompt touches, speedup when compute is cut to 900 GFLOP/s, and the prompt length above which it would matter."""
    N = 500; file_gb = 69.0; exp_gb = NL * E125 * MB125 / 1000
    comp = 6e12 / (PREFILL_GFLOPS * 1e9); decode_rate_t = 6e12 / 55e9
    io = {r: file_gb / r for r in (2.4, 3.1, 1.56)}
    cover = {n: dict(real_30b={nm[-5:]: r3(touched_fraction(nm, n)) for nm in TRACES}, random_128=r3(unique_random(E30, K30, n) / E30), random_512=r3(unique_random(E125, K125, n) / E125)) for n in (10, 50, 100, 200)}
    cover[500] = dict(random_128=r3(unique_random(E30, K30, 500) / E30), random_512=r3(unique_random(E125, K125, 500) / E125))
    tt = {}
    for r, t_io in io.items():
        t0 = ttft(N, PREFILL_GFLOPS, file_gb, r); t1 = ttft(N, 900.0, file_gb, r); ts0 = ttft(N, PREFILL_GFLOPS, file_gb, r, False); ts1 = ttft(N, 900.0, file_gb, r, False)
        tt[str(r)] = dict(io_s=r3(t_io), overlap_cpu=r3(t0[0]), overlap_npu=r3(t1[0]), overlap_speedup=r3(t0[0] / t1[0]), serial_cpu=r3(ts0[0]), serial_npu=r3(ts1[0]), serial_speedup=r3(ts0[0] / ts1[0]),
                          io_over_compute=r3(t_io / comp), break_even_prompt_tokens=int(t_io * PREFILL_GFLOPS * 1e9 / 12e9))
    # the file read of a prefill is the share of experts the prompt touches; real single-prompt traces touch far fewer than a random model says, so test coverage scenarios
    cov_sc = {}
    for cvg in (1.0, 0.75, 0.6):
        for r in (2.4, 3.1):
            t0 = ttft(N, PREFILL_GFLOPS, exp_gb * cvg, r); t1 = ttft(N, 900.0, exp_gb * cvg, r)
            cov_sc[f"cov{cvg}_rate{r}"] = dict(io_s=r3(exp_gb * cvg / r), io_over_compute=r3(exp_gb * cvg / r / comp), cpu=r3(t0[0]), npu=r3(t1[0]), speedup=r3(t0[0] / t1[0]))
    long = {}
    for n in (500, 2000, 8000):
        a = ttft(n, PREFILL_GFLOPS, file_gb, 2.4); b = ttft(n, 900.0, file_gb, 2.4); long[n] = dict(cpu=r3(a[0]), npu=r3(b[0]), speedup=r3(a[0] / b[0]))
    return dict(prompt_tokens=N, compute_s_at_311=r3(comp), compute_s_at_55_decode_rate=r3(decode_rate_t), io_s={str(r): r3(v) for r, v in io.items()}, ttft=tt, claim_baseline_s=109.0,
                claim_ratio_109_over_29=r3(109 / 29.0), baseline_error_factor=r3(decode_rate_t / comp), coverage=cover, expert_file_gb=r3(exp_gb), ttft_by_prompt_length_2p4=long, coverage_scenarios=cov_sc,
                unmeasured="NPU/FPGA throughput (900 GFLOP/s is the proposal's figure), prefill GFLOP/s on the 125B (311 is the OLMoE llama-bench figure) and real overlap efficiency")

# ---------------------------------------------------------------- screen 2 #1: computational-storage drives
def drive_balance(name, drives=2, assign="half", E=E30):
    """Per token-layer, bytes on the busiest drive over the mean per drive, when each expert lives wholly on one drive: assign 'half' (id < E/2 -> drive 0), 'mod' (id % drives), 'hash'."""
    f = {"half": lambda e: min(drives - 1, e * drives // E), "mod": lambda e: e % drives, "hash": lambda e: (e * 2654435761 >> 7) % drives}[assign]
    ratios = []; mx = 0; tot = 0
    for tok in tokens(name):
        for ids in tok:
            c = collections.Counter(f(e) for e in ids); m = max(c.values()); ratios.append(m * drives / len(ids)); mx += m; tot += len(ids)
    return sum(ratios) / len(ratios), mx * drives / tot

def random_balance(E, k, drives, n, seed):
    """Same for uniformly random top-k of E experts, hash assignment (expert e on drive e % drives)."""
    rng = random.Random(seed); s = 0
    for _ in range(n):
        ids = rng.sample(range(E), k); c = collections.Counter(e % drives for e in ids); s += max(c.values()) * drives / k
    return s / n

def fam_s2_1():
    """Expert GEMV in computational-storage drives: the arithmetic (one NVMe, two striped NVMe, the 4.57 tok/s compute ceiling), and the per-layer imbalance when each expert lives on one drive."""
    one, two, ceil_c = 1 / (1.3 / RATE_EXPL4), 1 / (1.3 / (2 * RATE_EXPL4)), 55 / 12
    h = {gb: mean_hit(fp) for gb, fp in FOOT.items()}
    best = {f"{gb}GB": dict(hit=r3(h[gb]), serial=r3(tok_s(h[gb], RATE_4MIB)[0]), overlap=r3(tok_s(h[gb], RATE_4MIB)[1])) for gb in FOOT}
    imb = {}
    for assign in ("half", "mod", "hash"):
        v = [drive_balance(nm, 2, assign) for nm in TRACES]; imb[assign] = dict(mean_layer_max_over_mean=r3(sum(x[0] for x in v) / 3), total_max_over_mean=r3(sum(x[1] for x in v) / 3))
    rnd = r3(random_balance(E125, K125, 2, 20000, 22001))
    # per-token time with whole experts per drive, no RAM cache: sum over layers of the busiest drive's bytes / one drive's rate; striped host RAID0 splits each read across drives (balanced).
    per_drive_rate = RATE_EXPL4
    factor = imb["hash"]["mean_layer_max_over_mean"]; layer_imb_125 = rnd
    t_ideal = GB_TOK / (2 * per_drive_rate); t_csd = t_ideal * layer_imb_125
    csd = dict(io_s_ideal=r3(t_ideal), io_s_with_imbalance=r3(t_csd), tok_s_overlap_ideal=r3(1 / max(t_ideal, T_COMP)), tok_s_overlap_imbalanced=r3(1 / max(t_csd, T_COMP)),
               tok_s_serial_imbalanced=r3(1 / (t_csd + T_COMP)))
    pdr = {}
    for r in (RATE_EXPL4, RECORDED["threads_2p7mb"][16], 7.0):
        ti = GB_TOK / (2 * r); pdr[str(r)] = dict(io_s_ideal=r3(ti), tok_s_overlap_ideal=r3(1 / max(ti, T_COMP)), tok_s_overlap_imbalanced=r3(1 / max(ti * layer_imb_125, T_COMP)), tok_s_serial_imbalanced=r3(1 / (ti * layer_imb_125 + T_COMP)))
    serial_gain = r3((1 / max(t_ideal, T_COMP)) / best["8GB"]["serial"]); ovl_gain = r3((1 / max(t_ideal, T_COMP)) / best["8GB"]["overlap"])
    return dict(one_nvme_tok_s=r3(one), two_nvme_tok_s=r3(two), compute_ceiling_tok_s=r3(ceil_c), two_nvme_capped_tok_s=r3(min(two, ceil_c)), best_path=best,
                csd_per_drive_gb_per_token=r3(GB_TOK / 2), csd_per_drive_io_s=r3(GB_TOK / 2 / per_drive_rate), compute_s=r3(T_COMP), imbalance_real_128_top8=imb, imbalance_random_512_top10=rnd, csd=csd,
                csd_ideal_gain_over_best_8gb_serial=serial_gain, csd_ideal_gain_over_best_8gb_overlap=ovl_gain, striping_equals_csd_ideal=bool(abs(two - 1 / t_ideal) / two < 0.01), per_drive_rate_scenarios=pdr, recorded_nvme_threads=RECORDED["threads_2p7mb"],
                unmeasured="per-drive in-device NAND bandwidth of a computational-storage drive (no datasheet number; 7 GB/s per drive would be the survival threshold), FPGA GEMV throughput, dispatch overhead of 480 slice calls; bench_threads measures only this Mac's NVMe")

# ---------------------------------------------------------------- OLMoE (16 layers, 64 experts, top-8) traces recorded with the patched llama.cpp build: optional model-run evidence
def load_trace_file(path, drop_first=True):
    """Parse a GGML_MOE_TRACE file with trace_tools.parse. The first complete block is the batched prompt record and is dropped by default."""
    with open(path) as f: toks, _ = trace_tools.parse(f.read().splitlines())
    toks = [[list(t[l]) for l in sorted(t)] for t in toks]
    return toks[1:] if drop_first else toks

def union_factor_files(streams, B_list=(2, 4)):
    """Mean over all B-subsets of `streams` (list of token lists), over aligned tokens and layers, of distinct experts / (k*B)."""
    import itertools
    out = {}
    for B in B_list:
        vals = []
        for combo in itertools.combinations(range(len(streams)), B):
            T = min(len(streams[i]) for i in combo); tot = n = 0
            for t in range(T):
                for l in range(len(streams[combo[0]][t])):
                    u = set()
                    for i in combo: u |= set(streams[i][t][l])
                    tot += len(u) / (len(streams[combo[0]][t][l]) * B); n += 1
            vals.append(tot / n)
        out[B] = sum(vals) / len(vals)
    return out

def olmoe_study(paths_same, paths_diff, paths_all):
    """OLMoE evidence for #17 (union factor, same-domain vs different-domain prompts, B=2 and 4) and #20 (held-out co-activation layout, leave-one-prompt-out)."""
    same = [load_trace_file(p) for p in paths_same]; diff = [load_trace_file(p) for p in paths_diff]; allt = [load_trace_file(p) for p in paths_all]
    E = 64; layers = len(allt[0][0])
    reads_id = []; reads_ho = []
    for i, test in enumerate(allt):
        train = [x for j, x in enumerate(allt) if j != i]; pos = []
        for l in range(layers):
            order = coact_chain(train, l, E); pos.append(order)
        reads_id.append(layout_reads(test, None)[0]); reads_ho.append(layout_reads(test, pos)[0])
    mean = lambda v: sum(v) / len(v)
    return dict(union_factor_same_domain={B: r3(v) for B, v in union_factor_files(same).items()}, union_factor_different_domain={B: r3(v) for B, v in union_factor_files(diff).items()},
                random_64={B: r3(unique_random(64, 8, B) / (8 * B)) for B in (2, 4)}, prompts=len(allt), tokens_per_prompt=len(allt[0]), layers=layers,
                reads_identity=r3(mean(reads_id)), reads_held_out=r3(mean(reads_ho)), run_length_identity=r3(8 / mean(reads_id)), run_length_held_out=r3(8 / mean(reads_ho)))

FAMILIES = {
    "s1_16": dict(idea="Serve K independent prompts in lock-step so one read of each expert serves all K sequences", run=fam_s1_16, needs="local",
                  methods=["[screen 1 #16] published method: SSD Offloading for LLM MoE Weights Considered Harmful in Energy Efficiency", "[screen 1 #16] published method: MoE-Lightning (batch offloading, CPU-GPU-IO pipeline)",
                           "[screen 1 #16] published method: MoE-Gen module-based batching", "[screen 1 #16] published method: ExpertFlow (token reordering across batches to reduce expert loads)",
                           "[screen 1 #16] published method: Dynamic Expert Sharing (diffusion LLM)"]),
    "s1_17": dict(idea="Process B tokens/prompts together layer by layer so each expert read serves the batch (large aligned reads)", run=fam_s1_17, needs="local",
                  methods=["[screen 1 #17] published method: MoE-Lightning", "[screen 1 #17] published method: Klotski", "[screen 1 #17] published method: Mixtral offloading"]),
    "s1_18": dict(idea="Replace mmap demand paging by explicit per-layer batched pread (no cross-token cache)", run=fam_s1_18, needs="local",
                  methods=["[screen 1 #18] published method: llama.cpp issue #19825 (and a r/LocalLLaMA --moe-direct-io post)", "[screen 1 #18] published method: Edge0 (arXiv 2609.18063)"]),
    "s1_19": dict(idea="Repack gate/up/down of an expert into one contiguous run and cluster co-firing experts", run=fam_s1_19, needs="local",
                  methods=["[screen 1 #19] published method: Paging the Experts: flash-backed MoE inference on iPhone (arXiv 2609.29032)", "[screen 1 #19] published method: BuddyMoE (arXiv 2511.10054)",
                           "[screen 1 #19] published method: Janas-LLM SSD streaming engine", "[screen 1 #19] published method: BigMoeOnEdge"]),
    "s1_20": dict(idea="Rewrite the GGUF so experts that co-fire are stored adjacent on disk", run=fam_s1_20, needs="local",
                  methods=["[screen 1 #20] published method: mbolt - profile-guided co-activation layout optimization for GGUF MoE with explicit-read prefetch",
                           "[screen 1 #20] published method: LLM in a flash (Apple), row-column bundling and co-activation bundling", "[screen 1 #20] published method: MoE-Infinity"]),
    "s1_21": dict(idea="Draft K tokens (tiny model or n-gram) and verify them in one batched pass", run=fam_s1_21, needs="local",
                  methods=["[screen 1 #21] published method: MoE-SpeQ"]),
    "s2_0": dict(idea="NPU/FPGA co-processor for prefill GEMMs overlapped with next-layer expert reads", run=fam_s2_0, needs="local",
                 methods=["[screen 2 #0] published method: Efficient Mixture-of-Experts LLM Inference with Apple Silicon NPUs", "[screen 2 #0] published method: ProMoE", "[screen 2 #0] published method: DuoServe-MoE",
                          "[screen 2 #0] published method: ssd-llm", "[screen 2 #0] published method: Deltafin fork, Kimi K3 on a MacBook (blog, unverified)", "[screen 2 #0] published method: FluxMoE"]),
    "s2_1": dict(idea="Expert GEMV engines inside computational-storage FPGA drives so weights never cross PCIe", run=fam_s2_1, needs="local",
                 methods=["[screen 2 #1] published method: InstInfer: in-storage attention offloading to computational storage drives", "[screen 2 #1] published method: HillInfer: long-context edge LLM inference with SmartSSD",
                          "[screen 2 #1] published method: Conduit: near-data processing across SSD compute resources", "[screen 2 #1] published method: Samsung second-generation SmartSSD (Versal Prime VM1802)",
                          "[screen 2 #1] published method: DockerSSD in-storage processing"]),
}

def _main(argv):
    if not argv: print(__doc__); return 0
    if argv[0] == "bench-pread": print(json.dumps(bench_pread(argv[1], int(argv[2]) if len(argv) > 2 else 12, int(argv[3]) if len(argv) > 3 else 8), indent=1)); return 0
    if argv[0] == "bench-layout": print(json.dumps(bench_layout(argv[1], int(argv[2]) if len(argv) > 2 else 300), indent=1)); return 0
    if argv[0] == "bench-threads": print(json.dumps(bench_threads(argv[1], float(argv[2]) if len(argv) > 2 else 2.7), indent=1)); return 0
    if argv[0] == "olmoe-study":
        d = argv[1]; P = lambda ids: [os.path.join(d, f"ol{i}.txt") for i in ids]; print(json.dumps(olmoe_study(P([1, 2, 3, 4]), P([1, 5, 6, 7]), P(range(1, 9))), indent=1)); return 0
    for a in argv:
        key = a.replace("fam_", ""); print(json.dumps({key: FAMILIES[key]["run"]()}, indent=1, default=str))
    return 0

if __name__ == "__main__": sys.exit(_main(sys.argv[1:]))
