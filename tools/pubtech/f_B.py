#!/usr/bin/env python3
"""Published-technique family tests, group B: screen 1 ideas 8..15 (pinned set + routing bias, expert-major prefill, pin after prefill, prefill top-4,
GPU tier, top-4 for TTFT, repacked expert extents, no-cache slab reads).

Every number is computed from the three recorded router traces (Qwen3-30B-A3B: 128 experts/layer, top-8, 48 layers, data/traces/*.txt), from closed-form
throughput arithmetic, or (optional, PUBTECH_IO=1) from a read-only I/O micro-benchmark of a local GGUF file. Mapping to the 125B target is by token-footprints:
cache entries = footprints x 384 accesses per token (top-8); 8/16/32 GB machines have 2.5/9.15/25.15 GiB of expert cache = 2.1/7.6/20.8 footprints of the 125B.
Caveats that apply to every family: 30B decode traces stand in for the 125B (128 vs 512 experts per layer), traces are 249-299 tokens (cold start counted),
decode routing is not prefill routing, and no 125B-class model or perplexity run exists here.
Tags: [M] measured here, [C] computed here, [E] estimate/assumption carried from the screen, [A] assumption.
usage: python3 tools/pubtech/f_B.py [family key]   (PUBTECH_IO=1 adds the I/O micro-benchmark; PUBTECH_GGUF=<path> overrides the default model path)
"""
import collections, json, math, mmap, os, struct, sys, threading, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import trace_tools as tt

NAMES = ["qwen3-30b_db", "qwen3-30b_story", "qwen3-30b_code"]
NL, NE, K = 48, 128, 8
FOOT = NL * K                      # 384 accesses of one token = one footprint on the 30B
PAIRS = NL * NE                    # 6144 (layer, expert) pairs
EXP_GB = 1.3                       # [E] expert bytes per token of the 125B at top-10, 4.3 bpw, hit 0 (480 slices x 2.7 MB)
COMPUTE = 4.57                     # [E] decode compute ceiling tok/s (55 GFLOP/s over 12 GFLOP per token)
R_MMAP, R_1M, R_4M, R_09, R_27, R_4T = 0.34, 1.56, 2.4, 1.3, 2.2, 3.11   # [E] GB/s: mmap paging; 1 MiB and 4 MiB reads; 0.9 MB reads; 2.7 MB extent; 2.7 MB x 4 threads
MODEL_GB = 69.0                    # [E] expert bytes of the 125B
PREFILL_GFLOPS, FLOP_TOK, PROMPT = 311.0, 12.0, 500   # [E] measured prefill rate on OLMoE; GFLOP per token; prompt length
DENSE_FRAC = 3.4 / 6.0             # [E] dense share of active parameters
G_DECODE = 55.0
CACHE_GIB = {8: 2.5, 16: 9.15, 32: 25.15}

# ------------------------------------------------------------------ data and helpers
_CACHE = {}
def load(name):
    if name not in _CACHE:
        with open(os.path.join(HERE, "..", "..", "data", "traces", name + ".txt")) as f: _CACHE[name] = tt.parse(f)[0]
    return _CACHE[name]

def flat(tokens, k=None):
    """Flat (layer, expert) access sequence; k keeps the first k ranked ids of each token-layer (k=None: all)."""
    return [(l, e) for tk in tokens for l in range(NL) for e in (tk[l] if k is None else tk[l][:k])]

def fp_of_gib(g): return g * 1.074e9 / 2.7e6 / 480.0

def lru_scored(acc, cap, start=0):
    """Hit rate of an LRU of `cap` entries over acc[start:], the cache having seen acc[:start] (start=0: cold start counted)."""
    if cap <= 0: return 0.0
    c = collections.OrderedDict(); h = 0
    for i, a in enumerate(acc):
        if a in c:
            c.move_to_end(a)
            if i >= start: h += 1
        else:
            c[a] = None
            if len(c) > cap: c.popitem(last=False)
    n = len(acc) - start
    return h / n if n > 0 else 0.0

def thr(hit, rate, compute=COMPUTE, exp_gb=EXP_GB):
    """(serial, overlap) tokens/s: bytes = (1-hit) x exp_gb at `rate` GB/s; serial 1/(io+1/compute), overlap min(compute, rate/bytes)."""
    b = (1.0 - hit) * exp_gb
    return 1.0 / (b / rate + 1.0 / compute), (compute if b <= 0 else min(compute, rate / b))

def mean(v):
    v = list(v); return sum(v) / len(v) if v else 0.0

def top_pairs(cnt, n):
    return [k for k, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:max(0, n)]]

_PINS = {}
def cross_pins(name, n):
    """Pin set of the n most used pairs in the OTHER traces (held out), the way a shipped pin list would be learned."""
    key = (name, n)
    if key not in _PINS:
        cnt = collections.Counter()
        for o in NAMES:
            if o != name: cnt.update(flat(load(o)))
        _PINS[key] = frozenset(top_pairs(cnt, n))
    return _PINS[key]

def window_union(tokens, W, k, stride=None):
    """Mean over layers and windows of the number of distinct experts in W consecutive tokens (first k ranked ids per token-layer)."""
    stride = stride or max(1, W // 4); tot = n = 0
    for s in range(0, len(tokens) - W + 1, stride):
        for l in range(NL):
            u = set()
            for tk in tokens[s:s + W]: u.update(tk[l][:k])
            tot += len(u); n += 1
    return tot / n if n else 0.0

def prefix_union(tokens, W, k):
    """Mean over layers of distinct experts in the FIRST W tokens (the prompt-start window)."""
    return mean(len(set(e for tk in tokens[:W] for e in tk[l][:k])) for l in range(NL))

def marginal_coverage(tokens, n_tok, k=None):
    """Independent-draw model: per layer, expert e is in a token with probability p_e = count/T; coverage after n_tok tokens = mean_e 1-(1-p_e)^n_tok."""
    T = len(tokens); tot = 0.0
    for l in range(NL):
        cnt = collections.Counter(e for tk in tokens for e in tk[l][:k])
        tot += sum(1.0 - (1.0 - c / T) ** n_tok for c in cnt.values()) / NE
    return tot / NL

def decode_misses(name, k, cap_entries):
    """LRU misses per token with the first k ranked experts, at a fixed cache of cap_entries (cold start counted)."""
    tk = load(name); acc = flat(tk, k)
    return (1.0 - lru_scored(acc, cap_entries)) * len(acc) / len(tk)

def lru_hit_fp(fp, k=None):
    return mean(lru_scored(flat(load(n), k), int(fp * FOOT)) for n in NAMES)

# ------------------------------------------------------------------ GGUF header parsing (stdlib) and I/O micro-benchmarks
_GT = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
BLK = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34), 10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210)}
def _rd_str(b, o):
    n = struct.unpack_from("<Q", b, o)[0]; return b[o + 8:o + 8 + n].decode("utf-8", "replace"), o + 8 + n
def _skip(b, o, t):
    if t in _GT: return o + _GT[t]
    if t == 8: return o + 8 + struct.unpack_from("<Q", b, o)[0]
    if t == 9:
        et, n = struct.unpack_from("<IQ", b, o); o += 12
        if et in _GT: return o + _GT[et] * n
        for _ in range(n): o = _skip(b, o, et)
        return o
    raise ValueError("bad gguf value type %d" % t)
def gguf_tensor_infos(b):
    """-> (list of (name, dims, type, offset), data_start, alignment) from the bytes of a GGUF v2/v3 header."""
    if b[:4] != b"GGUF": raise ValueError("not a gguf file")
    ver, nt, nkv = struct.unpack_from("<IQQ", b, 4); o = 24; align = 32
    for _ in range(nkv):
        key, o = _rd_str(b, o); t = struct.unpack_from("<I", b, o)[0]; o += 4
        if key == "general.alignment" and t == 4: align = struct.unpack_from("<I", b, o)[0]
        o = _skip(b, o, t)
    infos = []
    for _ in range(nt):
        name, o = _rd_str(b, o); nd = struct.unpack_from("<I", b, o)[0]; o += 4
        dims = struct.unpack_from("<%dQ" % nd, b, o); o += 8 * nd
        ty, off = struct.unpack_from("<IQ", b, o); o += 12; infos.append((name, tuple(dims), ty, off))
    return infos, (o + align - 1) // align * align, align
def tensor_bytes(dims, ty):
    v, by = BLK[ty]; n = 1
    for d in dims: n *= d
    return n // v * by
def gguf_path():
    return os.environ.get("PUBTECH_GGUF") or os.path.join(os.path.expanduser("~"), "Downloads", "Qwen3-30B-A3B-Q4_K_M.gguf")
def read_header(path):
    with open(path, "rb") as f:
        for mb in (8, 32, 128):
            f.seek(0); b = f.read(mb << 20)
            try: return gguf_tensor_infos(b)
            except (struct.error, IndexError): continue
    raise ValueError("gguf header too large")
def expert_layout(infos, data_start, nl=NL):
    """-> {layer: {'gate'|'up'|'down': (abs_offset, slice_bytes)}} for the stacked per-layer expert tensors."""
    t = {n: (d, ty, off) for n, d, ty, off in infos}; out = {}; ne = None
    for l in range(nl):
        row = {}
        for w in ("gate", "up", "down"):
            nm = "blk.%d.ffn_%s_exps.weight" % (l, w)
            if nm not in t: continue
            d, ty, off = t[nm]; ne = d[-1]; row[w] = (data_start + off, tensor_bytes(d, ty) // d[-1])
        if row: out[l] = row
    return out, ne
def align_up(x, a): return (x + a - 1) // a * a
def page_amplification(off, size, page=16384):
    """Bytes of whole pages touched by [off, off+size) divided by size."""
    return (align_up(off + size, page) - off // page * page) / size

def layout_report(path):
    infos, ds, align = read_header(path); lay, ne = expert_layout(infos, ds)
    rows = []; al = tot = 0; amp = []
    for l, row in lay.items():
        for w, (off, sl) in row.items():
            for e in range(ne):
                o = off + e * sl; tot += 1; al += (o % 4096 == 0); amp.append(page_amplification(o, sl))
    sl_all = [sum(r[w][1] for w in r) for r in lay.values()]
    l0 = lay[0]; gap = l0["up"][0] - l0["gate"][0]
    return dict(layers=len(lay), experts=ne, alignment=align, slice_bytes_gate=l0["gate"][1], slice_bytes_up=l0["up"][1], slice_bytes_down=l0["down"][1],
                expert_bytes_mean=mean(sl_all), share_slices_4k_aligned=al / tot, mean_16k_page_amplification=mean(amp), gate_to_up_distance_bytes=gap,
                reads_per_expert_stock=3, reads_per_expert_repacked=1)

_IO = {}
def _buf_ptr(m):
    import ctypes
    return ctypes.addressof(ctypes.c_char.from_buffer(m))
def _mapping(path):
    if "map" not in _IO:
        fd = os.open(path, os.O_RDONLY); m = mmap.mmap(fd, 0, access=mmap.ACCESS_COPY); _IO["map"] = (fd, m, _buf_ptr(m))
    return _IO["map"]
def _nocache_fd(path):
    fd = os.open(path, os.O_RDONLY)
    if sys.platform == "darwin":
        import fcntl; fcntl.fcntl(fd, 48, 1)
    return fd
def _run_threads(n, fn):
    errs = []
    def w(i):
        try: fn(i)
        except Exception as ex: errs.append(repr(ex))
    ts = [threading.Thread(target=w, args=(i,)) for i in range(n)]; [t.start() for t in ts]; [t.join() for t in ts]
    return errs
def pread_rate(path, size, threads, secs, seed=1):
    import random
    fsz = os.path.getsize(path); fd = _nocache_fd(path); tot = [0] * threads; stop = time.time() + secs
    def w(i):
        rng = random.Random(seed * 100 + i); buf = mmap.mmap(-1, align_up(size, 4096))
        while time.time() < stop: tot[i] += os.preadv(fd, [memoryview(buf)[:size]], rng.randrange(0, (fsz - size) // 4096) * 4096)
    t0 = time.time(); errs = _run_threads(threads, w); dt = time.time() - t0; os.close(fd)
    return sum(tot) / dt / 1e9
def mmap_rate(path, size, threads, secs, seed=2):
    import random, ctypes
    fsz = os.path.getsize(path); _, _, base = _mapping(path); tot = [0] * threads; stop = time.time() + secs
    def w(i):
        rng = random.Random(seed * 100 + i); buf = ctypes.create_string_buffer(size)
        while time.time() < stop:
            ctypes.memmove(buf, base + rng.randrange(0, (fsz - size) // 4096) * 4096, size); tot[i] += size
    t0 = time.time(); _run_threads(threads, w); dt = time.time() - t0
    return sum(tot) / dt / 1e9

def replay(path, lay, name, policy, variant, tokens_n=3, warm=20, threads=8):
    """Read the experts one policy would fetch for tokens_n decode tokens of a trace. variant: split3 (3 reads per expert, real tensor offsets),
    coalesced (one read of the expert's summed size at its gate offset: what a repacked file would allow), mmap (split3 extents copied from a mapping)."""
    import ctypes
    T = load(name); fsz = os.path.getsize(path); sel = []; cache = collections.OrderedDict(); cap = 2 * FOOT; prev = set()
    for t in range(warm + tokens_n):
        keys = [(l, e) for l in range(NL) for e in T[t][l]]; miss = []
        for k in keys:
            if policy == "none": miss.append(k)
            elif policy == "prev":
                if k not in prev: miss.append(k)
            else:
                if k in cache: cache.move_to_end(k)
                else:
                    miss.append(k); cache[k] = None
                    if len(cache) > cap: cache.popitem(last=False)
        prev = set(keys)
        if t >= warm: sel.append(miss)
    tasks = []
    for miss in sel:
        for (l, e) in miss:
            r = lay[l]
            if variant == "coalesced":
                o = r["gate"][0] + e * r["gate"][1]; n = min(sum(r[w][1] for w in r), fsz - o); tasks.append((o, n))
            else:
                for w in ("gate", "up", "down"): tasks.append((r[w][0] + e * r[w][1], r[w][1]))
    nbytes = sum(n for _, n in tasks); big = align_up(max(n for _, n in tasks), 4096); nxt = [0]; lk = threading.Lock()
    if variant == "mmap": _, _, base = _mapping(path)
    fd = None if variant == "mmap" else _nocache_fd(path)
    def w(i):
        buf = mmap.mmap(-1, big) if variant != "mmap" else ctypes.create_string_buffer(big)
        while True:
            with lk:
                j = nxt[0]; nxt[0] += 1
            if j >= len(tasks): return
            o, n = tasks[j]
            if variant == "mmap": ctypes.memmove(buf, base + o, n)
            else: os.preadv(fd, [memoryview(buf)[:n]], o)
    t0 = time.time(); errs = _run_threads(threads, w); dt = time.time() - t0
    if fd is not None: os.close(fd)
    return dict(policy=policy, variant=variant, gb=nbytes / 1e9, seconds=dt, gb_s=nbytes / dt / 1e9, sec_per_token=dt / tokens_n, reads=len(tasks), errors=len(errs))

def io_results():
    """Optional (PUBTECH_IO=1) read-only micro-benchmark of the local GGUF. Timings are not deterministic; the dict is cached per process."""
    if "res" in _IO: return _IO["res"]
    path = gguf_path(); res = None
    if os.environ.get("PUBTECH_IO") == "1" and os.path.exists(path):
        secs = float(os.environ.get("PUBTECH_IO_SECS", "1.0")); infos, ds, _ = read_header(path); lay, _ = expert_layout(infos, ds)
        micro = []
        for mb in (0.9, 2.7, 4.0):
            sz = int(mb * 1048576) // 4096 * 4096
            for th in (1, 4, 8): micro.append(dict(extent_mb=mb, threads=th, method="pread nocache", gb_s=round(pread_rate(path, sz, th, secs), 3)))
            for th in (1, 8): micro.append(dict(extent_mb=mb, threads=th, method="mmap copy", gb_s=round(mmap_rate(path, sz, th, secs), 3)))
        rep = [replay(path, lay, "qwen3-30b_db", p, v) for p, v in (("none", "split3"), ("none", "coalesced"), ("prev", "coalesced"), ("lru2", "coalesced"), ("none", "mmap"))]
        for r in rep: r.update({k: round(v, 3) for k, v in r.items() if isinstance(v, float)})
        res = dict(file=os.path.basename(path), file_gb=round(os.path.getsize(path) / 1e9, 2), micro=micro, replay=rep)
    _IO["res"] = res; return res
def io_rate(res, method, mb, threads):
    if not res: return None
    for r in res["micro"]:
        if r["method"] == method and r["extent_mb"] == mb and r["threads"] == threads: return r["gb_s"]
    return None

# ------------------------------------------------------------------ family s1 #8: pinned set + router-logit bias
def pin_hit(name, pins):
    acc = flat(load(name)); return sum(a in pins for a in acc) / len(acc)

def swap_hit(name, pins, m=None, target=None, capped=True):
    """Bias model on the trace: per token-layer the lowest-ranked non-pinned chosen experts are swapped for pinned ones. m: at most m swaps; target: raise the pinned
    count to target; capped=False assumes a substitute always exists (the screen's optimistic model). Returns (hit, swapped share of selections)."""
    pl = collections.Counter(l for l, _ in pins); T = load(name); hit = sw = n = 0
    for tk in T:
        for l in range(NL):
            nat = sum((l, e) in pins for e in tk[l]); miss = K - nat
            s = miss if target is None and m is None else (min(m, miss) if target is None else max(0, min(miss, target - nat)))
            if capped: s = max(0, min(s, pl[l] - nat))
            hit += nat + s; sw += s; n += K
    return hit / n, sw / n

def lru_bias(name, cap, m):
    """LRU cache of cap entries with a router bias that swaps up to m of the lowest-ranked missing experts per token-layer for cached, unchosen experts of that layer
    (most recently used first). Returns (hit, swapped share). The substitutes' quality is not modelled."""
    G = collections.OrderedDict(); Ly = [collections.OrderedDict() for _ in range(NL)]; hit = tot = sw = 0
    for tk in load(name):
        for l in range(NL):
            sel = list(tk[l])
            if m > 0 and Ly[l]:
                have = set(sel); subs = [e for e in reversed(Ly[l]) if e not in have]; ns = 0
                for i in reversed([i for i, e in enumerate(sel) if (l, e) not in G]):
                    if ns >= m or not subs: break
                    sel[i] = subs.pop(0); ns += 1; sw += 1
            for e in sel:
                k = (l, e); tot += 1
                if k in G: G.move_to_end(k); Ly[l].move_to_end(e); hit += 1
                else:
                    G[k] = None; Ly[l][e] = None
                    while len(G) > cap:
                        (kl, ke), _ = G.popitem(last=False); del Ly[kl][ke]
    return hit / tot, sw / tot

def fam_s1_8():
    fr = (0.04, 0.07, 0.131, 0.17, 0.245, 0.40); out = {}
    pin = {}
    for f in fr:
        n = int(f * PAIRS); per = {nm: pin_hit(nm, cross_pins(nm, n)) for nm in NAMES}
        pin[f] = dict(entries=n, hit=mean(per.values()), per_trace={k[-5:]: round(v, 4) for k, v in per.items()})
    out["cross_prompt_pin_hit"] = {str(f): pin[f] for f in fr}
    bias = {}
    for f in (0.07, 0.17, 0.245, 0.40):
        n = int(f * PAIRS); row = {}
        for tag, kw in (("opt_m2", dict(m=2, capped=False)), ("opt_m4", dict(m=4, capped=False)), ("cap_m2", dict(m=2)), ("cap_m4", dict(m=4)), ("cap_half", dict(target=4))):
            r = [swap_hit(nm, cross_pins(nm, n), **kw) for nm in NAMES]; row[tag] = dict(hit=mean(x[0] for x in r), swapped=mean(x[1] for x in r))
        bias[str(f)] = row
    out["pin_plus_bias"] = bias
    # first-16-token pin (the screen's step-2 recipe applied to the traces): top 24.5% of pairs among the first 16 tokens, scored on the rest
    n = int(0.245 * PAIRS); r16 = []
    for nm in NAMES:
        T = load(nm); P = set(top_pairs(collections.Counter(flat(T[:16])), n)); rest = flat(T[16:]); r16.append(sum(a in P for a in rest) / len(rest))
    out["first16_pin_hit_24p5"] = mean(r16)
    cap8 = int(fp_of_gib(CACHE_GIB[8]) * FOOT)
    lru0 = [lru_bias(nm, cap8, 0) for nm in NAMES]; out["lru_8gb"] = dict(entries=cap8, hit=mean(x[0] for x in lru0))
    for m in (1, 2, 4):
        r = [lru_bias(nm, cap8, m) for nm in NAMES]; out["lru_bias_m%d_8gb" % m] = dict(hit=mean(x[0] for x in r), swapped=mean(x[1] for x in r))
    out["static_pin_8gb"] = dict(share=cap8 / PAIRS, hit=mean(pin_hit(nm, cross_pins(nm, cap8)) for nm in NAMES))
    rows = {}
    for tag, h in (("screen_lru_0.56", 0.56), ("screen_bias_0.78", 0.78), ("measured_lru_8gb", out["lru_8gb"]["hit"]), ("measured_lru_bias_m2_8gb", out["lru_bias_m2_8gb"]["hit"]),
                   ("measured_static_pin_8gb", out["static_pin_8gb"]["hit"]), ("pin24.5_cap_half", bias["0.245"]["cap_half"]["hit"])):
        s, o = thr(h, R_4M); rows[tag] = dict(hit=h, serial=s, overlap=o)
    out["throughput_explicit_4mib"] = rows
    b = rows["measured_lru_8gb"]; w = rows["measured_lru_bias_m2_8gb"]
    out["gain_vs_lru_8gb"] = dict(serial_bias_m2=w["serial"] / b["serial"], overlap_bias_m2=w["overlap"] / b["overlap"], serial_static_pin=rows["measured_static_pin_8gb"]["serial"] / b["serial"])
    out["gain_vs_mmap_8gb"] = dict(serial_bias_m2=w["serial"] / thr(out["lru_8gb"]["hit"], R_MMAP)[0])
    out["unmeasured"] = ["perplexity/quality of any routing bias (needs a model run with a logit patch)", "router scores (not in the trace): substitute quality", "125B (512 experts) pin hit rates"]
    return out

# ------------------------------------------------------------------ family s1 #9: expert-major prefill
def ttft(cov, rate, overlap, gb=MODEL_GB, tokens=PROMPT):
    io = gb * cov / rate; comp = tokens * FLOP_TOK / PREFILL_GFLOPS
    return max(io, comp) if overlap else io + comp

def fam_s1_9():
    out = {}; uniq = {}
    for nm in NAMES:
        T = load(nm); row = {}
        for N in (16, 32, 64, 128, len(T)):
            row[str(N)] = mean(len(set(e for tk in T[:N] for e in tk[l])) for l in range(NL))
        uniq[nm[-5:]] = row
    out["unique_experts_first_N_tokens_of_128"] = uniq
    out["coverage_all_tokens"] = {nm[-5:]: uniq[nm[-5:]][str(len(load(nm)))] / NE for nm in NAMES}
    model = {}
    for nm in NAMES:
        T = load(nm); model[nm[-5:]] = dict(model_at_64=marginal_coverage(T, 64), empirical_at_64=mean(window_union(T, 64, K, 8) for _ in [0]) / NE,
                                            model_at_500=marginal_coverage(T, 500), model_at_all=marginal_coverage(T, len(T)), uniform_500_top8_of_128=1 - (1 - 8 / 128) ** 500)
    out["marginal_model"] = model
    cov_proxy = mean(m["model_at_500"] for m in model.values()); cov_lo = mean(out["coverage_all_tokens"].values())
    out["coverage_proxy_500"] = dict(measured_all_tokens=cov_lo, model_500=cov_proxy, uniform_125b_top10_of_512=1 - (1 - 10 / 512) ** 500)
    rows = {}
    for tag, cov in (("full_1.0", 1.0), ("proxy_measured", cov_lo), ("proxy_model500", cov_proxy)):
        r = {}
        for rt, rate in (("mmap_0.34", R_MMAP), ("mmap_1.0", 1.0), ("explicit_2.4", R_4M)):
            r[rt] = dict(serial_s=ttft(cov, rate, False), overlap_s=ttft(cov, rate, True))
        r["gain_vs_mmap_0.34"] = dict(serial=r["mmap_0.34"]["serial_s"] / r["explicit_2.4"]["serial_s"], overlap=r["mmap_0.34"]["serial_s"] / r["explicit_2.4"]["overlap_s"])
        r["gain_vs_mmap_1.0"] = dict(serial=r["mmap_1.0"]["serial_s"] / r["explicit_2.4"]["serial_s"], overlap=r["mmap_1.0"]["serial_s"] / r["explicit_2.4"]["overlap_s"])
        rows[tag] = r
    out["ttft_500_tokens_125b"] = rows
    out["compute_floor_s"] = PROMPT * FLOP_TOK / PREFILL_GFLOPS
    out["compute_floor_if_55gflops_s"] = PROMPT * FLOP_TOK / G_DECODE
    out["per_token_loading_strawman"] = dict(gb=PROMPT * EXP_GB, seconds_at_2p4=PROMPT * EXP_GB / R_4M)
    out["breakeven_prompt_tokens_compute_bound"] = MODEL_GB / R_4M * PREFILL_GFLOPS / FLOP_TOK
    io = io_results()
    if io:
        mm, pr = io_rate(io, "mmap copy", 4.0, 8), io_rate(io, "pread nocache", 4.0, 8)
        out["measured_io_rates_gb_s"] = dict(mmap_copy_4mib_8thr=mm, pread_4mib_8thr=pr, note="random 4 MiB extents of a local model file, Python threads; mmap copy is not the stock sequential prefill pattern")
        if mm and pr:
            out["ttft_with_measured_rates"] = dict(mmap_serial_s=ttft(1.0, mm, False), explicit_overlap_s=ttft(1.0, pr, True), gain=ttft(1.0, mm, False) / ttft(1.0, pr, True))
    else: out["measured_io_rates_gb_s"] = None
    out["unmeasured"] = ["real prefill routing/coverage (traces are decode-only; the trace hook skips prefill)", "stock llama.cpp mmap read rate in a real prefill", "whether stock ubatch=512 prefill is already expert-major (source claim, not measured here)"]
    return out

# ------------------------------------------------------------------ family s1 #10: pin after prefill
def hybrid_hit(acc, start, pins, cap_lru):
    """Static pins plus an LRU of cap_lru entries for everything else; hit rate over acc[start:]."""
    c = collections.OrderedDict(); h = 0
    for i, a in enumerate(acc):
        if a in pins:
            if i >= start: h += 1
            continue
        if a in c:
            c.move_to_end(a)
            if i >= start: h += 1
        else:
            c[a] = None
            if len(c) > cap_lru: c.popitem(last=False)
    return h / (len(acc) - start)

def fam_s1_10():
    out = {}; shares = (0.06, 0.10, 0.20, 0.36)
    for W in (16, 64):
        res = {}
        for sh in shares:
            n = int(sh * PAIRS); per = {}
            for nm in NAMES:
                T = load(nm); acc = flat(T); st = W * FOOT; learn = flat(T[:W]); P = set(top_pairs(collections.Counter(learn), n)); rest = acc[st:]
                pin_h = sum(a in P for a in rest) / len(rest); lru_h = lru_scored(acc, n, st)
                P2 = set(top_pairs(collections.Counter(learn), n // 2)); hyb = hybrid_hit(acc, st, P2, n - n // 2)
                oracle = set(top_pairs(collections.Counter(rest), n)); orc = sum(a in oracle for a in rest) / len(rest)
                per[nm[-5:]] = dict(pin=pin_h, lru=lru_h, hybrid=hyb, oracle_in_sample_pin=orc)
            res[str(sh)] = dict(entries=int(sh * PAIRS), footprints=int(sh * PAIRS) / FOOT, per_trace=per,
                                mean_pin=mean(v["pin"] for v in per.values()), mean_lru=mean(v["lru"] for v in per.values()), mean_hybrid=mean(v["hybrid"] for v in per.values()),
                                mean_oracle=mean(v["oracle_in_sample_pin"] for v in per.values()))
        out["W%d" % W] = res
    n36 = int(0.36 * PAIRS)
    out["cross_prompt_pin_36"] = mean(pin_hit(nm, cross_pins(nm, n36)) for nm in NAMES)
    out["claim_check_40pct"] = dict(hit_of_10pct_pin_W64=out["W64"]["0.1"]["mean_pin"], hit_of_36pct_pin_W64=out["W64"]["0.36"]["mean_pin"])
    tp = {}
    for tag, h in (("claimed_h0.40", 0.40), ("pin_by_share_h0.26", 0.26), ("lru_8gb_h0.56", 0.56)): tp[tag] = dict(zip(("serial", "overlap"), thr(h, R_4M)))
    tp["pin36_32gb"] = dict(zip(("serial", "overlap"), thr(out["W64"]["0.36"]["mean_pin"], R_4M))); tp["lru36_32gb"] = dict(zip(("serial", "overlap"), thr(out["W64"]["0.36"]["mean_lru"], R_4M)))
    out["throughput_explicit_4mib"] = tp
    out["gain_pin_vs_lru_8gb"] = dict(serial=tp["pin_by_share_h0.26"]["serial"] / tp["lru_8gb_h0.56"]["serial"], overlap=tp["pin_by_share_h0.26"]["overlap"] / tp["lru_8gb_h0.56"]["overlap"])
    out["unmeasured"] = ["real prefill routing as the pin predictor (the first decode tokens stand in for it)", "125B (512 experts) share mapping", "long-run drift beyond ~300 tokens"]
    return out

# ------------------------------------------------------------------ family s1 #11: prefill top-4
def fam_s1_11():
    out = {}; wu = {}
    for nm in NAMES:
        T = load(nm); row = {}
        for W in (16, 64, 128, len(T)):
            a, b = window_union(T, W, 8), window_union(T, W, 4)
            if W == len(T): W = "all(%d)" % len(T)
            row[str(W)] = dict(k8=a, k4=b, ratio=a / b)
        wu[nm[-5:]] = row
    out["window_union_unique_experts_per_layer"] = wu
    out["first_window_union"] = {nm[-5:]: {str(W): dict(k8=prefix_union(load(nm), W, 8), k4=prefix_union(load(nm), W, 4)) for W in (16, 64)} for nm in NAMES}
    cov = {}
    for nm in NAMES:
        T = load(nm); cov[nm[-5:]] = dict(k8_500=marginal_coverage(T, 500, 8), k4_500=marginal_coverage(T, 500, 4), uniform_k10_512=1 - (1 - 10 / 512) ** 500, uniform_k4_512=1 - (1 - 4 / 512) ** 500)
    out["coverage_500_tokens_model"] = cov
    r_io = mean(c["k4_500"] / c["k8_500"] for c in cov.values()); r_io_u = (1 - (1 - 4 / 512) ** 500) / (1 - (1 - 10 / 512) ** 500)
    last = [wu[k][[x for x in wu[k]][-1]]["k4"] / wu[k][[x for x in wu[k]][-1]]["k8"] for k in wu]
    out["io_ratio_k4_over_k10"] = dict(model_500_on_traces=r_io, uniform_125b=r_io_u, window_all_tokens=mean(last))
    flop_k4 = DENSE_FRAC + (1 - DENSE_FRAC) * 0.4
    comp10, comp4 = PROMPT * FLOP_TOK / PREFILL_GFLOPS, PROMPT * FLOP_TOK * flop_k4 / PREFILL_GFLOPS; rows = {}
    for tag, ratio in (("trace_model_500", r_io), ("window_all_tokens", mean(last)), ("uniform_125b", r_io_u)):
        io10 = MODEL_GB / R_4M; io4 = io10 * ratio
        rows[tag] = dict(io10_s=io10, io4_s=io4, comp10_s=comp10, comp4_s=comp4, ttft10_overlap=max(io10, comp10), ttft4_overlap=max(io4, comp4), gain_overlap=max(io10, comp10) / max(io4, comp4),
                         ttft10_serial=io10 + comp10, ttft4_serial=io4 + comp4, gain_serial=(io10 + comp10) / (io4 + comp4))
    out["ttft_500_tokens_125b"] = rows; out["flop_factor_k4"] = flop_k4
    cap = int(2.1 * FOOT); dm = {}
    for k in (8, 4, 3): dm[str(k)] = {nm[-5:]: decode_misses(nm, k, cap) for nm in NAMES}
    out["decode_misses_per_token_fixed_cache_2p1fp"] = dm
    out["decode_miss_ratio_k4"] = mean(dm["4"][n] / dm["8"][n] for n in dm["4"])
    out["decode_compute_ceiling_k4"] = G_DECODE / (FLOP_TOK * flop_k4)
    # hit curve: same trace, entries scaled to the same number of footprints of the SMALLER selection
    out["lru_hit_equal_footprints"] = {str(k): {str(fp): lru_hit_fp(fp, k) for fp in (1.0, 2.1, 4.0)} for k in (8, 4)}
    out["unmeasured"] = ["quality of k=4 (needs perplexity/benchmark runs; cited elsewhere: 77.87 -> 70.72 MMLU for 8 -> 4 on Qwen3-30B per the paper)", "real prefill coverage (decode traces used)",
                         "prefill tok/s ratio with the expert-count override (needs a llama-bench run on a model)", "whether the first 4 ids of the trace are in gate-score order (assumed)"]
    return out

# ------------------------------------------------------------------ family s1 #12: GPU tier
def hash_geometry(tokens, s, seed=7):
    """Trace with each expert id split into s sub-ids chosen by a deterministic per-(token,layer,expert) hash: an s-times sparser expert geometry (pessimistic: sub-experts independent)."""
    out = []
    for t, tk in enumerate(tokens):
        row = {}
        for l in range(NL):
            row[l] = tuple(e * s + ((((t * 73856093) ^ (l * 19349663) ^ (e * 83492791) ^ (seed * 2654435761)) * 2654435761 >> 13) % s) for e in tk[l])
        out.append(row)
    return out

def fam_s1_12():
    out = {}; fr = (0.06, 0.12, 0.17, 0.29, 0.50)
    same = {}
    for f in fr:
        same[str(f)] = {nm[-5:]: lru_scored(flat(load(nm)), int(f * PAIRS)) for nm in NAMES}
    out["lru_hit_same_prompt"] = same
    warm = {}
    for f in (0.29,):
        n = int(f * PAIRS); r = {}
        for nm in NAMES:
            o = [x for x in NAMES if x != nm][0]; r[nm[-5:]] = lru_scored(flat(load(o)) + flat(load(nm)), n, len(flat(load(o))))
        warm[str(f)] = r
    out["lru_hit_warm_from_other_prompt"] = warm
    out["pin_hit_cross_prompt_0.29"] = {nm[-5:]: pin_hit(nm, cross_pins(nm, int(0.29 * PAIRS))) for nm in NAMES}
    sparse = {}
    for s in (1, 2, 4):
        row = {}
        for f in (0.12, 0.29, 0.50):
            row[str(f)] = mean(lru_scored(flat(hash_geometry(load(nm), s)), int(f * PAIRS * s)) for nm in NAMES)
        sparse[str(s)] = row
    out["lru_hit_by_sparsity_fraction_of_experts"] = sparse
    out["control_hash_s1_equals_original"] = flat(hash_geometry(load(NAMES[0]), 1)) == flat(load(NAMES[0]))
    out["unit_error_check"] = dict(claimed_slice_mb=0.9, real_expert_mb=2.7, claim_gb_per_token=278 * 0.9e-3, corrected_gb_per_token=278 * 2.7e-3, claimed_toks_at_1p56=1 / (278 * 0.9e-3 / 1.56),
                                   corrected_toks_at_1p56_io_only=1 / (278 * 2.7e-3 / 1.56))
    tiers = {}
    for tag, gb in (("vram_6gb", 5.0), ("vram_12gb", 10.0), ("vram_24gb", 20.0)):
        f = gb / MODEL_GB; fp = fp_of_gib(gb / 1.074)
        h_dense = mean(lru_scored(flat(load(nm)), int(f * PAIRS)) for nm in NAMES)
        h_sparse = sparse["4"]["0.29"] if gb == 20.0 else mean(lru_scored(flat(hash_geometry(load(nm), 4)), int(f * PAIRS * 4)) for nm in NAMES)
        r = {}
        for tag2, h in (("dense30b", h_dense), ("sparse4x", h_sparse)):
            b = (1 - h) * EXP_GB; r[tag2] = dict(hit=h, io_bound_toks=(R_4M / b if b > 0 else None), serial_cpu_ceiling=thr(h, R_4M)[0], overlap_cpu_ceiling=thr(h, R_4M)[1])
        r["cache_gb"] = gb; r["footprints_125b"] = fp; tiers[tag] = r
    out["vram_tiers"] = tiers
    out["cpu_only_best_path_tok_s"] = {str(ram): dict(zip(("serial", "overlap"), thr(lru_hit_fp(fp_of_gib(CACHE_GIB[ram])), R_4M))) for ram in (8, 16, 32)}
    out["gpu_ceiling_assumed_toks"] = dict(value=1.0 / (6e9 / 300e9 + 0.005), tag="[A] 6 GB active weights at 300 GB/s VRAM plus 5 ms overhead; NOT measured")
    out["unmeasured"] = ["any GPU timing (no discrete GPU here)", "OLMoE (64 experts) hit sweep (no OLMoE routing trace in the repo)", "125B (512 experts) real hit rates", "PCIe transfer costs of misses"]
    return out

# ------------------------------------------------------------------ family s1 #13: top-4 for TTFT (+ decode)
def fam_s1_13():
    out = {}; cap = int(2.1 * FOOT)
    dm = {str(k): {nm[-5:]: decode_misses(nm, k, cap) for nm in NAMES} for k in (8, 6, 4, 3)}
    out["decode_misses_per_token_fixed_cache_2p1fp"] = dm
    out["decode_miss_ratio"] = {k: mean(dm[k][n] / dm["8"][n] for n in dm["8"]) for k in dm}
    out["prefill_touch_probability_125b"] = dict(top10=1 - (1 - 10 / 512) ** 500, top4=1 - (1 - 4 / 512) ** 500)
    f4 = DENSE_FRAC + (1 - DENSE_FRAC) * 0.4
    out["flop_split"] = dict(dense_frac=DENSE_FRAC, expert_frac=1 - DENSE_FRAC, flop_factor_k4=f4, compute_gain_k4=1 / f4, claim_ignores_dense=2.5)
    comp10, comp4 = PROMPT * FLOP_TOK / PREFILL_GFLOPS, PROMPT * FLOP_TOK * f4 / PREFILL_GFLOPS
    io = MODEL_GB / R_4M; io4 = io * out["prefill_touch_probability_125b"]["top4"] / out["prefill_touch_probability_125b"]["top10"]
    out["prefill_ttft_s"] = dict(k10_overlap=max(io, comp10), k4_overlap=max(io4, comp4), gain_overlap=max(io, comp10) / max(io4, comp4), k10_serial=io + comp10, k4_serial=io4 + comp4,
                                 gain_serial=(io + comp10) / (io4 + comp4), compute_floor_k10=comp10, compute_floor_k4=comp4, io_floor_k10=io, io_floor_k4=io4,
                                 compute_floor_if_55gflops=PROMPT * FLOP_TOK / G_DECODE)
    ceil4 = G_DECODE / (FLOP_TOK * f4); dec = {}
    ratio = out["decode_miss_ratio"]["4"]
    for ram in (8, 16, 32):
        fp = fp_of_gib(CACHE_GIB[ram]); h8 = lru_hit_fp(fp); capr = int(fp * FOOT)
        m8 = mean(decode_misses(nm, 8, capr) for nm in NAMES); m4 = mean(decode_misses(nm, 4, capr) for nm in NAMES); r = m4 / m8
        b8 = (1 - h8) * EXP_GB; b4 = b8 * r
        s8 = 1 / (b8 / R_4M + 1 / COMPUTE); s4 = 1 / (b4 / R_4M + 1 / ceil4); o8 = min(COMPUTE, R_4M / b8) if b8 else COMPUTE; o4 = min(ceil4, R_4M / b4) if b4 else ceil4
        dec[str(ram)] = dict(footprints=fp, hit_k8=h8, miss_ratio_k4=r, serial_k8=s8, serial_k4=s4, overlap_k8=o8, overlap_k4=o4, gain_serial=s4 / s8, gain_overlap=o4 / o8,
                             gain_vs_mmap_serial=s4 / thr(h8, R_MMAP)[0])
    out["decode_top4_tok_s"] = dec; out["decode_compute_ceiling_k4"] = ceil4
    out["quality_cited_olmoe_relative_ppl_loss"] = dict(top6=0.088, top4=0.456, note="earlier project measurement on OLMoE (requantised), not re-run here")
    out["unmeasured"] = ["perplexity/KL for k=4 on any model here (no text corpus or model run)", "prefill speed-up with the expert-count override (needs llama-bench)", "GPU offload of the dense part (no discrete GPU)",
                         "whether prefill is read- or compute-bound on a real 8 GB machine"]
    return out

# ------------------------------------------------------------------ family s1 #14: repacked expert extents
def fam_s1_14():
    out = {}; rows = {}
    for ram in (8, 16, 32):
        fp = fp_of_gib(CACHE_GIB[ram]); h = lru_hit_fp(fp); r = {}
        for tag, rate in (("mmap", R_MMAP), ("read_1mib", R_1M), ("read_0.9mb", R_09), ("read_2.7mb", R_27), ("read_4mib", R_4M)): r[tag] = dict(zip(("serial", "overlap"), thr(h, rate)))
        r["footprints"] = fp; r["hit"] = h
        r["gain_4mib_vs_mmap"] = dict(serial=r["read_4mib"]["serial"] / r["mmap"]["serial"], overlap=r["read_4mib"]["overlap"] / r["mmap"]["overlap"])
        r["gain_2.7mb_vs_mmap"] = dict(serial=r["read_2.7mb"]["serial"] / r["mmap"]["serial"], overlap=r["read_2.7mb"]["overlap"] / r["mmap"]["overlap"])
        r["gain_repacked_2.7mb_vs_stock_split_0.9mb"] = r["read_2.7mb"]["serial"] / r["read_0.9mb"]["serial"]
        rows[str(ram)] = r
    out["by_ram"] = rows
    out["reads_per_token_30b"] = dict(stock=FOOT * 3, repacked=FOOT)
    out["padding_4k_overhead_2p7mb_expert"] = (align_up(2700000, 4096) - 2700000) / 2700000
    out["prefill_load_s_69gb"] = dict(mmap=MODEL_GB / R_MMAP, explicit=MODEL_GB / R_4M, gain=R_4M / R_MMAP)
    p = _gguf_try()
    out["stock_layout_of_local_gguf"] = layout_report(p) if p else None
    out["io_microbench"] = io_results()
    io = out["io_microbench"]
    if io:
        a, b = io_rate(io, "pread nocache", 2.7, 8), io_rate(io, "mmap copy", 2.7, 8)
        out["measured_gain_pread_over_mmap_2.7mb_8thr"] = (a / b if a and b else None)
    out["unmeasured"] = ["a real repacked file and loader (only the byte layout and read rates are examined)", "125B file (the 30B file stands in for the layout)", "Windows/Linux direct-I/O alignment behaviour"]
    return out

def _gguf_try():
    p = gguf_path(); return p if os.path.exists(p) else None

# ------------------------------------------------------------------ family s1 #15: no-cache slab pread
def prev_reuse(name):
    return tt.persistence(load(name))

def fam_s1_15():
    out = {}
    out["prev_token_reuse"] = {nm[-5:]: prev_reuse(nm) for nm in NAMES}
    out["lru_hit_at_footprints"] = {str(k): {nm[-5:]: lru_scored(flat(load(nm)), int(k * FOOT)) for nm in NAMES} for k in (1, 2, 3)}
    pol = {"none": 0.0, "prev": mean(out["prev_token_reuse"].values()), "lru2": mean(out["lru_hit_at_footprints"]["2"].values())}
    tp = {}
    for rtag, rate in (("4mib_2.4", R_4M), ("1mib_1.56", R_1M), ("0.9mb_1.3", R_09)):
        tp[rtag] = {p: dict(hit=h, **dict(zip(("serial", "overlap"), thr(h, rate)))) for p, h in pol.items()}
    out["throughput_by_read_size"] = tp
    mm = thr(0.56, R_MMAP); best = thr(0.56, R_4M)
    out["baselines_8gb"] = dict(mmap=dict(zip(("serial", "overlap"), mm)), best_path=dict(zip(("serial", "overlap"), best)))
    n = tp["4mib_2.4"]["none"]
    out["none_vs_mmap_8gb"] = dict(serial=n["serial"] / mm[0], overlap=n["overlap"] / mm[1])
    out["none_vs_best_path_8gb"] = dict(serial=n["serial"] / best[0], overlap=n["overlap"] / best[1])
    out["io_time_speedup_vs_none"] = {p: (1 - 0.0) / (1 - h) for p, h in pol.items()}
    out["kill_rule_policy_beats_none_by_over_1.7x_io"] = {p: (1.0 / (1 - h)) > 1.7 for p, h in pol.items()}
    p = _gguf_try()
    out["stock_layout_of_local_gguf"] = layout_report(p) if p else None
    out["io_replay"] = io_results()
    io = out["io_replay"]
    if io:
        d = {(r["policy"], r["variant"]): r for r in io["replay"]}
        a, b = d.get(("none", "split3")), d.get(("none", "mmap"))
        out["measured_none_split3_vs_mmap_replay_speedup"] = (b["sec_per_token"] / a["sec_per_token"] if a and b and a["sec_per_token"] else None)
    out["unmeasured"] = ["end-to-end tokens/s of any read path (no model run)", "OS read-ahead or page-cache interactions of a real run", "125B expert offsets (the 30B file stands in)", "macOS F_NOCACHE vs true O_DIRECT behaviour on other systems"]
    return out

FAMILIES = {
 "s1_8": {"idea": "Pin a static expert set (26 GB of the 125B) and bias router logits so about half of each token's experts come from it.",
          "methods": ["Cache-Aware Joint Router Adaptation for Memory-Efficient MoE Inference", "llama.cpp discussion: MoE offload to disk with on-demand paging (RFC+PoC)",
                      "Mixtral-offloading / SpecMD study of speculative prefetch", "Research Square paper with a '0.15 rule' on a llama.cpp-based engine"], "run": fam_s1_8, "needs": "model"},
 "s1_9": {"idea": "Run prefill expert-major: read each expert once per layer and apply it to all prompt positions.",
          "methods": ["SeqMoE: predictive and graph-compatible MoE offloading", "ExpertFlow (token scheduler grouping tokens by routing path)", "MoE-Lightning (CGOPipe, paged weights, batch offloading)",
                      "fMoE fine-grained expert offloading (TTFT comparison)", "Reddit/blog: -ub 512 -> larger ubatch raises prompt processing on gpt-oss-120b partially offloaded"], "run": fam_s1_9, "needs": "model"},
 "s1_10": {"idea": "After prefill, pin the expert slices the prompt loaded (36% of the experts) and stream the rest.",
           "methods": ["ProMoE: proactive caching for MoE offloading", "DALI: workload-aware offloading for local PCs", "ELDR: expert-locality-aware decode routing", "Fate (cross-layer prediction, shallow-favoring expert cache)",
                       "MoE-Infinity: sparsity-aware expert cache", "In-depth analysis of caching and prefetching in MoE offloading", "llama.cpp issue #20757", "FlashMoE: ML-based cache replacement to cut SSD I/O"], "run": fam_s1_10, "needs": "model"},
 "s1_11": {"idea": "Cap routing to the top-4 experts per token during prefill only so fewer expert slices are read.",
           "methods": ["Prefill-stage top-k reduction (paper title not retrieved)", "Lynx dynamic batch-aware expert selection", "Alloc-MoE budget-aware expert activation",
                       "Distribution-consistent inference for dynamic sparse MoE", "Training-free halving of activated experts", "llama.cpp expert_used_count override (existing way to run this test)"], "run": fam_s1_11, "needs": "model"},
 "s1_12": {"idea": "Add a discrete GPU as a resident tier (20 GB of hot experts in VRAM) and lift the CPU compute ceiling.",
           "methods": ["llama.cpp GPU expert cache --moe-cache-mib (mainline, per blog citing PR #29887)", "KTransformers hot experts on GPU, cold on CPU", "Fiddler CPU-GPU orchestration", "HybriMoE / HOBBIT / MoE-Infinity / DAOP"], "run": fam_s1_12, "needs": "hardware"},
 "s1_13": {"idea": "Lower active experts from top-10 to top-4 (runtime override) and put the dense part on a GPU to cut TTFT.",
           "methods": ["LExI layer-adaptive active experts, training-free", "Expert-Sample, fine-grained MoE", "MoDES dynamic expert skipping (Qwen3-VL-MoE-30B)", "llama.cpp runtime top-k override request and --override-kv workaround", "Hybrid CPU-GPU MoE TTFT"],
           "run": fam_s1_13, "needs": "model"},
 "s1_14": {"idea": "Repack GGUF expert tensors so each expert is one contiguous aligned extent read with cache-bypassing reads.",
           "methods": ["SGLang Expert Pack (LMSYS blog, 2026-08-29)", "BigMoeOnEdge benchmarks (gpt-oss)", "Flash-MoE (397B Qwen3.5 on a 48 GB MacBook)", "Jetson Orin Nano streaming fork (dev.to)", "Apple, LLM in a flash (2023); HN comment on io_uring layer in llama.cpp fork"],
           "run": fam_s1_14, "needs": "local"},
 "s1_15": {"idea": "Drop the mmap fault path and any cache: one batched expert-aligned pread per routed expert into a slab each token.",
           "methods": ["LLM in a flash (Apple)", "Edge0 (arXiv 2609.18063)", "llama.cpp discussion 19883"], "run": fam_s1_15, "needs": "local"},
}

if __name__ == "__main__":
    keys = sys.argv[1:] or list(FAMILIES)
    for k in keys: print(json.dumps({k: FAMILIES[k]["run"]()}, indent=1, default=str))
