"""Published-technique family tests, group F: screen 2 families 18..25.
One function per family (fam_s2_18 .. fam_s2_25) returning a dict of measured or computed numbers, plus the FAMILIES registry.
Everything runs on the three recorded Qwen3-30B router traces (data/traces, 128 experts, top-8, 48 layers) and on closed-form arithmetic.
Trace results are mapped to the 125B target by token-footprints (cache entries = footprints x 384 accesses per token at top-8; 8/16/32 GB machines
have about 2.5/9.15/25.15 GiB of expert cache = 2.07/7.6/20.8 footprints). The 30B traces stand in for the 125B: every byte figure is a projection.
Optional inputs (never downloaded): the 30B Q4_K_M weights file (env PUBTECH_GGUF, default in the user's Downloads folder) for the weight-level
measurements of families 19 and 25; a llama.cpp build and OLMoE/tiny GGUF files (env LLAMA_BIN, PUBTECH_OLMOE, PUBTECH_TINY) for the optional
timing helpers of families 20 and 23, which are NOT part of the default runs because they are not deterministic.
"""
import collections, json, math, operator, os, random, re, struct, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if TOOLS not in sys.path: sys.path.insert(0, TOOLS)
import cachesim, combo_model as cm, combo_trace_sim as cs   # noqa: E402

# ------------------------------------------------------------------ constants (tags: [M] measured earlier, [E] estimate)
LAYERS, TOP30, E30, E125, TOP125, SL = 48, 8, 128, 512, 10, 480
SLICE_MB = 2.7                                   # [E] gate+up+down of one 125B expert at 4.3 bpw
EXP_GB = SL * SLICE_MB / 1000.0                  # 1.296 GB of expert bytes per token at hit 0
ALL_EXPERT_GB = E125 * LAYERS * SLICE_MB / 1000  # 66.4 GB of expert weights (the notes say 69 GB incl. dense)
FLOP_TOK, G_DEC, G_PRE = 12.0, 55.0, 311.0       # GFLOP per token; GFLOP/s decode [M]; GFLOP/s batched prefill on OLMoE [M]
COMP_DEC = FLOP_TOK / G_DEC                      # 0.2182 s per token = the 4.58 tok/s compute ceiling
R_MMAP, R_SCREEN, R_LOCAL = 0.34, 2.4, 3.11      # GB/s [M]: mmap paging; the screen's explicit-read figure; combo model's explicit 2.7 MB x4 reads
RAMS = (8, 16, 32)
TRACES = list(cs.TRACES)


def footprints(ram):
    """Expert-cache size of an `ram` GB machine in token-footprints of the 125B (480 slices x 2.7 MB)."""
    return cs.CACHE_GIB[ram] * 1.074e9 / (SLICE_MB * 1e6) / SL


_HIT = {}
def lru_hit(trace, fp):
    """Cold-start-included LRU hit rate of one trace at `fp` token-footprints (cache entries = fp x 384)."""
    key = (trace, round(fp, 9))
    if key not in _HIT:
        T = cs.tr(trace); acc = [(l, e) for tok in T for l in range(LAYERS) for e in tok[l]]
        _HIT[key] = cachesim.lru_hits(acc, int(fp * cs.FOOT_ENTRIES)) / len(acc)
    return _HIT[key]


def mean_hit(fp): return sum(lru_hit(t, fp) for t in TRACES) / len(TRACES)


def serial_toks(h, rate, comp=COMP_DEC, exp_gb=EXP_GB):
    """Serial tok/s: compute plus the missed expert bytes read at `rate` GB/s."""
    return 1.0 / (comp + (1 - h) * exp_gb / rate)


def overlap_toks(h, rate, comp=COMP_DEC, exp_gb=EXP_GB):
    """Overlap upper bound tok/s: perfect overlap of reads with compute, 1/max(compute, io)."""
    return 1.0 / max(comp, (1 - h) * exp_gb / rate)


def _r(x, n=4): return None if x is None else round(x, n)


# ------------------------------------------------------------------ pass-cost table shared by families 21 and 24 (verify / batched forward of k tokens)
_MISS = {}
def misses_per_pass(ram, k):
    """Mean over the 3 traces of expert-cache misses of one forward pass that processes k consecutive decode tokens together (union of their experts), LRU at `ram` GB."""
    if (ram, k) not in _MISS:
        v = []
        for t in TRACES:
            s = cs.simulate(t, "lru", "full", 4.3, ram, k if k > 1 else 0, 0.0); v.append(s["misses"] / s["passes"])
        _MISS[(ram, k)] = sum(v) / len(v)
    return _MISS[(ram, k)]


def _batch_eff(k):
    """Batch efficiency vs batch 1 from the combination model's likely scenario (b4, b8): [E]."""
    sc = cm.sc_at(1)
    if k <= 1: return 1.0
    if k >= 8: return sc["b8"]
    if k <= 4: return 1 + (sc["b4"] - 1) * (k - 1) / 3.0
    return sc["b4"] + (sc["b8"] - sc["b4"]) * (k - 4) / 4.0


def pass_cost(ram, k, model="screen", rate=R_LOCAL):
    """Seconds of ONE forward over k tokens at 125B scale. model 'screen': k=1 at 55 GFLOP/s, k>=2 at 311 GFLOP/s (the screens' assumption);
    model 'combo': 55 GFLOP/s times the combination model's batch efficiency. Returns dict(io, comp, serial, overlap)."""
    io = misses_per_pass(ram, k) * (SL / cs.FOOT_ENTRIES) * SLICE_MB * 1e6 / 1e9 / rate
    comp = COMP_DEC if k == 1 else (k * FLOP_TOK / G_PRE if model == "screen" else k * FLOP_TOK / (G_DEC * _batch_eff(k)))
    return dict(io=io, comp=comp, serial=io + comp, overlap=max(io, comp))


def cost_of_sizes(sizes, ram, model="screen", mode="serial", kmax=16):
    """Total seconds of a list of forward sizes (a size above kmax is split into kmax-sized forwards)."""
    tot = 0.0
    for s in sizes:
        while s > kmax: tot += pass_cost(ram, kmax, model)[mode]; s -= kmax
        if s > 0: tot += pass_cost(ram, s, model)[mode]
    return tot


# ================================================================== family 18: expert store in a peer machine's RAM over RDMA
def fam_s2_18():
    """Closed form + trace LRU: tok/s with the missed expert bytes read from local NVMe vs a RAM peer at 100GbE/TB5 speed."""
    rates = {"mmap_paging": R_MMAP, "nvme_screen_2.4": R_SCREEN, "nvme_explicit_3.11": R_LOCAL, "tcp_10gbe_1.1": 1.1, "rdma_tb5_10": 10.0, "rdma_100gbe_11": 11.0}
    out = {"ceiling_tok_s": _r(1 / COMP_DEC), "footprints": {}, "hit": {}, "serial": {}, "overlap": {}, "gain_vs_nvme311": {}, "gain_vs_nvme24": {}, "gain_vs_mmap": {}, "combo_model_serial": {}}
    c = dict(K=10, bits=4.3, lowrank=0, route=0, overlap=0, spec=0, gpu=0, lut=0, exit=1.0); sc = cm.sc_at(1)
    for ram in RAMS:
        fp = footprints(ram); h = mean_hit(fp); out["footprints"][ram] = _r(fp); out["hit"][ram] = _r(h)
        out["serial"][ram] = {n: _r(serial_toks(h, r)) for n, r in rates.items()}
        out["overlap"][ram] = {n: _r(overlap_toks(h, r)) for n, r in rates.items()}
        for key, base in (("gain_vs_nvme311", "nvme_explicit_3.11"), ("gain_vs_nvme24", "nvme_screen_2.4"), ("gain_vs_mmap", "mmap_paging")):
            out[key][ram] = {"serial": _r(out["serial"][ram]["rdma_100gbe_11"] / out["serial"][ram][base]), "overlap": _r(out["overlap"][ram]["rdma_100gbe_11"] / out["overlap"][ram][base])}
        out["combo_model_serial"][ram] = {str(r): _r(cm.tok_s(c, f"{ram} GB", sc, rate=r)) for r in (R_LOCAL, 11.0)}
    # the claim's own arithmetic
    out["claim"] = {"io_ms_peer_nocache": _r(EXP_GB / 11.0 * 1000, 1), "io_ms_nvme24_nocache": _r(EXP_GB / R_SCREEN * 1000, 1), "io_ratio": _r(11.0 / R_SCREEN, 3),
                    "pure_io_tok_s_at_11GBs": _r(11.0 / EXP_GB, 2), "exceeds_compute_ceiling": 11.0 / EXP_GB > 1 / COMP_DEC,
                    "serial_tok_s_nocache_peer": _r(serial_toks(0.0, 11.0)), "serial_tok_s_nocache_nvme24": _r(serial_toks(0.0, R_SCREEN))}
    # rate at which reads stop being the bottleneck under perfect overlap (per machine) and per-fetch latency cost
    out["rate_to_hit_ceiling_GBs"] = {ram: _r((1 - out["hit"][ram]) * EXP_GB / COMP_DEC) for ram in RAMS}
    out["fetch_latency_ms_per_token"] = {ram: {"rdma_10us": _r((1 - out["hit"][ram]) * SL * 10e-6 * 1000, 3), "tcp_100us": _r((1 - out["hit"][ram]) * SL * 100e-6 * 1000, 3)} for ram in RAMS}
    out["unmeasured"] = ["any real peer/RDMA/TB5 transfer (no hardware)", "per-fetch latency and queueing of 2.7 MB reads over a real fabric", "the 'decode on the peer' variant (uses the peer's compute, a different machine)"]
    return out


# ================================================================== family 19: quantized residual against the previous layer's predicted expert
def bytes_bound(r, p=0.55):
    """Upper bound on the bytes gain when a share p of slices (the predicted ones) shrinks by factor r and the rest is read in full: 1/((1-p)+p/r)."""
    return 1.0 / ((1 - p) + p / r)


def bits_saved(ratio):
    """High-resolution estimate of bits per weight saved by coding W as c*B + residual instead of W, from the residual/target RMS ratio (-log2 ratio, never negative)."""
    return max(0.0, -math.log2(max(ratio, 1e-12)))


def comp_ratio(saved, bpw=4.5):
    """Residual compression factor r = bpw / (bpw - saved) (inf when the saving covers the whole word)."""
    return float("inf") if saved >= bpw else bpw / (bpw - saved)


def dot(a, b): return sum(map(operator.mul, a, b))


def cos_and_ratio(a, b):
    """Cosine of a and b, and RMS(a - c*b)/RMS(a) with the least-squares scalar c. The identity ratio = sqrt(1 - cos^2) holds exactly."""
    na, nb, ab = dot(a, a), dot(b, b), dot(a, b)
    if na == 0 or nb == 0: return 0.0, 1.0
    cos = ab / math.sqrt(na * nb); return cos, math.sqrt(max(0.0, 1 - cos * cos))


# ---- minimal GGUF reader (header, tensor table) and Q4_K dequantizer: stdlib only
_GT = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_GTF = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
Q4K_BLOCK, Q4K_BYTES = 256, 144


def _skip(f, t):
    if t in _GT: f.seek(_GT[t], 1)
    elif t == 8: n = struct.unpack("<Q", f.read(8))[0]; f.seek(n, 1)
    elif t == 9:
        et, n = struct.unpack("<IQ", f.read(12))
        if et in _GT: f.seek(_GT[et] * n, 1)
        else:
            for _ in range(n): _skip(f, et)
    else: raise ValueError("unknown gguf value type %d" % t)


def read_gguf(path):
    """Returns dict(version, kv (scalars and short strings only), tensors {name: dict(dims, type, offset)}, data_start). Raises ValueError on a bad file."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF": raise ValueError("not a gguf file")
        ver, nt, nkv = struct.unpack("<IQQ", f.read(20)); kv = {}
        for _ in range(nkv):
            kl = struct.unpack("<Q", f.read(8))[0]; key = f.read(kl).decode("utf-8", "replace"); t = struct.unpack("<I", f.read(4))[0]
            if t in _GT: kv[key] = struct.unpack(_GTF[t], f.read(_GT[t]))[0]
            elif t == 8:
                n = struct.unpack("<Q", f.read(8))[0]; s = f.read(n); kv[key] = s.decode("utf-8", "replace") if n < 256 else None
            else: _skip(f, t)
        tensors = {}
        for _ in range(nt):
            nl = struct.unpack("<Q", f.read(8))[0]; name = f.read(nl).decode("utf-8", "replace"); nd = struct.unpack("<I", f.read(4))[0]
            dims = list(struct.unpack("<%dQ" % nd, f.read(8 * nd))); typ, off = struct.unpack("<IQ", f.read(12)); tensors[name] = dict(dims=dims, type=typ, offset=off)
        pos = f.tell(); al = int(kv.get("general.alignment", 32)); data_start = (pos + al - 1) // al * al
    return dict(version=ver, kv=kv, tensors=tensors, data_start=data_start)


def _scales_mins_k4(sc):
    s = [0] * 8; m = [0] * 8
    for j in range(4): s[j] = sc[j] & 63; m[j] = sc[j + 4] & 63
    for j in range(4, 8): s[j] = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4); m[j] = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4)
    return s, m


def dequant_q4k(buf):
    """Dequantize whole Q4_K blocks (144 bytes -> 256 floats): y = d*scale*q - dmin*min, 64-weight groups, low nibbles first then high nibbles."""
    out = []
    for b in range(len(buf) // Q4K_BYTES):
        blk = buf[b * Q4K_BYTES:(b + 1) * Q4K_BYTES]; d, dmin = struct.unpack_from("<ee", blk, 0); s, m = _scales_mins_k4(blk[4:16]); qs = blk[16:144]
        for j in range(4):
            q = qs[32 * j:32 * j + 32]; d1, m1, d2, m2 = d * s[2 * j], dmin * m[2 * j], d * s[2 * j + 1], dmin * m[2 * j + 1]
            out.extend([d1 * (x & 15) - m1 for x in q]); out.extend([d2 * (x >> 4) - m2 for x in q])
    return out


def build_gguf(tensors, align=32):
    """Tiny GGUF writer for tests: tensors = [(name, dims, type, raw bytes)]. No metadata beyond the alignment."""
    def s(x): b = x.encode(); return struct.pack("<Q", len(b)) + b
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 1) + s("general.alignment") + struct.pack("<II", 4, align)
    infos = b""; off = 0; datas = []
    for name, dims, typ, raw in tensors:
        infos += s(name) + struct.pack("<I", len(dims)) + struct.pack("<%dQ" % len(dims), *dims) + struct.pack("<IQ", typ, off); pad = (-len(raw)) % align; datas.append(raw + b"\0" * pad); off += len(raw) + pad
    pre = head + infos; pre += b"\0" * ((-len(pre)) % align)
    return pre + b"".join(datas)


def gguf_path():
    p = os.environ.get("PUBTECH_GGUF") or os.path.join(os.path.expanduser("~"), "Downloads", "Qwen3-30B-A3B-Q4_K_M.gguf")
    return p if os.path.exists(p) else None


class Weights:
    """Row-sampled access to the per-expert tensors of the 30B GGUF (Q4_K): rows are read directly from disk, nothing is loaded whole."""
    ROWS = [i * 48 for i in range(16)]       # 16 rows spread over the 768 rows of an expert matrix (16 x 2048 = 32,768 weights per expert)
    def __init__(self, path):
        self.path = path; self.h = read_gguf(path); self.memo = {}
    def info(self, layer, kind="gate"): return self.h["tensors"]["blk.%d.ffn_%s_exps.weight" % (layer, kind)]
    def ok(self, layer, kind="gate"):
        try: i = self.info(layer, kind); return i["type"] == 12 and len(i["dims"]) == 3
        except KeyError: return False
    def expert(self, layer, e, kind="gate"):
        key = (layer, e, kind)
        if key not in self.memo:
            i = self.info(layer, kind); ne0, ne1 = i["dims"][0], i["dims"][1]; rb = ne0 // Q4K_BLOCK * Q4K_BYTES; vec = []
            with open(self.path, "rb") as f:
                for r in self.ROWS:
                    f.seek(self.h["data_start"] + i["offset"] + (e * ne1 + r) * rb); vec.extend(dequant_q4k(f.read(rb)))
            self.memo[key] = vec
        return self.memo[key]


_W = {}
def weights():
    p = gguf_path()
    if p is None: return None
    if p not in _W:
        try: _W[p] = Weights(p)
        except Exception: _W[p] = None
    return _W[p]


def predictor_precision(trace, train_frac=1.0):
    """Cross-layer predictor: the top-8 experts of layer L by summed co-activation counts with the experts chosen at layer L-1. train_frac=1 is in-sample (leaks);
    otherwise the first train_frac of the tokens train and the rest test (held out). Returns mean precision (|pred & actual| / 8) over layers 1..47 and test tokens."""
    T = cs.tr(trace); cut = len(T) if train_frac >= 1 else int(len(T) * train_frac); prec = []
    for L in range(1, LAYERS):
        C = collections.defaultdict(collections.Counter)
        for tok in T[:cut]:
            for b in tok[L - 1]:
                for a in tok[L]: C[b][a] += 1
        for tok in (T[:cut] if train_frac >= 1 else T[cut:]):
            sc = collections.Counter()
            for b in tok[L - 1]: sc.update(C[b])
            pred = {a for a, _ in sc.most_common(TOP30)}; prec.append(len(pred & set(tok[L])) / TOP30)
    return sum(prec) / len(prec)


def coactivation_base(layer, a):
    """For expert a of `layer`, the layer-1 expert that most often fires on the same token, over all three traces (ties: lowest id)."""
    c = collections.Counter()
    for t in TRACES:
        for tok in cs.tr(t):
            if a in tok[layer]:
                for b in tok[layer - 1]: c[b] += 1
    return min(c, key=lambda b: (-c[b], b)) if c else None


def median(v):
    s = sorted(v); n = len(s); return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def weight_pair_stats(layer=11, n_target=8, n_cand=24, seed=19):
    """Measured on the real 30B Q4_K weights: cosine and LS-residual ratio between experts of layer-1 (bases) and layer (targets), gate matrices, 16 sampled rows per expert.
    Returns None when the weights file is absent."""
    W = weights()
    if W is None or not (W.ok(layer) and W.ok(layer - 1)): return None
    perm = random.Random(seed).sample(range(E30), E30); tg = perm[:n_target]; cand = perm[n_target:n_target + n_cand]
    cross = []; best = []
    for a in tg:
        va = W.expert(layer, a); rs = []
        for b in cand:
            cos, ratio = cos_and_ratio(va, W.expert(layer - 1, b)); cross.append((abs(cos), ratio)); rs.append(ratio)
        best.append(min(rs))
    same = []
    for i in range(len(tg)):
        for j in range(i + 1, len(tg)):
            cos, ratio = cos_and_ratio(W.expert(layer, tg[i]), W.expert(layer, tg[j])); same.append((abs(cos), ratio))
    idx = [cos_and_ratio(W.expert(layer, a), W.expert(layer - 1, a)) for a in tg]            # same expert id in the previous layer
    pred = []
    for a in tg:
        b = coactivation_base(layer, a)
        if b is not None: cos, ratio = cos_and_ratio(W.expert(layer, a), W.expert(layer - 1, b)); pred.append((abs(cos), ratio))
    st = lambda v: dict(median_abs_cos=_r(median([x[0] for x in v]), 5), max_abs_cos=_r(max(x[0] for x in v), 5), median_ratio=_r(median([x[1] for x in v]), 5), min_ratio=_r(min(x[1] for x in v), 5), n=len(v))
    out = {"cross_layer_random": st(cross), "same_layer_control": st(same), "same_expert_id_prev_layer": st([(abs(c), r) for c, r in idx]), "coactivation_base": st(pred),
           "best_of_candidates_median_ratio": _r(median(best), 5), "n_weights_per_expert": len(W.expert(layer, tg[0])), "tensor_type": W.info(layer)["type"]}
    out["saved_bits_median"] = _r(bits_saved(out["cross_layer_random"]["median_ratio"]), 5)
    out["saved_bits_best_of_candidates"] = _r(bits_saved(out["best_of_candidates_median_ratio"]), 5)
    out["saved_bits_coactivation"] = _r(bits_saved(out["coactivation_base"]["median_ratio"]), 5)
    return out


def fam_s2_19():
    held = sum(predictor_precision(t, 0.5) for t in TRACES) / 3; insamp = sum(predictor_precision(t, 1.0) for t in TRACES) / 3
    ws = weight_pair_stats()
    out = {"predictor_precision_in_sample": _r(insamp), "predictor_precision_held_out": _r(held), "claimed_precision": 0.55,
           "bound_table_p055": {str(r): _r(bytes_bound(r, 0.55), 4) for r in (1.0, 1.05, 1.9, 4.0, 1e9)},
           "bound_table_p_held_out": {str(r): _r(bytes_bound(r, held), 4) for r in (1.0, 1.05, 1.9, 4.0, 1e9)},
           "claim_arith": {"mb_if_ratio_1.9": _r(1300 / 1.9, 1), "claimed_mb": 331, "ratio_implied_by_331mb": _r(1300 / 331, 3), "stack_8.7x_decomposition": _r(3.93 * 2.2, 2)},
           "weights": ws}
    if ws:
        sv = ws["saved_bits_best_of_candidates"]; r = comp_ratio(sv); out["residual_ratio_r_best_of_candidates"] = _r(r, 4)
        out["bytes_gain_bound_measured"] = _r(bytes_bound(r, 0.55), 4); out["bytes_gain_bound_measured_held_out_p"] = _r(bytes_bound(r, held), 4)
    out["unmeasured"] = ["only 16 sampled rows x 24x8 expert pairs of gate matrices of layers 10/11 of the 30B (not the 125B, not up/down, not all layers)", "end-to-end quality of residual reconstruction", "a calibrated learned base (merged/clustered experts) rather than a raw neighbour"]
    return out


# ================================================================== family 20 and 22: prefill block statistics
def uniform_frac(n, E, k): return 1.0 - (1.0 - k / E) ** n


def union_frac(trace, n):
    """Mean over layers and non-overlapping windows of n consecutive decode tokens of (distinct experts / 128). None if the trace is shorter than n."""
    T = cs.tr(trace); fr = []
    for s in range(0, len(T) - n + 1, n):
        w = T[s:s + n]; fr.append(sum(len({e for tok in w for e in tok[l]}) for l in range(LAYERS)) / (LAYERS * E30))
    return sum(fr) / len(fr) if fr else None


def union_count(trace, k):
    """Mean distinct experts per layer among k consecutive decode tokens (of 8 per token), non-overlapping windows."""
    f = union_frac(trace, k); return None if f is None else f * E30


def prefill_pass_s(n_tokens, frac):
    """Seconds of one prefill pass over n tokens at 125B scale: max(compute at the measured batched rate, expert bytes at the explicit-read rate); `frac` = share of the expert store touched."""
    return max(n_tokens * FLOP_TOK / G_PRE, frac * ALL_EXPERT_GB / R_LOCAL)


def frac_models(n, tmax):
    """Share of experts a pass over n prompt tokens touches, three ways: measured on the 30B decode traces (mean of 3; for n above the shortest trace the
    full-trace window is used = a lower bound), the same clustering ratio scaled to the 125B, and uniform routing over 512 experts, top-10."""
    n_eff = min(n, tmax); m = sum(union_frac(t, n_eff) for t in TRACES) / 3
    u30 = uniform_frac(n_eff, E30, TOP30); u125 = uniform_frac(n, E125, TOP125)
    return dict(measured30=m, scaled125=min(1.0, u125 * (m / u30)), uniform125=u125, n_eff=n_eff)


def fam_s2_20():
    tmax = min(len(cs.tr(t)) for t in TRACES)
    tab = {}
    for n in (1, 4, 8, 16, 32, 64, 128, tmax):
        tab[n] = {"measured30": {t.split("_")[1]: _r(union_frac(t, n)) for t in TRACES}, "uniform30": _r(uniform_frac(n, E30, TOP30))}
    f64, f500 = frac_models(64, tmax), frac_models(500, tmax); out = {"tokens_per_trace_min": tmax, "union_table": tab, "frac64": {k: _r(v) for k, v in f64.items()}, "frac500": {k: _r(v) for k, v in f500.items()}}
    t_comp64, t_comp500 = 64 * FLOP_TOK / G_PRE, 500 * FLOP_TOK / G_PRE
    out["compute_s"] = {"block64": _r(t_comp64, 2), "prompt500": _r(t_comp500, 2), "claimed_block_s": _r(64 * FLOP_TOK / G_DEC, 1), "claimed_full_s": _r(500 * FLOP_TOK / G_DEC, 1)}
    blk = {}
    for m in ("measured30", "scaled125", "uniform125"):
        b64, b500 = prefill_pass_s(64, f64[m]), prefill_pass_s(500, f500[m])
        blk[m] = {"gb64": _r(f64[m] * ALL_EXPERT_GB, 1), "s_block1": _r(b64, 1), "s_full": _r(b500, 1), "block1_over_full": _r(b64 / b500, 3), "touches_over_half_of_experts": f64[m] > 0.5,
                  "chunk_passes_8x64_bytes_over_one_pass": _r(8 * f64[m] / f500[m], 2)}
    out["block"] = blk
    # speculative prefill: keep a fraction q of the prompt: compute scales with q, bytes with the union of 500q tokens
    sp = {}
    for q in (1.0, 0.5, 0.25, 0.125):
        n = max(1, int(500 * q)); fr = frac_models(n, tmax)["scaled125"] if n <= 500 else 1.0
        sp[str(q)] = {"tokens": n, "frac": _r(fr), "s": _r(prefill_pass_s(n, fr), 1), "speedup_vs_full": _r(prefill_pass_s(500, f500["scaled125"]) / prefill_pass_s(n, fr), 3)}
    out["spec_prefill"] = sp
    out["kill_criteria"] = {"block1_over_half_of_full_time_scaled125": blk["scaled125"]["block1_over_full"] > 0.5, "block1_touches_over_half_of_experts_scaled125": f64["scaled125"] > 0.5}
    out["unmeasured"] = ["real prefill routing (no prefill trace exists; decode traces are a proxy and are more clustered than prefill)", "whether provisional tokens from a 64-token prompt match the full-prompt answer (needs a model)", "timing of -p 64 vs -p 500 (optional helper olmoe_prefill_timing, not run by default)"]
    return out


# ---- optional host helpers (not deterministic; used once by hand for the report)
def parse_llama_bench_json(text):
    """llama-bench -o json -> list of (n_prompt, n_gen, threads, tok/s). Tolerates leading log lines."""
    i = text.find("["); rows = json.loads(text[i:]) if i >= 0 else []
    return [(r.get("n_prompt", 0), r.get("n_gen", 0), r.get("n_threads", 0), r.get("avg_ts")) for r in rows]


def llama_bench(model, args, timeout=170):
    """Run llama-bench from env LLAMA_BIN; returns parsed rows or None when the binary or model is missing. Never called by the default runs."""
    b = os.path.join(os.environ.get("LLAMA_BIN", ""), "llama-bench")
    if not os.path.exists(b) or not model or not os.path.exists(model): return None
    env = dict(os.environ); env["DYLD_LIBRARY_PATH"] = os.environ.get("LLAMA_BIN", "")
    p = subprocess.run([b, "-m", model, "-o", "json"] + args, capture_output=True, text=True, timeout=timeout, env=env)
    return parse_llama_bench_json(p.stdout)


def olmoe_prefill_timing(model=None):
    """Step 2 of family 20: tokens/s of a 64-token and a 500-token prefill on OLMoE (time ratio = block cost over full cost)."""
    rows = llama_bench(model or os.environ.get("PUBTECH_OLMOE"), ["-p", "64,500", "-n", "0", "-t", "4", "-r", "1"])
    if not rows: return None
    d = {r[0]: r[3] for r in rows}; t64, t500 = 64 / d[64], 500 / d[500]; return dict(tok_s_p64=d[64], tok_s_p500=d[500], s_p64=t64, s_p500=t500, block_over_full=t64 / t500)


# ================================================================== family 21: prompt-lookup / suffix n-gram drafting
_TOKEN_RE = re.compile(r"\s*\w+|\s*[^\w\s]|\s+")
def tok_text(text):
    """Crude BPE stand-in: a word with its leading space, or a punctuation mark with its leading space, or a whitespace run."""
    return _TOKEN_RE.findall(text)


def pld_run(seq, start, min_n=3, max_n=4, max_draft=8):
    """Teacher-forced prompt-lookup decoding over seq[start:] (seq[:start] is the prompt). At each step the longest suffix (max_n..min_n tokens) of the context is looked up
    at its latest earlier occurrence and the tokens that followed it (up to max_draft, never beyond the context) are the draft. One forward verifies draft+1 positions,
    accepts the longest correct prefix a and yields a+1 tokens. Returns dict(sizes, accepted, drafted, out, forwards)."""
    idx = {n: {} for n in range(min_n, max_n + 1)}; nxt = min_n; i = start; sizes = []; acc = drafted = 0; N = len(seq)
    while i < N:
        for p in range(nxt, i):                              # index n-grams that END at p <= i-1
            for n in idx:
                if p >= n: idx[n][tuple(seq[p - n:p])] = p
        nxt = max(nxt, i); draft = []
        for n in range(max_n, min_n - 1, -1):
            if i >= n:
                p = idx[n].get(tuple(seq[i - n:i]))
                if p is not None and p < i: draft = seq[p:min(p + max_draft, i)]; break
        draft = draft[:N - i]; a = 0
        while a < len(draft) and draft[a] == seq[i + a]: a += 1
        sizes.append(len(draft) + 1); acc += a; drafted += len(draft); i += min(a + 1, N - i)
    return dict(sizes=sizes, accepted=acc, drafted=drafted, out=N - start, forwards=len(sizes))


def _read(rel):
    with open(os.path.join(ROOT, rel), errors="ignore") as f: return f.read()


def corpus_texts():
    """Deterministic text sets: prose (the Python documentation topics shipped with the interpreter, else the README), code (this repo's own scripts), JSON rows (seeded)."""
    try:
        import pydoc_data.topics as tp
        prose = "\n".join(tp.topics[k] for k in sorted(tp.topics)[:12])[:60000]
    except Exception:
        prose = _read("README.md") * 3
    code = "\n".join(_read(f) for f in ("tools/cachesim.py", "tools/trace_tools.py", "tools/combo_model.py", "tools/hit_rate_sim.py", "tools/reference_impl.py", "tools/combo_trace_sim.py"))
    rng = random.Random(21); names = ["alice", "bob", "carol", "dave", "erin", "frank", "grace", "heidi"]; st = ["active", "pending", "closed"]
    rows = ",\n".join(json.dumps({"id": i, "name": "%s_%d" % (rng.choice(names), rng.randrange(100)), "status": rng.choice(st), "score": round(rng.random() * 10, 2)}) for i in range(300))
    return {"prose": prose, "code": code, "json_rows": rows}


def edit_control(tokens, frac, seed):
    """Prompt = tokens; output = the same tokens with `frac` of positions replaced by random vocabulary tokens (a code-edit / rewrite stand-in). Returns (seq, start)."""
    rng = random.Random(seed); vocab = sorted(set(tokens)); out = [rng.choice(vocab) if rng.random() < frac else t for t in tokens]; return tokens + out, len(tokens)


def random_control(n, vocab, seed):
    rng = random.Random(seed); return [rng.randrange(vocab) for _ in range(n)], n // 3


def speedup_from_f(f, ram, model="screen", mode="serial", k=8):
    """The screen's formula: a fraction f of tokens is emitted through k-token spans verified in one pass with 100% acceptance; speedup = t1 / ((1-f)*t1 + f*t_k/k)."""
    t1 = pass_cost(ram, 1, model)[mode]; tk = pass_cost(ram, k, model)[mode] / k
    return t1 / ((1 - f) * t1 + f * tk)


def fam_s2_21():
    texts = corpus_texts(); sets = {}
    for name, txt in texts.items():
        seq = tok_text(txt)[:6000]; sets[name] = (seq, int(0.3 * len(seq)))
    sets["edit_control_10pct"] = edit_control(tok_text(texts["prose"])[:1500], 0.10, 211)
    sets["random_control"] = random_control(3000, 5000, 212)
    res = {}
    for name, (seq, start) in sets.items():
        r = pld_run(seq, start); n = r["out"]
        res[name] = {"out_tokens": n, "f_accepted": _r(r["accepted"] / n), "forwards_per_token": _r(r["forwards"] / n), "mean_draft": _r(r["drafted"] / max(1, r["forwards"]), 3),
                     "acceptance_of_drafted": _r(r["accepted"] / r["drafted"]) if r["drafted"] else None,
                     "speedup_serial": {ram: _r(n * pass_cost(ram, 1)["serial"] / cost_of_sizes(r["sizes"], ram, "screen", "serial")) for ram in RAMS},
                     "speedup_serial_combo_batch": {ram: _r(n * pass_cost(ram, 1, "combo")["serial"] / cost_of_sizes(r["sizes"], ram, "combo", "serial")) for ram in RAMS},
                     "speedup_overlap": {ram: _r(n * pass_cost(ram, 1)["overlap"] / cost_of_sizes(r["sizes"], ram, "screen", "overlap")) for ram in RAMS}}
    un = {k: _r(sum(union_count(t, k) for t in TRACES) / 3, 2) for k in (1, 2, 3, 4, 8)}
    out = {"corpora": res, "union_experts_per_layer": un, "union_claimed": {k: cm.UNION.get(k) for k in (2, 3, 4, 8)},
           "bytes_per_token_vs_single_k8": _r(un[8] / (8 * un[1]), 3),
           "pass_cost_s": {ram: {k: _r(pass_cost(ram, k)["serial"], 4) for k in (1, 2, 4, 8)} for ram in RAMS},
           "screen_curve_100pct_accept": {str(f): {ram: _r(speedup_from_f(f, ram), 3) for ram in RAMS} for f in (0.15, 0.3, 0.6, 0.85)},
           "screen_curve_combo_batch": {str(f): {ram: _r(speedup_from_f(f, ram, "combo"), 3) for ram in RAMS} for f in (0.15, 0.3, 0.6, 0.85)}}
    out["unmeasured"] = ["acceptance on real model outputs (none saved here; text corpora are teacher-forced stand-ins)", "a real llama.cpp lookup/ngram run (no model allowed in this screen)", "expert counts for spans beyond 16 tokens on a 512-expert layer"]
    return out


# ================================================================== family 22: prefill while the user is typing
def common_prefix_len(a, b):
    n = min(len(a), len(b)); i = 0
    while i < n and a[i] == b[i]: i += 1
    return i


def typing_ttft(N, chunk, rate_tps, cost_fn):
    """Seconds from Enter to the end of prefill when chunk passes run while the user types `N` tokens at `rate_tps` (inf = paste).
    Chunk i is ready at i*chunk/rate, passes run one at a time; after Enter the unfinished passes and the tail (N mod chunk tokens) still run. cost_fn(tokens) = seconds of one pass."""
    full = N // chunk; t_enter = 0.0 if math.isinf(rate_tps) else N / rate_tps; fin = 0.0
    for i in range(1, full + 1):
        ready = 0.0 if math.isinf(rate_tps) else i * chunk / rate_tps; fin = max(ready, fin) + cost_fn(chunk)
    tail = N - full * chunk
    return max(0.0, fin - t_enter) + (cost_fn(tail) if tail else 0.0)


def per_token_hit(trace, fp):
    """LRU hit per decode token (cold start included), list of rates per token."""
    T = cs.tr(trace); acc = []
    c = collections.OrderedDict(); cap = int(fp * cs.FOOT_ENTRIES); out = []
    for tok in T:
        h = n = 0
        for l in range(LAYERS):
            for e in tok[l]:
                k = (l, e); n += 1
                if k in c: c.move_to_end(k); h += 1
                else:
                    c[k] = None
                    if len(c) > cap: c.popitem(last=False)
        out.append(h / n)
    return out


def fam_s2_22():
    tmax = min(len(cs.tr(t)) for t in TRACES); out = {"tokens_per_trace_min": tmax}
    out["unique_experts_64_window"] = {t.split("_")[1]: {"measured_of_128": _r(union_frac(t, 64) * E30, 1), "share": _r(union_frac(t, 64))} for t in TRACES}
    out["uniform_share_64_30B"] = _r(uniform_frac(64, E30, TOP30)); out["uniform_share_64_125B"] = _r(uniform_frac(64, E125, TOP125))
    f64 = frac_models(64, tmax); f500 = frac_models(500, tmax)
    out["premise_over_85pct_measured"] = f64["measured30"] > 0.85
    models = {}
    for m in ("measured30", "scaled125", "uniform125"):
        cf = lambda n, m=m: prefill_pass_s(n, frac_models(n, tmax)[m] if n else 0.0) if n else 0.0
        P64, P500 = cf(64), cf(500)
        sims = {("paste" if math.isinf(r) else str(r)): {"ttft_after_enter_s": _r(typing_ttft(500, 64, r, cf), 1), "ttft_gain_vs_single_pass": _r(P500 / max(1e-9, typing_ttft(500, 64, r, cf)), 3)} for r in (float("inf"), 5.0, 3.0, 2.0, 1.0)}
        models[m] = {"chunk64_pass_s": _r(P64, 1), "full500_pass_s": _r(P500, 1), "typing": sims, "disk_bytes_chunked_over_single": _r(7 * f64[m] / f500[m] + frac_models(52, tmax)[m] / f500[m], 2),
                     "fill_seconds_of_one_chunk_at_1_3_tok_s": [_r(64 / 1.0, 0), _r(64 / 3.0, 0)]}
    out["models"] = models
    ph = {t: per_token_hit(t, footprints(8)) for t in TRACES}
    first = sum(sum(v[:10]) / 10 for v in ph.values()) / 3; rest = sum(sum(v[50:]) / len(v[50:]) for v in ph.values()) / 3
    out["decode_cold_start_8gb"] = {"hit_first10_tokens": _r(first), "hit_after_50": _r(rest), "extra_io_s_first10_tokens": _r(10 * (rest - first) * EXP_GB / R_LOCAL, 1)}
    out["kill"] = {"chunk_pass_over_14s_scaled125": models["scaled125"]["chunk64_pass_s"] > 14.0, "paste_has_no_gain_scaled125": models["scaled125"]["typing"]["paste"]["ttft_gain_vs_single_pass"] <= 1.0}
    out["unmeasured"] = ["real prefill timing per chunk (needs a model run)", "keystroke edit patterns and tokenization at chunk boundaries", "decode-cache state after chunk passes (modelled only as the generic cold start)"]
    return out


# ================================================================== family 23: ahead-of-time specialised / fused kernels (dispatch overhead)
def amdahl(share): return 1.0 / (1.0 - share) if share < 1 else float("inf")


def fam_s2_23():
    out = {}
    out["flops"] = {"decode_s_per_token": _r(COMP_DEC, 4), "prefill500_s_at_55": _r(500 * FLOP_TOK / G_DEC, 1), "prefill500_s_at_311": _r(500 * FLOP_TOK / G_PRE, 1), "claimed_base_s": 109, "claimed_after_s": 45, "claimed_cut": _r(1 - 45 / 109, 3),
                    "tflop_in_109s_at_311": _r(311e9 * 109 / 1e12, 1), "tflop_500_tokens": _r(500 * FLOP_TOK / 1000, 1)}
    ops = {"graph_ops_per_token": 3 * LAYERS, "slice_matmuls_per_token": 3 * SL, "prefill_expert_matmuls": 3 * LAYERS * E125}
    out["ops"] = ops
    sh = {}
    for us in (2, 5, 10, 50):
        d_dec = ops["slice_matmuls_per_token"] * us * 1e-6; d_pre = ops["prefill_expert_matmuls"] * us * 1e-6
        sh[str(us)] = {"decode_ms": _r(d_dec * 1000, 2), "decode_share": _r(d_dec / COMP_DEC, 4), "decode_amdahl": _r(amdahl(d_dec / COMP_DEC), 4),
                       "prefill_s": _r(d_pre, 2), "prefill_share_of_19s": _r(d_pre / (500 * FLOP_TOK / G_PRE), 4), "prefill_amdahl": _r(amdahl(d_pre / (500 * FLOP_TOK / G_PRE)), 4)}
    out["dispatch_us_scenarios"] = sh
    cut = 0.6
    out["needed_for_claim"] = {"us_per_slice_matmul_decode": _r(cut * COMP_DEC / ops["slice_matmuls_per_token"] * 1e6, 1), "us_per_graph_op_decode": _r(cut * COMP_DEC / ops["graph_ops_per_token"] * 1e6, 1),
                               "us_per_expert_matmul_prefill_19s": _r(cut * 500 * FLOP_TOK / G_PRE / ops["prefill_expert_matmuls"] * 1e6, 1), "implied_rate_multiplier": _r(1 / (1 - cut), 2)}
    # decode is I/O-bound at 8 and 16 GB, so even a perfect compute fix cannot move it
    io = {}
    for ram in RAMS:
        h = mean_hit(footprints(ram)); io[ram] = {"hit": _r(h), "serial_tok_s": _r(serial_toks(h, R_LOCAL)), "serial_if_compute_x2": _r(serial_toks(h, R_LOCAL, COMP_DEC / 2)), "gain_x2_compute": _r(serial_toks(h, R_LOCAL, COMP_DEC / 2) / serial_toks(h, R_LOCAL), 3),
                                                  "io_s": _r((1 - h) * EXP_GB / R_LOCAL, 4)}
    out["decode_if_compute_doubled"] = io
    out["unmeasured"] = ["share of wall time outside the quantised matmul loops (needs a profiler run on a real build)", "thread scaling -t 1 vs -t 4 of prefill (optional helper olmoe_thread_scaling)", "any fused/AOT kernel (none built)"]
    return out


def layer_bound(tok_s, layers, target_layers=LAYERS, per_token_s=COMP_DEC):
    """Upper bound on the dispatch/overhead share of a decode token from a tiny model: its whole per-token time over its layers bounds the per-layer overhead (it also contains real compute,
    the output head and sampling); scaled to `target_layers` and divided by the target's per-token compute time."""
    per_layer = 1.0 / tok_s / layers; scaled = per_layer * target_layers; share = scaled / per_token_s
    return dict(per_layer_ms=per_layer * 1000, scaled_ms=scaled * 1000, share=share, amdahl=amdahl(min(share, 0.999)))


def tiny_decode_bound(model=None, threads=(1, 4)):
    """Optional host measurement for family 23: decode tok/s of the 2-layer tiny MoE and the resulting layer_bound per thread count (not deterministic; not run by default)."""
    model = model or os.environ.get("PUBTECH_TINY")
    if not model or not os.path.exists(model): return None
    layers = [v for k, v in read_gguf(model)["kv"].items() if k.endswith(".block_count") and isinstance(v, int) and v > 0][0]; out = {}
    for t in threads:
        rows = llama_bench(model, ["-p", "0", "-n", "128", "-t", str(t), "-r", "3"])
        if rows: out[t] = dict(tok_s=rows[0][3], **layer_bound(rows[0][3], layers))
    return out


def olmoe_thread_scaling(model=None):
    """Optional: prompt-processing tok/s at 1 and 4 threads; scaling near 4x means compute-bound, not dispatch-bound. Noisy on a shared machine."""
    rows = llama_bench(model or os.environ.get("PUBTECH_OLMOE"), ["-p", "512", "-n", "0", "-t", "1,4", "-r", "2"])
    if not rows: return None
    d = {r[2]: r[3] for r in rows}; return dict(tok_s_t1=d[1], tok_s_t4=d[4], scaling=d[4] / d[1])


# ================================================================== family 24: grammar-constrained jump-forward decoding
# schema nodes: ("obj", [(name, schema, required)]) (required properties first, optional ones last), ("str",), ("enum", [options]), ("int",), ("bool",), ("arr", item)
WORDS = ("data build plan test merge branch commit review file module cache index query result error retry window token layer expert memory disk read write check update remove "
         "create verify report summary config server client request response limit offset count sort filter order group value field table column row schema").split()
PLAN = ("obj", [("goal", ("str",), True), ("steps", ("arr", ("obj", [("id", ("int",), True), ("title", ("str",), True), ("tool", ("enum", ["search", "read_file", "run_tests", "edit_file", "summarize"]), True),
                                                              ("risk", ("enum", ["low", "medium", "high"]), True), ("notes", ("str",), False)])), True), ("done", ("bool",), True)])
VERDICT = ("obj", [("verdict", ("enum", ["approve", "reject", "revise"]), True), ("confidence", ("int",), True), ("issues", ("arr", ("obj", [("severity", ("enum", ["minor", "major", "blocker"]), True), ("message", ("str",), True)])), True),
                   ("summary", ("str",), True)])
TOOL = ("obj", [("name", ("enum", ["search_docs", "run_query", "fetch_url", "write_file"]), True), ("arguments", ("obj", [("query", ("str",), True), ("limit", ("int",), True), ("offset", ("int",), False)]), True)])
RECORD = ("obj", [("id", ("int",), True), ("title", ("str",), True), ("status", ("enum", ["open", "closed", "pending"]), True), ("tags", ("arr", ("str",)), True), ("assignee", ("str",), False), ("priority", ("int",), False)])
CLAIM = ("obj", [("claim", ("str",), True), ("supported", ("bool",), True), ("evidence", ("arr", ("str",)), True), ("score", ("int",), True)])
SCHEMAS = [("plan", PLAN), ("verdict", VERDICT), ("tool_call", TOOL), ("record", RECORD), ("claim", CLAIM)]


def gen_value(sc, rng):
    t = sc[0]
    if t == "str": return " ".join(rng.choice(WORDS) for _ in range(rng.randint(1, 9)))
    if t == "enum": return rng.choice(sc[1])
    if t == "int": return rng.randint(0, 999)
    if t == "bool": return rng.random() < 0.5
    if t == "arr": return [gen_value(sc[1], rng) for _ in range(rng.randint(1, 4))]
    d = {}
    for name, sub, req in sc[1]:
        if req or rng.random() < 0.5: d[name] = gen_value(sub, rng)
    return d


class Chars:
    """Output as characters, each with (forced?, kind). kind: S structural, K key text, V value text."""
    def __init__(self): self.c = []; self.free_next = False
    def lit(self, s, forced=True, kind="S"):
        for i, ch in enumerate(s):
            f = forced and not (i == 0 and self.free_next); self.c.append((ch, f, kind))
            if i == 0: self.free_next = False
    def val(self, s, flags, kind="V"):
        for ch, f in zip(s, flags): self.c.append((ch, f, kind))
        self.free_next = False


def enum_flags(options, chosen):
    """Per-character forcedness of `chosen` followed by its closing quote: a character is forced iff all options sharing the prefix agree on the next symbol (end of word counts as a symbol)."""
    fl = []
    for j in range(len(chosen) + 1):
        pre = chosen[:j]; nx = {o[j] if j < len(o) else None for o in options if o.startswith(pre)}; fl.append(len(nx) == 1)
    return fl


def emit(sc, v, o):
    t = sc[0]
    if t == "str": o.lit('"'); o.val(v, [False] * len(v)); o.lit('"', forced=False)
    elif t == "enum": fl = enum_flags(sc[1], v); o.lit('"'); o.val(v, fl[:-1]); o.lit('"', forced=fl[-1])
    elif t == "int": o.val(str(v), [False] * len(str(v))); o.free_next = True            # the next structural char competes with another digit
    elif t == "bool": s = "true" if v else "false"; o.val(s, [False] + [True] * (len(s) - 1))
    elif t == "arr":
        o.lit("[")
        for i, x in enumerate(v):
            if i: o.lit(",", forced=False)
            emit(sc[1], x, o)
        o.lit("]", forced=False)
    else:
        o.lit("{"); first = True
        for n, s, r in sc[1]:
            if n not in v: continue
            if first: o.lit('"', forced=r)                 # a first optional property competes with the closing brace
            else: o.lit(",", forced=r); o.lit('"')          # an optional property competes with the closing brace after the comma choice
            o.val(n, [True] * len(n), "K"); o.lit('":'); emit(s, v[n], o); first = False
        o.lit("}", forced=all(n in v for n, _, r in sc[1] if not r))                # forced unless an absent optional property could still follow


def tokenize_flagged(chars, merge, max_piece=8):
    """Segment the flagged characters into tokens. A token is forced iff all its characters are forced. merge=False: every structural character is its own token;
    merge=True: runs of up to 3 adjacent structural characters merge (BPE-like). Keys and values split into word pieces of at most max_piece characters."""
    toks = []; i = 0; n = len(chars)
    while i < n:
        ch, f, k = chars[i]
        if k == "S":
            j = i + 1
            if merge:
                while j < n and j - i < 3 and chars[j][2] == "S": j += 1
            toks.append(all(c[1] for c in chars[i:j])); i = j
        else:
            j = i
            while j < n and chars[j][2] == k: j += 1
            seg = chars[i:j]; txt = "".join(c[0] for c in seg); p = 0
            for m in re.finditer(r"\s?[A-Za-z]+|\d+|\s?[^A-Za-z\d\s]|\s", txt):
                a, b = m.start(), m.end()
                for s in range(a, b, max_piece): toks.append(all(c[1] for c in seg[s:min(b, s + max_piece)]))
            i = j
    return toks


def forced_runs(flags):
    """Lengths of maximal runs of forced tokens."""
    runs = []; c = 0
    for f in flags:
        if f: c += 1
        elif c: runs.append(c); c = 0
    if c: runs.append(c)
    return runs


def forward_sizes(flags):
    """Forward sizes with jump-forward: one forward per free token, processing the previous free token (or the prompt tail) plus the forced run before it. An EOS (free) token is appended."""
    flags = list(flags) + [False]; sizes = []; run = 0
    for f in flags:
        if f: run += 1
        else: sizes.append(1 + run); run = 0
    return sizes


def build_outputs(per_schema=4, seed=24):
    rng = random.Random(seed); outs = []
    for name, sc in SCHEMAS:
        for _ in range(per_schema):
            v = gen_value(sc, rng); o = Chars(); emit(sc, v, o); outs.append((name, v, o.c))
    return outs


def render(chars): return "".join(c[0] for c in chars)


def fam_s2_24():
    outs = build_outputs(); res = {}
    for label, merge in (("unmerged_punct", False), ("merged_punct", True)):
        allf = []; sizes = []; n_tok = 0; per = []
        for name, v, ch in outs:
            fl = tokenize_flagged(ch, merge); allf += fl; sizes += forward_sizes(fl); n_tok += len(fl) + 1; per.append(sum(fl) / (len(fl) + 1))
        runs = forced_runs(allf); hist = collections.Counter(runs)
        sp = {ram: _r(n_tok * pass_cost(ram, 1)["serial"] / cost_of_sizes(sizes, ram, "screen", "serial")) for ram in RAMS}
        spc = {ram: _r(n_tok * pass_cost(ram, 1, "combo")["serial"] / cost_of_sizes(sizes, ram, "combo", "serial")) for ram in RAMS}
        spo = {ram: _r(n_tok * pass_cost(ram, 1)["overlap"] / cost_of_sizes(sizes, ram, "screen", "overlap")) for ram in RAMS}
        res[label] = {"tokens": n_tok, "forced": sum(allf), "f": _r(sum(allf) / n_tok), "f_per_output_min": _r(min(per)), "f_per_output_max": _r(max(per)), "n_runs": len(runs), "mean_run": _r(sum(runs) / len(runs), 3),
                      "run_hist": {str(k): hist[k] for k in sorted(hist)}, "forwards": len(sizes), "forwards_per_token": _r(len(sizes) / n_tok, 4), "mean_forward_size": _r(sum(sizes) / len(sizes), 3),
                      "speedup_serial": sp, "speedup_serial_combo_batch": spc, "speedup_overlap": spo, "ideal_1_over_1_minus_f": _r(1 / (1 - sum(allf) / n_tok), 3)}
    out = {"outputs": len(outs), "by_variant": res, "claim": {"f": 0.7, "claimed_speedup": _r(1 / 0.3, 2)},
           "screen_bound_f07_serial": {ram: _r(speedup_from_f(0.7, ram), 3) for ram in RAMS}, "sample_output": render(outs[0][2])[:160],
           "kill": {"f_below_0.5_unmerged": res["unmerged_punct"]["f"] < 0.5, "f_below_0.5_merged": res["merged_punct"]["f"] < 0.5, "mean_run_below_3_merged": res["merged_punct"]["mean_run"] < 3}}
    out["unmeasured"] = ["real planner/verifier/tool-call outputs (private; 20 seeded synthetic schema outputs are used instead, so f depends on my value-length choices)", "the real tokenizer (a rule-based stand-in is used)", "different tokenization of forced text changing later logits"]
    return out


# ================================================================== family 25: 2-bit base plus 4-bit refinement tiles
BPW_FULL, BPW_BASE, BPW_REF = 4.3, 2.2, 4.0


def tile_model(h_base, ref_share, rate, ref_mode, comp=COMP_DEC, base_bpw=BPW_BASE):
    """Seconds and tok/s of the base+refinement layout. Base bytes per token at hit 0 = EXP_GB*2.2/4.3. ref_mode 'every_token': a share `ref_share` of the 480 slices reads its 4-bit tile every token
    (not cached); 'on_miss': the tile is read only for base misses (share of missed slices); 'none': pure 2.2 bpw."""
    base = (1 - h_base) * EXP_GB * base_bpw / BPW_FULL
    if ref_mode == "every_token": ref = ref_share * EXP_GB * BPW_REF / BPW_FULL
    elif ref_mode == "on_miss": ref = ref_share * (1 - h_base) * EXP_GB * BPW_REF / BPW_FULL
    else: ref = 0.0
    io = (base + ref) / rate
    return dict(bytes_gb=base + ref, serial=1 / (comp + io), overlap=1 / max(comp, io))


def hit_at_bpw(ram, bpw):
    """Mean LRU hit when slices shrink to `bpw` bits per weight (cache holds 4.3/bpw times more entries)."""
    return mean_hit(footprints(ram) * BPW_FULL / bpw)


def q2_groupwise(w, g=16):
    """Requantize a float list to 2 bits per weight with per-group min/scale (4 levels); returns the reconstruction."""
    out = []
    for i in range(0, len(w), g):
        c = w[i:i + g]; lo, hi = min(c), max(c); sc = (hi - lo) / 3.0 if hi > lo else 1.0
        out.extend([lo + round((x - lo) / sc) * sc if hi > lo else lo for x in c])
    return out


def top_share(v, p):
    """Share of the total of v held by its largest fraction p of entries."""
    s = sorted(v, reverse=True); n = max(1, int(round(len(s) * p))); return sum(s[:n]) / sum(s)


def residual_energy(vectors, g=32):
    """Requantize each vector to 2 bits (group-wise min/scale) and describe the residual: per-vector relative norm ||r||/||w|| (mean, coefficient of variation) and how much of the
    total residual energy sits in the top 10% of vectors (expert slices) and of g-weight groups. A uniform spread would put 10% there."""
    en = []; eres = []; gres = []
    for w in vectors:
        q = q2_groupwise(w); r = [a - b for a, b in zip(w, q)]; en.append(dot(w, w)); eres.append(dot(r, r))
        for i in range(0, len(w), g): gres.append(dot(r[i:i + g], r[i:i + g]))
    rel = [math.sqrt(a / b) for a, b in zip(eres, en)]; mean = sum(rel) / len(rel); sd = math.sqrt(sum((x - mean) ** 2 for x in rel) / len(rel))
    return dict(n_slices=len(vectors), rel_residual_mean=_r(mean, 5), rel_residual_cv=_r(sd / mean, 5), top10pct_slices_energy_share=_r(top_share(eres, 0.10), 4), top10pct_groups_energy_share=_r(top_share(gres, 0.10), 4), n_groups=len(gres))


def residual_stats(layer, experts, g=32):
    """residual_energy on sampled rows of the real gate matrices of `experts` in `layer` (None without the weights file)."""
    W = weights()
    if W is None or not W.ok(layer): return None
    return residual_energy([W.expert(layer, e) for e in experts], g)


def fam_s2_25():
    out = {"hit": {}, "tok_s": {}}
    for ram in RAMS:
        hb = {"b43": mean_hit(footprints(ram)), "b22": hit_at_bpw(ram, BPW_BASE)}; out["hit"][ram] = {k: _r(v) for k, v in hb.items()}; row = {}
        for rate_name, rate in (("2.4", R_SCREEN), ("3.11", R_LOCAL)):
            base = tile_model(hb["b43"], 0, rate, "none", base_bpw=BPW_FULL); row[rate_name] = {"lru_4.3bit": {"serial": _r(base["serial"]), "overlap": _r(base["overlap"])}}
            for nm, ms, ref in (("pure_2.2bit", "none", 0.0), ("tiles_every_token_10pct", "every_token", 0.10), ("tiles_on_miss_10pct", "on_miss", 0.10)):
                t = tile_model(hb["b22"], ref, rate, ms); row[rate_name][nm] = {"serial": _r(t["serial"]), "overlap": _r(t["overlap"]), "gain_serial": _r(t["serial"] / base["serial"], 3), "gain_overlap": _r(t["overlap"] / base["overlap"], 3), "gb_per_token": _r(t["bytes_gb"], 3)}
        out["tok_s"][ram] = row
    out["file_size_gb_6bpw"] = _r(69 * 6.0 / 4.3, 1); out["cap_ratio"] = _r(BPW_FULL / BPW_BASE, 3)
    perm = random.Random(25).sample(range(E30), E30)
    out["residual_layer10"] = residual_stats(10, perm[:16]); out["residual_layer30"] = residual_stats(30, perm[:16])
    out["unmeasured"] = ["perplexity or KL of a 2-bit base (OLMoE Q2_K +26% is the only number, requantised so an upper bound)", "the per-token activation test that would trigger refinement", "weight statistics beyond 16 experts x 16 rows of the gate matrix in layers 10 and 30 of the 30B", "real tiled storage layout I/O"]
    return out


FAMILIES = {
    "s2_18": {"idea": "Expert store in a peer machine's RAM, fetched over RDMA (about 11 GB/s) instead of local NVMe", "run": fam_s2_18, "needs": "hardware",
              "methods": ["llama.cpp RPC backend (rpc-server holds weights in remote RAM)", "MegaScale-Infer, disaggregated expert parallelism", "llama.cpp issue 19745, RPC large-model load hang"]},
    "s2_19": {"idea": "Store expert slices as quantized residuals against the expert the previous layer's router predicts", "run": fam_s2_19, "needs": "local",
              "methods": ["D2-MoE: Delta Decompression for MoE-based LLMs Compression (ICML 2025)", "DeRS: Towards Extremely Efficient Upcycled MoE Models", "Breaking the MoE LLM Trilemma: Dynamic Expert Clustering with Structured Compression",
                          "Shape Mutating Expert Compression: LorExperts and BTExperts", "Improving Parameter Utilization by Sharing Neural Experts Across Layers", "DIMoE-Adapters"]},
    "s2_20": {"idea": "Emit provisional tokens after the first 64-token prompt block while the rest of the prompt prefills", "run": fam_s2_20, "needs": "model",
              "methods": ["SpecPrefill (ICML 2025), a small draft model picks the important prompt tokens and only those go to the big model", "Cross-Family Speculative Prefill, ICLR 2026",
                          "Sarathi-Serve chunked prefill with stall-free batching, OSDI 2024", "Layered prefill for MoE serving"]},
    "s2_21": {"idea": "Prompt-local suffix-automaton / n-gram drafting verified in one batched pass", "run": fam_s2_21, "needs": "local",
              "methods": ["Prompt Lookup Decoding (apoorvumang) and llama.cpp examples/lookup (PR #4484), now --spec-type ngram-cache/ngram-simple/ngram-map", "SAM Decoding: Speculative Decoding via Suffix Automaton (ACL 2025)",
                          "Speculative Decoding: Performance or Illusion? (arXiv 2601.11580)", "MoE-Spec / EVICT / AcceptMoE / SpecMoEOff (MoE speculative decoding)", "N-Gram Trie Speculative Decoding (EMNLP 2025)"]},
    "s2_22": {"idea": "Chunked prefill of the partial prompt while the user is still typing, reusing the exact causal-prefix KV", "run": fam_s2_22, "needs": "model",
              "methods": ["llama.cpp same-slot prefix KV reuse (server cache_prompt, LCP slot choice)", "llama.cpp host-RAM prompt cache, --cache-ram", "Mini-SGLang chunked prefill with radix cache", "Sarathi-Serve (OSDI 2024), chunked prefill (secondary source)"]},
    "s2_23": {"idea": "Ahead-of-time specialised, fused kernels to remove dispatch overhead from decode and prefill", "run": fam_s2_23, "needs": "model",
              "methods": ["TaxBreak: overhead decomposition of LLM inference", "Fused-kernel gains on GPU (search snippet; the Mixtral and OLMoE figures come from separate sources that I did not open)", "ik_llama.cpp", "KTransformers (SOSP 25)", "MobileMoE (search snippet)", "ASAP (NPU MoE prefill)"]},
    "s2_24": {"idea": "Grammar/schema-constrained decoding with jump-forward tokens (forced tokens ride along in the next forward)", "run": fam_s2_24, "needs": "local",
              "methods": ["LMSYS compressed FSM / jump-forward decoding (SGLang)", "SGLang paper", "XGrammar paper", "llguidance fast-forward tokens (used in llama.cpp via PR 10224)", "vLLM structured decoding note (BentoML)", "Reviews of constrained-decoding speedups (DOMINO, Formatron arXiv 2506.01151)"]},
    "s2_25": {"idea": "2-bit base plus 4-bit refinement tiles read only for slices whose residual is large", "run": fam_s2_25, "needs": "local",
              "methods": ["DynaExq (2511.15015)", "Any-Precision LLM (2402.10517), github.com/SNU-ARK/any-precision-llm style bit-plane layout", "AnyBCQ (2510.10467)"]},
}

if __name__ == "__main__":
    which = sys.argv[1:] or list(FAMILIES)
    for k in which: print(k, json.dumps(FAMILIES[k]["run"](), indent=1, default=str))
