#!/usr/bin/env python3
"""Published-technique family tests, group A (brainstorm screen 1, ideas #0..#7).
Every family re-implements the decisive free test of its idea on the real Qwen3-30B router traces (data/traces/*.txt):
top-8 of 128 experts, 48 layers, ~250-300 decode tokens each. The 30B traces stand in for the ~125B target (top-10 of 512, 48 layers):
bytes are mapped by token footprints (one token = 384 trace accesses = 480 slices of 2.7 MB = 1.296 GB on the target).
Throughput constants are the ones the earlier screens used (explicit 4 MiB reads 2.4 GB/s, decode 55 GFLOP/s, 12 GFLOP per token, prefill 311 GFLOP/s).
Nothing here runs a model: every number is computed from the traces or is closed-form arithmetic. Parts that need a model or hardware are listed in each result under 'unmeasured'.
usage: python3 tools/pubtech/f_A.py [fam_s1_0 ...]   (prints the result dicts as JSON)"""
import collections, itertools, json, math, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import trace_tools as tt, cachesim as cs
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
TRACES = ["qwen3-30b_db", "qwen3-30b_story", "qwen3-30b_code"]
NL, TOP, FOOT = 48, 8, 384                       # layers, experts per token, accesses of one token in the trace
SLICES_125, SLICE_B = 480, 2.7e6                 # target: 48 layers x top-10 slices of ~2.7 MB (4.3 bpw)
FP_GB = SLICES_125 * SLICE_B / 1e9               # 1.296 GB read by one uncached token
BW = 2.4e9                                       # bytes/s, explicit 4 MiB reads (screens)
G_DEC, G_PRE, FLOP_TOK = 55e9, 311e9, 12e9       # FLOP/s decode, FLOP/s prefill, FLOP per token
COMP_S = FLOP_TOK / G_DEC                        # 0.2182 s per token => 4.58 tok/s compute ceiling
GIB = 1.073741824e9
CACHE_GIB = {8: 2.5, 16: 9.15, 32: 25.15}        # expert cache on an 8/16/32 GB machine
EXPERT_SHARE = 0.404                             # expert share of per-token FLOPs: 1.3 GB at 4.3 bpw = 2.42 B params of 6 B active
EXPERT_FILE_GB = 69.0                            # the idea's figure for the 125B expert file at 4.3 bpw
_MEMO = {}
_TOK = {}

def clear_cache():
    _MEMO.clear()

def mean(xs):
    xs = list(xs); return sum(xs) / len(xs) if xs else 0.0

def tokens(name):
    """tokens[t][layer] = list of the 8 expert ids in trace (rank) order."""
    if name not in _TOK:
        with open(os.path.join(ROOT, "data", "traces", name + ".txt")) as f: toks, _ = tt.parse(f)
        _TOK[name] = [[list(t[l]) for l in range(NL)] for t in toks]
    return _TOK[name]

def cache_footprints(gib, bpw=4.3):
    """Footprints of the target model that fit in `gib` GiB of expert cache at `bpw` bits per weight."""
    return gib * GIB / (SLICES_125 * SLICE_B * bpw / 4.3)

def expected_accepted(k, a):
    """Tokens consumed per verify block of k positions when position j is accepted with P(n>=j)=a^(j-1) (the screens' definition)."""
    return float(k) if a >= 1 else (1 - a ** k) / (1 - a)

def io_s(miss_fp, gb=FP_GB):
    """Seconds of reads for `miss_fp` footprints of misses per token."""
    return miss_fp * gb * 1e9 / BW

def tps_serial(io, comp=COMP_S): return 1.0 / (io + comp)
def tps_overlap(io, comp=COMP_S): return 1.0 / max(io, comp)

def breakeven_gflops(k, a, mpa_fp, base_fp):
    """Batch compute rate (GFLOP/s) at which a k-position verify block ties plain cached decode (serial model).
    Per accepted token: io(mpa) + (k/E) * F / G == io(base) + F / G_DEC. None = the I/O alone already loses."""
    den = io_s(base_fp) + COMP_S - io_s(mpa_fp)
    if den <= 0: return None
    return (k / expected_accepted(k, a)) * FLOP_TOK / den / 1e9

class LRU:
    """LRU over (layer, expert); global, or one independent LRU per layer (cap // 48 each). cap <= 0 never hits."""
    def __init__(self, cap, per_layer=False):
        self.zero = cap <= 0; self.pl = per_layer
        if per_layer: self.cap = max(1, int(cap) // NL); self.c = [collections.OrderedDict() for _ in range(NL)]
        else: self.cap = max(1, int(cap)); self.c = collections.OrderedDict()
    def touch(self, l, e):
        if self.zero: return False
        c = self.c[l] if self.pl else self.c; key = e if self.pl else (l, e)
        if key in c: c.move_to_end(key); return True
        c[key] = None
        if len(c) > self.cap: c.popitem(last=False)
        return False

def union_layer(per_token_ids):
    """Distinct ids of several tokens' picks, first-seen order."""
    return list(dict.fromkeys(itertools.chain.from_iterable(per_token_ids)))

def spec_sim(name, k, a, fp, per_layer=False, mode="oracle", seed=0):
    """Block (speculative / grouped) decoding on a real trace. Each block loads, layer by layer, the union of the experts of k positions through a shared LRU of fp footprints.
    a >= 1: all k accepted (pure grouping). a < 1: the accepted prefix n has P(n>=j)=a^(j-1) (n>=1) and the next block starts after it.
    mode 'oracle': the k positions are the true next k tokens (optimistic: wrong drafts load the experts of tokens that really come next).
    mode 'wrong' : only the n accepted positions are true; the other k-n draft positions load the experts of random other tokens of the same trace (pessimistic).
    Returns dict(misses, accepted, blocks, accesses, mpa) with mpa = misses per accepted token."""
    key = ("spec", name, k, round(a, 6), round(fp, 6), per_layer, mode, seed)
    if key in _MEMO: return _MEMO[key]
    T = tokens(name); n = len(T); cache = LRU(int(fp * FOOT), per_layer); ra = random.Random(seed); rw = random.Random(seed + 7919)
    pos = 0; mis = acc = blocks = accepted = 0; fast = not per_layer and not cache.zero; gc = cache.c; gcap = cache.cap
    while pos < n:
        kk = min(k, n - pos); m = kk
        if a < 1:
            m = 1
            while m < kk and ra.random() < a: m += 1
        if mode == "wrong": wrong = [T[rw.randrange(n)] for _ in range(kk - m)]
        for l in range(NL):
            if mode == "oracle": ids = [T[pos + j][l] for j in range(kk)]
            else: ids = [T[pos + j][l] for j in range(m)] + [w[l] for w in wrong]
            u = ids[0] if len(ids) == 1 else union_layer(ids)
            if fast:
                for e in u:
                    key = (l, e)
                    if key in gc: gc.move_to_end(key)
                    else:
                        mis += 1; gc[key] = None
                        if len(gc) > gcap: gc.popitem(last=False)
                acc += len(u)
            else:
                for e in u:
                    acc += 1
                    if not cache.touch(l, e): mis += 1
        accepted += m; blocks += 1; pos += m if a < 1 else kk
    r = dict(misses=mis, accepted=accepted, blocks=blocks, accesses=acc, mpa=mis / accepted); _MEMO[key] = r; return r

def union_stats(name, k):
    """Mean unique experts per layer over aligned blocks of k consecutive tokens (of 8k picks)."""
    T = tokens(name); tot = cnt = 0
    for p in range(0, len(T) - k + 1, k):
        for l in range(NL): tot += len(union_layer([T[p + j][l] for j in range(k)])); cnt += 1
    return tot / cnt

def uniform_union_fraction(E, K, k):
    """Union / (K*k) for k tokens that each pick K of E experts uniformly and independently (an upper bound: real routing is temporally correlated)."""
    return E * (1 - (1 - K / E) ** k) / (K * k)

def _tr(quick): return TRACES[:1] if quick else TRACES

def _spec_ratio_table(traces, ks, alphas, fps, per_layer, modes_fn):
    """ratio[key] = mean over traces of (misses per accepted token at block size k) / (misses per token at k=1), same LRU size."""
    out = {}
    for fp in fps:
        base = {t: spec_sim(t, 1, 1.0, fp, per_layer)["mpa"] for t in traces}
        for k in ks:
            for a in alphas:
                for mode in (("oracle", "wrong") if a < 1 else ("oracle",)):
                    out[f"fp{fp}|k{k}|a{a}|{mode}"] = mean(spec_sim(t, k, a, fp, per_layer, mode)["mpa"] / base[t] for t in traces)
    return out

# --------------------------------------------------------------------------------------------------------------------------------
def fam_s1_0(quick=False):
    """k=8 lookahead/draft tree expert-major: union bytes vs the LRU path. Decisive test: bytes per accepted token relative to k=1 LRU."""
    tr = _tr(quick); ks = (4, 8) if quick else (2, 4, 8); alphas = (0.9, 1.0) if quick else (0.6, 0.8, 0.9, 1.0); fps = (2.1, 4.0)
    ratio = _spec_ratio_table(tr, ks, alphas, fps, True, None)
    union = {f"k{k}": mean(union_stats(t, k) for t in tr) for k in (2, 4, 8)}
    uf = {kk: v / (TOP * int(kk[1:])) for kk, v in union.items()}
    gb = {}
    for a in (0.5, 0.7, 0.8, 0.9, 1.0): gb[f"a{a}"] = uf["k8"] * 8 * FP_GB / expected_accepted(8, a)
    be = next((round(i / 100, 2) for i in range(1, 101) if uf["k8"] * 8 * FP_GB / expected_accepted(8, i / 100 if i < 100 else 1.0) < FP_GB), None)
    low = {k_: v for k_, v in ratio.items() if "|a1.0|" not in k_}
    cells = {k_: v for k_, v in ratio.items()}
    best = min(cells, key=cells.get)
    # compute side (closed form): verify block of 8 at the prefill rate, a=0.8
    E8 = expected_accepted(8, 0.8)
    return dict(family="screen 1 #0", per_layer_lru=True, union_per_layer=union, union_fraction=uf, uncached_gb_per_accepted_k8=gb, uncached_break_even_alpha_k8=be,
                lru_relative_bytes=ratio, lru_relative_bytes_min=cells[best], lru_relative_bytes_min_cell=best,
                lru_relative_bytes_min_alpha_le_0p9=min(low.values()), n_cells=len(cells), n_cells_below_1=sum(1 for v in cells.values() if v < 1.0),
                compute_ceiling_tps_k8_a0p8_at_311=E8 / (8 * FLOP_TOK / G_PRE), decode_ceiling_tps=1 / COMP_S,
                paper_claims=dict(moe_spec_throughput="1.10-1.30x (budgeted experts, GPU offload)", ecospec="up to 1.62x", moe_speq="up to 2.34x (GPU/PCIe offload)"),
                unmeasured=["batch-8 verify GFLOP/s on the CPU (needs hardware)", "real draft acceptance for a 125B target", "wrong-branch experts are modelled by random same-trace tokens, not by a real drafter"])

def fam_s1_1(quick=False):
    """Decode k tokens of one sequence as a group: union of experts. Decisive test: misses per accepted token vs k=1 LRU over cap and acceptance."""
    tr = _tr(quick); ks = (1, 4, 8) if quick else (1, 2, 4, 8); fps = (2.1, 4.0) if quick else (2.1, 3.0, 4.0, 8.0); alphas = (0.7,) if quick else (0.5, 0.7, 0.9)
    oracle = {}; drafted = {}
    for fp in fps:
        for k in ks:
            oracle[f"fp{fp}|k{k}"] = mean(spec_sim(t, k, 1.0, fp)["mpa"] for t in tr) / FOOT
            if k > 1 and fp in (2.1, 4.0):
                for a in alphas: drafted[f"fp{fp}|k{k}|a{a}"] = mean(spec_sim(t, k, a, fp, False, "wrong")["mpa"] for t in tr) / FOOT
    cold = {f"k{k}": mean(spec_sim(t, k, 1.0, 0)["mpa"] for t in tr) / FOOT for k in ks}
    gain_o = {c: oracle[f"fp{c.split('|')[0][2:]}|k1"] / v for c, v in oracle.items() if not c.endswith("|k1")}
    gain_d = {c: oracle[f"{c.split('|')[0]}|k1"] / v for c, v in drafted.items()}
    return dict(family="screen 1 #1", global_lru=True, misses_per_token_in_footprints_oracle=oracle, misses_per_accepted_in_footprints_drafted=drafted, cold_union_fraction=cold,
                gain_vs_k1_oracle=gain_o, gain_vs_k1_drafted=gain_d, best_gain_oracle=max(gain_o.values()), best_gain_drafted=max(gain_d.values()),
                best_gain_oracle_cell=max(gain_o, key=gain_o.get), kill_threshold=1.3, killed=max(max(gain_o.values()), max(gain_d.values())) < 1.3,
                unmeasured=["accepted length of a real draft (needs a draft model run)", "real llama-speculative miss bytes on OLMoE (needs model)"])

def fam_s1_2(quick=False):
    """Prefill as expert-major work. Local part: window-union proxy on decode traces, token-major vs layer-major bytes, TTFT arithmetic. The real prefill trace needs a patched OLMoE run."""
    tr = _tr(quick); Ws = (1, 16, 64, 128) if quick else (1, 8, 16, 32, 64, 128, 249)
    union = {}
    for W in Ws:
        vals = []
        for t in tr:
            T = tokens(t); fr = []
            for p in range(0, len(T) - W + 1, W):
                for l in range(NL): fr.append(len(union_layer([T[p + j][l] for j in range(W)])) / 128.0)
            vals.append(mean(fr))
        union[f"W{W}"] = mean(vals)
    # iid estimate from the per-layer marginals (does not know about correlation; unseen experts get probability 0)
    def iid(t, W):
        T = tokens(t); fr = []
        for l in range(NL):
            c = collections.Counter(e for tok in T for e in tok[l]); fr.append(sum(1 - (1 - v / len(T)) ** W for v in c.values()) / 128.0)
        return mean(fr)
    iid_est = {f"W{W}": mean(iid(t, W) for t in tr) for W in (128, 500)}
    uni = {"qwen3_128_top8_W128": 1 - (1 - 8 / 128) ** 128, "target_512_top10_W500": 1 - (1 - 10 / 512) ** 500}
    full_gb = 512 * 48 * SLICE_B / 1e9
    comp_s = 500 * FLOP_TOK / G_PRE; ttft = {}
    for f in (0.5, 0.6, 0.75, 0.95, 1.0):
        by = f * full_gb; mm = by / 0.34; ov = max(comp_s, by / 2.4); sr = comp_s + by / 2.4
        ov156 = max(comp_s, by / 1.56); ttft[f"union{f}"] = dict(read_gb=by, mmap_s=mm, explicit_overlap_s=ov, explicit_serial_s=sr, gain_vs_mmap_overlap=mm / ov, gain_vs_mmap_serial=mm / sr, gain_vs_mmap_overlap_at_1p56=mm / ov156)
    lm = {}
    for W in ((64,) if quick else (64, 128)):
        for fp in ((2.07,) if quick else (2.07, 7.58, 20.84)):
            tm = mean(spec_sim(t, 1, 1.0, fp)["mpa"] for t in tr); lmv = mean(spec_sim(t, W, 1.0, fp)["mpa"] for t in tr)
            lm[f"W{W}|fp{fp}"] = dict(token_major_lru_misses_per_token=tm, layer_major_misses_per_token=lmv, ratio=tm / lmv)
    return dict(family="screen 1 #2", window_union_fraction=union, iid_marginal_union_fraction=iid_est, uniform_routing_union_fraction=uni, expert_file_gb=full_gb,
                prefill_compute_s_500=comp_s, uncached_token_major_gb_500=500 * FP_GB, ttft=ttft, token_vs_layer_major=lm,
                gain_vs_best_path="1.0x by construction: the best path is also layer-major and reads each layer's union once; not measured",
                unmeasured=["union of a real 500-token prefill on OLMoE (needs a patched trace build and a model run)", "mmap vs explicit-read prefill wall time (needs a capped model run)", "the 125B union fraction (0.75-0.95 is the screen's assumption)"])

def _stream(name, K):
    return [(l, e) for t in tokens(name) for l in range(NL) for e in t[l][:K]]

def _kcut_misses(name, K, cap, per_layer=False):
    s = _stream(name, K); n = len(tokens(name))
    h = cs.per_layer_lru_hits(s, max(1, cap // NL)) if per_layer else cs.lru_hits(s, cap)
    return (len(s) - h) / n

def fam_s1_3(quick=False):
    """Cut routed experts per token (6-of-10). Decisive free test: LRU misses for K=6 vs K=8 at equal cache entries; compute side by Amdahl on the expert share."""
    tr = _tr(quick); Ks = (8, 6) if quick else (8, 6, 5)
    pl = {f"K{K}": {t[-5:]: _kcut_misses(t, K, 24 * NL, True) for t in tr} for K in Ks}
    pl_gain = {f"K{K}": {t[-5:]: pl["K8"][t[-5:]] / pl[f"K{K}"][t[-5:]] for t in tr} for K in Ks if K != 8}
    rams = (8,) if quick else (8, 16, 32); gl = {}; rows = {}
    for r in rams:
        cap = int(cache_footprints(CACHE_GIB[r]) * FOOT)
        for K in Ks: gl[f"{r}GB|K{K}"] = mean(_kcut_misses(t, K, cap) for t in tr)
    for r in rams:
        for K, keep in ((6, 0.75), (5, 0.6)):
            if K not in Ks: continue
            io0 = io_s(gl[f"{r}GB|K8"] / FOOT); io1 = io_s(gl[f"{r}GB|K{K}"] / FOOT); c1 = COMP_S * (1 - EXPERT_SHARE * (1 - keep))
            rows[f"{r}GB|K{K}"] = dict(miss_gain=gl[f"{r}GB|K8"] / gl[f"{r}GB|K{K}"], serial_gain=tps_serial(io1, c1) / tps_serial(io0), overlap_gain=tps_overlap(io1, c1) / tps_overlap(io0))
    amdahl = {"6of10": 1 / (1 - EXPERT_SHARE * 0.4), "6of8": 1 / (1 - EXPERT_SHARE * 0.25), "4of10": 1 / (1 - EXPERT_SHARE * 0.6)}
    proj = {}
    if not quick:
        import combo_trace_sim as cts
        for r in (8, 16, 32):
            m = {sel: mean(cts.simulate(t, "lru", sel, 4.3, r, 0, 0.0)["misses"] for t in tr) for sel in ("full", "top6", "aware6")}
            proj[f"{r}GB"] = dict(top6_vs_full=m["full"] / m["top6"], aware6_vs_top6=m["top6"] / m["aware6"], aware6_vs_full=m["full"] / m["aware6"])
        ppl = cts.PPL["sel"]
    else: ppl = {}
    return dict(family="screen 1 #3", per_layer_lru_24_entries_misses_per_token=pl, per_layer_miss_gain_vs_K8=pl_gain, global_lru_misses_per_token=gl, throughput_rows=rows,
                amdahl_compute_ceiling_gain=amdahl, expert_share_of_flops=EXPERT_SHARE, compute_ceiling_tps=1 / COMP_S, project_simulator_cross_check=proj,
                earlier_olmoe_relative_ppl_loss=ppl, quality="not re-measured here; the earlier OLMoE perplexity loss is quoted from the repo's combination simulator",
                unmeasured=["perplexity of K=6 on OLMoE or the 125B (needs a model run)", "trace order is assumed to be gate-weight order, so ids[:K] are the K largest-weight experts", "measured compute gain of a lower top-k (needs hardware)"])

def fam_s1_4(quick=False):
    """k same-sequence positions expert-major. Decisive free test: miss bytes per token at k=1/2/4/8 for cap 0/2/4 footprints, plus the batch-compute break-even."""
    tr = _tr(quick); ks = (1, 4, 8) if quick else (1, 2, 4, 8); fps = (0, 2, 4)
    miss = {f"fp{fp}|k{k}": mean(spec_sim(t, k, 1.0, fp)["mpa"] for t in tr) / FOOT for fp in fps for k in ks}
    rel = {c: miss[c] / miss[f"{c.split('|')[0]}|k1"] for c in miss}
    drafted = {f"fp{fp}|k{k}|a0.7": mean(spec_sim(t, k, 0.7, fp, False, "wrong")["mpa"] for t in tr) / FOOT for fp in ((2,) if quick else (2, 4)) for k in ks if k > 1}
    drel = {c: v / miss[f"{c.split('|')[0]}|k1"] for c, v in drafted.items()}
    be = {}
    for r in ((8,) if quick else (8, 16, 32)):
        fp = cache_footprints(CACHE_GIB[r]); base = mean(spec_sim(t, 1, 1.0, fp)["mpa"] for t in tr) / FOOT
        for k in ((4,) if quick else (2, 4, 8)):
            for a in ((0.85,) if quick else (0.7, 0.85)):
                mpa = mean(spec_sim(t, k, a, fp, False, "wrong")["mpa"] for t in tr) / FOOT
                be[f"{r}GB|k{k}|a{a}"] = dict(misses_per_accepted_fp=mpa, base_fp=base, io_ratio=mpa / base, breakeven_gflops=breakeven_gflops(k, a, mpa, base), decode_gflops=G_DEC / 1e9, prefill_gflops=G_PRE / 1e9)
    uni = {"qwen3_128_top8_k8": uniform_union_fraction(128, 8, 8), "target_512_top10_k8": uniform_union_fraction(512, 10, 8)}
    return dict(family="screen 1 #4", global_lru=True, miss_fraction_of_full_token_read=miss, relative_to_k1=rel, drafted_a0p7_fraction=drafted, drafted_relative_to_k1=drel,
                union_fraction_k8_measured=mean(union_stats(t, 8) for t in tr) / 64, uniform_union_fraction=uni, batch_compute_breakeven=be, cold_gain_k8=1 / rel["fp0|k8"],
                unmeasured=["batched-matmul GFLOP/s at n=2..8 on the CPU (needs hardware)", "acceptance of prompt-lookup drafts on OLMoE (needs a model run)", "union fraction of the 125B router (512 experts, top-10); only the uniform-routing bound is computed"])

def fam_s1_5(quick=False):
    """Small resident draft + union verify (+ confidence gate). Decisive free test: unique experts per block and bytes per accepted token vs the cached k=1 baseline."""
    tr = _tr(quick); ks = (2, 4) if quick else (2, 3, 4, 8); alphas = (0.7,) if quick else (0.5, 0.7, 0.85); fps = (2.1,) if quick else (2.1, 3.0)
    uniq = {f"k{k}": mean(union_stats(t, k) for t in tr) for k in ks}
    unc = {}
    for k in ks:
        gb_blk = uniq[f"k{k}"] / (TOP * k) * k * FP_GB
        for a in alphas:
            E = expected_accepted(k, a); E1 = (1 - a ** (k + 1)) / (1 - a)     # E1: the screen's other reading, k drafts plus the bonus token
            t_blk = gb_blk * 1e9 / BW + k * FLOP_TOK / G_PRE
            unc[f"k{k}|a{a}"] = dict(gb_per_block=gb_blk, tps=E / t_blk, tps_with_bonus_token=E1 / t_blk, gb_per_accepted=gb_blk / E)
    plain = 1 / (FP_GB * 1e9 / BW + COMP_S)
    cached = {}
    for fp in fps:
        base = mean(spec_sim(t, 1, 1.0, fp)["mpa"] for t in tr) / FOOT; btps = tps_serial(io_s(base))
        for k in ks:
            for a in (alphas if fp == fps[0] else (0.7,)):
                for mode in ("oracle", "wrong"):
                    mpa = mean(spec_sim(t, k, a, fp, False, mode)["mpa"] for t in tr) / FOOT; E = expected_accepted(k, a)
                    tps = 1 / (io_s(mpa) + (k / E) * FLOP_TOK / G_PRE)
                    cached[f"fp{fp}|k{k}|a{a}|{mode}"] = dict(gb_per_accepted=mpa * FP_GB, bytes_gain_vs_k1=base / mpa, tps=tps, tps_gain_vs_k1=tps / btps)
    gains07 = [v["bytes_gain_vs_k1"] for c_, v in cached.items() if "|a0.7|" in c_]
    gains07_or = [v["bytes_gain_vs_k1"] for c_, v in cached.items() if "|a0.7|" in c_ and c_.endswith("oracle")]
    tps07 = {c_: v["tps_gain_vs_k1"] for c_, v in cached.items() if "|a0.7|" in c_ and "fp2.1" in c_}
    cov = 0.6; hit_fp = mean(spec_sim(t, 1, 1.0, 2.1)["mpa"] for t in tr) / FOOT
    gate = dict(coverage=cov, compute_only_ceiling_tps=1 / ((1 - cov) * COMP_S), uncached_serial_tps=1 / ((1 - cov) * (FP_GB * 1e9 / BW + COMP_S)),
                lru2p1_serial_tps=1 / ((1 - cov) * (io_s(hit_fp) + COMP_S)), note="tokens taken from the draft are unverified, so quality is the draft's on that share")
    return dict(family="screen 1 #5", unique_per_layer=uniq, uncached=unc, plain_uncached_tps=plain, cached=cached, best_bytes_gain_a0p7=max(gains07), best_bytes_gain_a0p7_oracle=max(gains07_or), tps_gain_a0p7_fp2p1=tps07, tps_assumes_batch_gflops=G_PRE / 1e9,
                kill_threshold=1.3, gate=gate,
                unmeasured=["real draft acceptance against the big model (needs draft + big model)", "draft agreement vs confidence for the gate (needs models)", "RAM and compute cost of the resident draft model", "KV backfill cost of skipped positions"])

def stick_tokens(name, f, seed=0):
    """Temporal-stickiness router: of each token's picks that were not in the previous (already modified) token, a fraction f is swapped for previous-token experts that the token did not pick."""
    T = tokens(name); rng = random.Random(seed); out = [[list(x) for x in T[0]]]
    for t in range(1, len(T)):
        row = []
        for l in range(NL):
            prev = out[t - 1][l]; cur = list(T[t][l]); ps = set(prev)
            new_idx = [i for i, e in enumerate(cur) if e not in ps]; avail = [e for e in prev if e not in set(cur)]
            n = min(round(f * len(new_idx)), len(avail))
            for i, e in zip(sorted(rng.sample(new_idx, n)), avail[:n]): cur[i] = e
            row.append(cur)
        out.append(row)
    return out

def _hit_for(toks_, fp):
    s = [(l, e) for t in toks_ for l in range(NL) for e in t[l]]
    return cs.lru_hits(s, int(fp * FOOT)) / len(s)

def _persist(toks_):
    return mean(len(set(toks_[t][l]) & set(toks_[t - 1][l])) / TOP for t in range(1, len(toks_)) for l in range(NL))

def fam_s1_6(quick=False):
    """Router fine-tuned for temporal stickiness. Decisive free test: replace a fraction f of new picks by the previous token's experts and replay LRU at 1.5/2/3 footprints."""
    tr = _tr(quick); fs = (0.0, 0.5) if quick else (0.0, 0.25, 0.5, 0.75, 1.0); fps = (2.0,) if quick else (1.5, 2.0, 3.0)
    hit = {}; per = {}; sw = {}
    for f in fs:
        mods = {t: stick_tokens(t, f) for t in tr}
        per[f"f{f}"] = mean(_persist(mods[t]) for t in tr)
        for fp in fps: hit[f"f{f}|fp{fp}"] = mean(_hit_for(mods[t], fp) for t in tr)
        for W in ((64,) if quick else (16, 64, 128)):
            sw[f"f{f}|W{W}"] = mean(len(union_layer([mods[t][p + j][l] for j in range(W)])) / 128.0 for t in tr for p in range(0, len(mods[t]) - W + 1, W) for l in range(NL))
    gain = {}
    for f in fs:
        for fp in fps:
            h0 = hit[f"f0.0|fp{fp}"]; h = hit[f"f{f}|fp{fp}"]; io0 = io_s(1 - h0); io1 = io_s(1 - h)
            gain[f"f{f}|fp{fp}"] = dict(hit=h, serial_tps=tps_serial(io1), overlap_tps=tps_overlap(io1), serial_gain=tps_serial(io1) / tps_serial(io0), overlap_gain=tps_overlap(io1) / tps_overlap(io0))
    return dict(family="screen 1 #6", hit=hit, token_overlap=per, window_union_fraction=sw, throughput=gain, compute_ceiling_tps=1 / COMP_S, claimed_decode_tps_range=(5, 8),
                claimed_range_above_ceiling=5 > 1 / COMP_S, prefill_compute_s_500=500 * FLOP_TOK / G_PRE,
                unmeasured=["perplexity / task quality of any router change (needs a model run; a 125B router fine-tune is out of reach)", "the sim swaps previous-token experts at random positions; a trained router would pick which to keep", "no router logits exist in the traces"])

def fam_s1_7(quick=False):
    """Requantise experts to 2/3 bit so more fit in RAM. Decisive free test: byte-capacity LRU replay on the traces at 4.3/3.0/2.6/2.06 bits per weight."""
    tr = _tr(quick); bpws = (4.3, 2.6) if quick else (4.3, 3.0, 2.6, 2.06); rams = (8,) if quick else (8, 16, 32); out = {}
    streams = {t: tt.accesses([{l: tuple(x) for l, x in enumerate(tok)} for tok in tokens(t)]) for t in tr}
    for r in rams:
        for b in bpws:
            fpn = cache_footprints(CACHE_GIB[r], b); cap = int(fpn * FOOT); hit = mean(cs.lru_hits(streams[t], cap) / len(streams[t]) for t in tr)
            gb = FP_GB * b / 4.3; io = io_s(1 - hit, gb)
            out[f"{r}GB|{b}"] = dict(footprints=fpn, hit=hit, gb_per_token_read=(1 - hit) * gb, serial_tps=tps_serial(io), overlap_tps=tps_overlap(io))
    gains = {}
    for r in rams:
        for b in bpws:
            if b != 4.3: gains[f"{r}GB|{b}"] = dict(serial=out[f"{r}GB|{b}"]["serial_tps"] / out[f"{r}GB|4.3"]["serial_tps"], overlap=out[f"{r}GB|{b}"]["overlap_tps"] / out[f"{r}GB|4.3"]["overlap_tps"])
    size = {str(b): EXPERT_FILE_GB * b / 4.3 for b in (4.3, 3.0, 2.6, 2.06)}
    fits32 = {k: v <= CACHE_GIB[32] * GIB / 1e9 for k, v in size.items()}
    sb = {}
    for gbs in ((4.5,) if quick else (4.5, 12.0, 26.0)):
        for b in (4.3, 2.6):
            fpn = gbs * 1e9 / (FP_GB * 1e9 * b / 4.3); hit = mean(cs.lru_hits(streams[t], int(fpn * FOOT)) / len(streams[t]) for t in tr); io = io_s(1 - hit, FP_GB * b / 4.3)
            sb[f"{gbs}GB|{b}"] = dict(footprints=fpn, hit=hit, serial_tps=tps_serial(io), overlap_tps=tps_overlap(io))
    sbg = {f"{g}GB": dict(serial=sb[f"{g}GB|2.6"]["serial_tps"] / sb[f"{g}GB|4.3"]["serial_tps"], overlap=sb[f"{g}GB|2.6"]["overlap_tps"] / sb[f"{g}GB|4.3"]["overlap_tps"]) for g in ((4.5,) if quick else (4.5, 12.0, 26.0))}
    spec = {}
    if not quick:
        fpn = cache_footprints(CACHE_GIB[8], 2.6); b1 = mean(spec_sim(t, 1, 1.0, fpn)["mpa"] for t in tr); b2 = mean(spec_sim(t, 2, 0.7, fpn, False, "wrong")["mpa"] for t in tr)
        spec = dict(footprints=fpn, k2_a0p7_bytes_per_accepted_vs_k1=b2 / b1)
    return dict(family="screen 1 #7", cache=out, gain_vs_4bit=gains, screen_budget_cache=sb, screen_budget_gain_2p6_vs_4p3=sbg, trace_cold_start_hit_ceiling=max(v["hit"] for v in out.values()), expert_file_gb=size, whole_file_fits_32gb_cache=fits32, cache_32gb_gb=CACHE_GIB[32] * GIB / 1e9,
                spec_k2_on_2p6bit_8gb=spec, relative_ppl_loss_quoted=dict(Q3_K_S=0.104, Q2_K=0.26, source="earlier OLMoE perplexity runs recorded in the repo's combination simulator; requantised, so upper bounds"),
                unmeasured=["accuracy / perplexity of mixed 2/3/4-bit experts on the 125B (needs model runs)", "dequant cost of 2-bit kernels on this CPU (needs hardware)", "hit rates are 30B-trace footprints mapped to the 125B, unvalidated on 512 experts"])

RUNNERS = dict(fam_s1_0=fam_s1_0, fam_s1_1=fam_s1_1, fam_s1_2=fam_s1_2, fam_s1_3=fam_s1_3, fam_s1_4=fam_s1_4, fam_s1_5=fam_s1_5, fam_s1_6=fam_s1_6, fam_s1_7=fam_s1_7)
# Catalog method rows per family (names exactly as in the catalog). needs: 'local' = the decisive free test runs here on the traces; 'model' = the decisive test needs a model run that is not available here (the closed-form and trace-proxy parts are computed).
FAMILIES = {
    "fam_s1_0": {'idea': "k=8 draft/lookahead tree verified expert-major: load the union of experts once per layer", 'methods': ["MoE-Spec: Expert Budgeting for Efficient Speculative Decoding (Feb 2026)",
        "EcoSpec: Less Experts, Faster Decoding (2607.12696)",
        "DraftExpert: self-speculative decoding for end-device expert offload (2607.24434)",
        "SpecMoEOff: hiding offloading latency with speculative decoding (2508.21706)",
        "SP-MoE (2510.10302), MoE-SpeQ (2511.14102)",
        "S2-MoE self-speculative decoding on edge, built on llama.cpp (2608.15018)",
        "SpecExec (2406.02532)"], 'run': fam_s1_0, 'needs': "local"},
    "fam_s1_1": {'idea': "decode a group of k tokens as one so the union of their experts is read once (disk bytes per token)", 'methods': ["Utility-Driven Speculative Decoding for MoE (Cascade)",
        "MoE-Spec: expert budgeting for speculative decoding",
        "AcceptMoE",
        "MoE-SpeQ / SP-MoE / DraftExpert / S2-MoE (speculation plus expert prefetch)",
        "Hiding offloading latency with speculative decoding"], 'run': fam_s1_1, 'needs': "local"},
    "fam_s1_2": {'idea': "500-token prefill as expert-major work: read each needed expert slice once per layer", 'methods': ["EStream: MoE prefill through expert virtualization on mobile NPUs",
        "Fate: fast edge MoE inference via cross-layer gate",
        "DAOP: data-aware offloading for MoE",
        "Fiddler (ICLR 2025)",
        "FlashMoE: SSD I/O bottlenecks with ML cache replacement",
        "llama.cpp ggml_mul_mat_id (PR #4406)",
        "Flash-MoE (397B on a 48GB MacBook)"], 'run': fam_s1_2, 'needs': "model"},
    "fam_s1_3": {'idea': "cut routed experts per token from top-10 to 6 (drop the lowest-weight experts)", 'methods': ["Not All Experts are Equal (NAEE): expert pruning and dynamic skipping",
        "Opportunistic Expert Activation: batch-aware routing",
        "LExI: layer-adaptive active experts",
        "REAP: router-weighted expert activation pruning (Cerebras, ICLR 2026)",
        "SliceMoE: bit-sliced expert caching with cache-aware routing",
        "DyMoE: 8/4-bit/skip per expert for edge devices",
        "vllm-mlx --moe-top-k guide"], 'run': fam_s1_3, 'needs': "local"},
    "fam_s1_4": {'idea': "process k same-sequence positions together, expert-major, each slice read once for all positions", 'methods': ["DraftExpert (self-speculative decoding for end-device MoE, flash-resident experts)",
        "AcceptMoE (verifier expert sets under offloading)",
        "MoE-Spec (budgeted expert set during verification)",
        "Reddit snapshot claiming spec decoding is dead under SSD offload (low reliability)"], 'run': fam_s1_4, 'needs': "local"},
    "fam_s1_5": {'idea': "small resident draft + union verify of the big MoE, plus a confidence gate that skips the big model", 'methods': ["MoE-Spec: Expert Budgeting for Efficient Speculative Decoding",
        "SP-MoE: Speculative Decoding and Prefetching for MoE inference",
        "Utility-Driven Speculative Decoding for MoE (Cascade)",
        "Faster Cascades via Speculative Decoding (ICLR 2025)",
        "Single-user llama.cpp MoE speculative test (hackmd)"], 'run': fam_s1_5, 'needs': "local"},
    "fam_s1_6": {'idea': "fine-tune the router with a temporal-stickiness term so consecutive tokens reuse experts", 'methods': ["ReMoE: router fine-tuning with a temporal term for expert reuse in memory-constrained MoE inference (ICML 2026)",
        "Mixture of Cache-Conditional Experts (Skliar et al., TMLR 2025), training-free cache-aware routing",
        "Not All Models Suit Expert Offloading: local routing consistency of 20 MoE LLMs (ICLR 2026)",
        "vLLM issue 38256: dedup of expert loads in batched prefill",
        "ExpertFlow (arXiv 2410.17954)"], 'run': fam_s1_6, 'needs': "local"},
    "fam_s1_7": {'idea': "requantise experts to mixed 2/3/4 bit so far more of the file fits in RAM", 'methods': ["Eliseev and Mazur, Fast Inference of MoE LMs with Offloading",
        "HOBBIT mixed-precision expert offloading (on llama.cpp)",
        "MoE-SpeQ, speculative quantized decoding with expert prefetch",
        "Adaptive Verification for MoE speculative decoding (EVICT)",
        "Bandwidth-efficient adaptive MoE via low-rank compensation"], 'run': fam_s1_7, 'needs': "local"},
}

if __name__ == "__main__":
    for n in (sys.argv[1:] or sorted(RUNNERS)): print(n, json.dumps(RUNNERS[n](), indent=1, sort_keys=True, default=str))
