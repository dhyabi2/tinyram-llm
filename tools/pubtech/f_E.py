#!/usr/bin/env python3
"""Published-technique family tests, group E: second screen families 10 to 17.
Each fam_s2_N() recomputes the family's free decisive test on the three recorded Qwen3-30B router traces
(data/traces/qwen3-30b_{db,story,code}.txt: top-8 of 128 experts, 48 layers, about 250-300 decode tokens each) or, where a model
or hardware is needed, computes the closed-form part and says what is left unmeasured. Stdlib only, deterministic.
Mapping 30B -> 125B (512 experts, top-10, 48 layers, 2.7 MB per expert at 4.3 bpw): cache entries = footprints x 384 accesses per token;
8/16/32 GB machines have about 2.1/7.6/20.8 footprints of expert cache. The 30B traces stand in for the 125B: that is a caveat on every number here."""
import collections, math, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import combo_model as cm, combo_trace_sim as cs, cachesim as cx

TRACES = list(cs.TRACES); NL, K8, E128, FOOT = 48, 8, 128, 384
FP = {"8 GB": 2.1, "16 GB": 7.6, "32 GB": 20.8}                  # footprints of expert cache of the 125B per machine size
SLICE_MB = 2.7; SLICES_125 = 480
FD = cm.FLOP_TOT * cm.DENSE_FRAC; FE = cm.FLOP_TOT - FD          # dense and expert GFLOP per token of the 125B (combo_model constants, [E])
COMP_S = cm.FLOP_TOT / cm.G_CPU                                  # seconds of compute per token on the 4-core laptop (decode rate 55 GFLOP/s [M])

def toks_of(name): return cs.tr(name)                            # [token][layer] = ranked expert ids
def stream_of(T, m=None, layers=None):
    return [(l, e) for tok in T for l in (range(NL) if layers is None else layers) for e in (tok[l] if m is None else tok[l][:m])]
def mean(xs): xs = list(xs); return sum(xs) / len(xs)
def r4(x): return round(x, 4)

def serial_tps(comp_s, io_s): return 1.0 / (comp_s + io_s)
def overlap_tps(comp_s, io_s): return 1.0 / max(comp_s, io_s)     # ideal full overlap bound (an upper bound, not an implemented path)

# ---------------------------------------------------------------- family 10: expert compute server (split 512 experts across 2 nodes)
def hyper_straggler(k, n):
    """Exact E[max(a, k-a)] / (k/2) when k distinct experts are drawn uniformly from n and the n experts are split n/2 : n/2."""
    h = n // 2
    return sum(math.comb(h, j) * math.comb(h, k - j) * max(j, k - j) for j in range(k + 1)) / math.comb(n, k) / (k / 2)

def straggler(T, side):
    """sum over tokens and layers of max(a, b) divided by the fair share k/2; side[l][e] in {0, 1} is the node of expert e at layer l."""
    s = n = 0.0
    for tok in T:
        for l in range(NL):
            a = sum(side[l][e] for e in tok[l]); s += max(a, len(tok[l]) - a); n += len(tok[l]) / 2
    return s / n

def random_side(seed, E=E128):
    side = []
    for l in range(NL):
        rng = random.Random(seed * 1009 + l); perm = list(range(E)); rng.shuffle(perm); s = [0] * E
        for e in perm[: E // 2]: s[e] = 1
        side.append(s)
    return side

def balanced_side(freq, E=E128):
    """Greedy balanced split per layer from usage counts freq[l][e]: heaviest first to the lighter node, 64 experts per node."""
    side = []
    for l in range(NL):
        load = [0, 0]; cnt = [0, 0]; s = [0] * E
        for e in sorted(range(E), key=lambda e: (-freq[l][e], e)):
            n = 0 if (load[0] <= load[1] and cnt[0] < E // 2) or cnt[1] >= E // 2 else 1
            s[e] = n; load[n] += freq[l][e]; cnt[n] += 1
        side.append(s)
    return side

def freqs(Ts, E=E128):
    f = [[0] * E for _ in range(NL)]
    for T in Ts:
        for tok in T:
            for l in range(NL):
                for e in tok[l]: f[l][e] += 1
    return f

def node_streams(T, side):
    s0 = []; s1 = []
    for tok in T:
        for l in range(NL):
            for e in tok[l]: (s1 if side[l][e] else s0).append((l, e))
    return s0, s1

def fam_s2_10(seeds=20):
    Ts = {n: toks_of(n) for n in TRACES}; res = {"straggler": {}, "hits": {}, "tps": {}}
    res["straggler_uniform_k8_n128"] = r4(hyper_straggler(8, 128)); res["straggler_uniform_k10_n512"] = r4(hyper_straggler(10, 512))
    per_tr = {}
    for n in TRACES:
        T = Ts[n]; rs = [straggler(T, random_side(s)) for s in range(seeds)]
        others = [Ts[o] for o in TRACES if o != n]; half = len(T) // 2
        bal_in = straggler(T, balanced_side(freqs([T]))); bal_ho = straggler(T, balanced_side(freqs(others)))
        bal_split = straggler(T[half:], balanced_side(freqs([T[:half]])))
        per_tr[n] = dict(random_mean=r4(mean(rs)), random_min=r4(min(rs)), random_max=r4(max(rs)), balanced_in_sample=r4(bal_in), balanced_held_out=r4(bal_ho), balanced_first_to_second_half=r4(bal_split))
    res["straggler"] = per_tr
    keys = ("random_mean", "random_min", "random_max", "balanced_in_sample", "balanced_held_out", "balanced_first_to_second_half")
    res["straggler_mean_over_traces"] = {k: r4(mean(per_tr[n][k] for n in TRACES)) for k in keys}
    s_rand = res["straggler_mean_over_traces"]["random_mean"]; s_bal = res["straggler_mean_over_traces"]["balanced_held_out"]
    rtt = 48 * 0.3e-3
    for mc, fp in FP.items():
        cap = int(fp * FOOT); h1 = []; hn = []; miss_n = []
        for n in TRACES:
            T = Ts[n]; full = stream_of(T); h1.append(cx.lru_hits(full, cap) / len(full))
            s0, s1 = node_streams(T, random_side(0)); m0 = len(s0) - cx.lru_hits(s0, cap); m1 = len(s1) - cx.lru_hits(s1, cap)
            hn.append(1 - (m0 + m1) / (len(s0) + len(s1))); miss_n.append(max(m0, m1) / len(T))
        h_single = mean(h1); h_pool = mean(hn)
        ent = SLICES_125 / FOOT * SLICE_MB * 1e-3                 # GB per trace entry on the 125B
        io1 = (1 - h_single) * cm.EXP_GB / cm.R_EXPL
        io2 = mean(miss_n) * ent / cm.R_EXPL                      # busier node's miss bytes per token
        comp2 = (FD + FE * s_rand / 2) / cm.G_CPU + rtt           # dense part stays on node 0 (Amdahl); experts split with the measured straggler
        comp2_all = cm.FLOP_TOT * s_rand / 2 / cm.G_CPU + rtt     # the screen's assumption: ALL compute halves (only valid if dense is split too)
        comp2_screen_bal = cm.FLOP_TOT / 2 / cm.G_CPU             # perfect halving, no straggler, no RTT (the claimed 9.2 tok/s ceiling)
        res["hits"][mc] = dict(single_node=r4(h_single), pooled_two_nodes=r4(h_pool))
        res["tps"][mc] = dict(single_serial=r4(serial_tps(COMP_S, io1)), single_overlap_bound=r4(overlap_tps(COMP_S, io1)),
                              two_serial=r4(serial_tps(comp2, io2)), two_overlap_bound=r4(overlap_tps(comp2, io2)),
                              gain_serial=r4(serial_tps(comp2, io2) / serial_tps(COMP_S, io1)), gain_overlap_bound=r4(overlap_tps(comp2, io2) / overlap_tps(COMP_S, io1)),
                              gain_serial_if_all_compute_split=r4(serial_tps(comp2_all, io2) / serial_tps(COMP_S, io1)), gain_overlap_if_all_compute_split=r4(overlap_tps(comp2_all, io2) / overlap_tps(COMP_S, io1)))
    res["compute_gain_amdahl_two_nodes"] = r4(COMP_S / ((FD + FE * s_rand / 2) / cm.G_CPU))
    res["compute_gain_if_all_split"] = r4(COMP_S / (cm.FLOP_TOT * s_rand / 2 / cm.G_CPU))
    res["screen_claimed_compute_bound_2_over_straggler"] = r4(2 / 1.26)
    res["rtt_ms_48_layers_at_0p3ms"] = r4(rtt * 1e3)
    res["straggler_balanced_below_1p15"] = s_bal < 1.15
    return res

# ---------------------------------------------------------------- family 11: layer-split expert store on a peer (RDMA / 25 GbE)
def two_tier_io(foot_gb, h_local, h_total, rate_peer, rate_disk=cm.R_EXPL):
    """Seconds of expert I/O per token when `h_local` of accesses hit the laptop cache, `h_total - h_local` are served from the peer's DRAM at rate_peer, and the rest from local NVMe."""
    return (h_total - h_local) * foot_gb / rate_peer + (1 - h_total) * foot_gb / rate_disk

def layer_split_io(miss_gb, local_layers, rate_peer, rate_disk=cm.R_EXPL, parallel=True):
    """Misses split evenly over layers: `local_layers` of 48 come from local NVMe, the rest from the peer over the link."""
    a = miss_gb * local_layers / 48 / rate_disk; b = miss_gb * (48 - local_layers) / 48 / rate_peer
    return max(a, b) if parallel else a + b

def fam_s2_11():
    sc = cm.sc_at(1); base = dict(K=10, bits=4.3, lowrank=False, route=False, overlap=True, spec=0, gpu=0, lut=False, exit=1.0); ser = dict(base, overlap=False)
    rates = (cm.R_EXPL, 2.5, 1.1, 0.11); res = {"model_rows": {}, "peer_cache": {}, "layer_split": {}}
    for mc in ("8 GB", "16 GB"):
        o0 = cm.tok_s(base, mc, sc); s0 = cm.tok_s(ser, mc, sc)
        for r in rates:
            o = cm.tok_s(base, mc, sc, rate=r); s = cm.tok_s(ser, mc, sc, rate=r)
            res["model_rows"][f"{mc} @ {r} GB/s"] = dict(overlap=r4(o), serial=r4(s), overlap_vs_local=r4(o / o0), serial_vs_local=r4(s / s0))
    # peer DRAM as extra cache: laptop cache (cm.CACHE_GIB) plus 13 GB on the peer; hits beyond the local cache cost a network fetch
    for mc in ("8 GB", "16 GB"):
        foot = cm.EXP_GB; hl = cm.lru(cm.CACHE_GIB[mc] * 1.074 / foot); ht = cm.lru((cm.CACHE_GIB[mc] + 13) * 1.074 / foot); io0 = (1 - hl) * foot / cm.R_EXPL
        for r in rates:
            io = two_tier_io(foot, hl, ht, r)
            res["peer_cache"][f"{mc} @ {r} GB/s"] = dict(h_local=r4(hl), h_with_peer=r4(ht), serial=r4(serial_tps(COMP_S, io)), serial_vs_local=r4(serial_tps(COMP_S, io) / serial_tps(COMP_S, io0)),
                                                         overlap_bound=r4(overlap_tps(COMP_S, io)), overlap_vs_local=r4(overlap_tps(COMP_S, io) / overlap_tps(COMP_S, io0)))
    # the idea as stated: 10 of 48 layers local on NVMe, 38 on the peer; misses split by layers; local cache hit from the model's LRU curve
    for mc in ("8 GB", "16 GB"):
        hl = cm.lru(cm.CACHE_GIB[mc] * 1.074 / cm.EXP_GB); miss_gb = (1 - hl) * cm.EXP_GB; io0 = miss_gb / cm.R_EXPL
        for r in rates:
            for par in (True, False):
                io = layer_split_io(miss_gb, 10, r, parallel=par)
                res["layer_split"][f"{mc} @ {r} GB/s {'parallel' if par else 'sum'}"] = dict(serial=r4(serial_tps(COMP_S, io)), overlap_bound=r4(overlap_tps(COMP_S, io)),
                                                                                       serial_vs_local=r4(serial_tps(COMP_S, io) / serial_tps(COMP_S, io0)), overlap_vs_local=r4(overlap_tps(COMP_S, io) / overlap_tps(COMP_S, io0)))
    layer_gb = 512 * SLICE_MB * 1e-3
    res["peer_gb_needed_for_38_layers"] = r4(38 * layer_gb); res["layers_a_16GB_peer_holds"] = int(16.0 // layer_gb); res["gb_per_layer"] = r4(layer_gb)
    res["compute_ceiling_tps"] = r4(1 / COMP_S)
    res["max_gain_from_faster_link"] = r4(overlap_tps(COMP_S, 0) / overlap_tps(COMP_S, (1 - cm.lru(cm.CACHE_GIB["8 GB"] * 1.074 / cm.EXP_GB)) * cm.EXP_GB / cm.R_EXPL))
    return res

# ---------------------------------------------------------------- family 12: zone-map sparsity inside experts (needs activations: model)
def gate_first_factor(d): return 1 / 3 + 2 / 3 * (1 - d)          # read gate fully, then only the live share of up and down rows
def predictor_factor(d): return 1 - d                              # an oracle or learned predictor that avoids reading gate for dead neurons
def union_live(d, k): return 1 - d ** k                            # live share of the union of k tokens if dead sets were independent (pessimistic for the idea only if tokens correlate)
def read_rate(size_bytes, bw_gbs=cm.R_EXPL, lat_s=80e-6, qd=4):
    """Effective GB/s of reads of `size_bytes` with queue depth qd. lat_s is an ASSUMED per-request latency [E]; the formula reproduces the measured 3.1 GB/s near 2.7 MB."""
    t = lat_s / qd + size_bytes / (bw_gbs * 1e9); return size_bytes / t / 1e9
def block_bytes(B, H, bpw=4.3): return B * H * bpw / 8

def persistence(T):
    """Mean over layers and consecutive tokens of |S_t & S_(t-1)| / k."""
    s = n = 0
    for t in range(1, len(T)):
        for l in range(NL): s += len(set(T[t][l]) & set(T[t - 1][l])); n += len(T[t][l])
    return s / n

def fam_s2_12():
    res = {"gate_first": {}, "predictor": {}, "frag": {}, "measured": {}}
    for d in (0.2, 0.3, 0.5, 0.6, 0.8):
        f = gate_first_factor(d); fl = FD + FE * f
        res["gate_first"][str(d)] = dict(bytes_factor=r4(f), bytes_gain=r4(1 / f), compute_ceiling_tps=r4(cm.G_CPU / fl), ceiling_lift=r4(cm.FLOP_TOT / fl))
        res["predictor"][str(d)] = dict(bytes_gain=r4(1 / predictor_factor(d)), compute_ceiling_tps=r4(cm.G_CPU / (FD + FE * predictor_factor(d))))
    H = 2048
    for B in (32, 128):
        bb = block_bytes(B, H); rr = read_rate(bb)
        res["frag"][f"block{B}"] = dict(block_bytes=int(bb), read_gbs=r4(rr), rate_loss_vs_2p7MB=r4(rr / read_rate(2.7e6)))
        for d in (0.3, 0.5):
            io_gain_bytes = 1 / gate_first_factor(d); io_gain_time = io_gain_bytes * rr / read_rate(2.7e6)
            res["frag"][f"block{B}"][f"io_time_gain_d{d}"] = r4(io_gain_time)
    res["read_rate_check_2p7MB"] = r4(read_rate(2.7e6)); res["union_live"] = {f"d{d}_k{k}": r4(union_live(d, k)) for d in (0.3, 0.5, 0.6) for k in (2, 4, 8)}
    pers = [persistence(toks_of(n)) for n in TRACES]; res["measured"]["persistence_per_trace"] = [r4(p) for p in pers]; res["measured"]["persistence_mean"] = r4(mean(pers))
    lru1 = [cx.lru_hits(stream_of(toks_of(n)), int(FP["8 GB"] * FOOT)) / len(stream_of(toks_of(n))) for n in TRACES]
    res["measured"]["lru_hit_at_8GB_mean"] = r4(mean(lru1))
    claimed = (1 - 0.5) * (1 - 0.42); proper = (1 - mean(lru1)) * gate_first_factor(0.5)
    res["claimed_bytes_factor_50dead_42reuse"] = r4(claimed); res["bytes_factor_after_lru_and_gate_first_d0.5"] = r4(proper)
    res["bytes_gain_over_best_path_d0.5_gate_first"] = r4(1 / gate_first_factor(0.5)); res["claimed_gain"] = 2.6
    res["unmeasured"] = "dead fraction d of the SwiGLU intermediate at a fixed output error (needs activations of a real model); quality loss"
    return res

# ---------------------------------------------------------------- family 13: typing-time chunked prefill
def chunk_unique(T, c, N=None):
    """(sum over consecutive chunks of c tokens of the unique (layer, expert) pairs, unique pairs of the whole first N tokens)."""
    N = min(N or len(T), len(T)); tot = 0
    for i in range(0, N, c):
        tot += len({(l, e) for tok in T[i:i + c] for l in range(NL) for e in tok[l]})
    return tot, len({(l, e) for tok in T[:N] for l in range(NL) for e in tok[l]})

def uniform_unique_per_layer(c, E=512, k=10): return E * (1 - (1 - k / E) ** c)

def typing_ttft(n_tokens, c, io_chunk_s, comp_chunk_s, typing_tps):
    """Seconds after send until the first token can start decoding: chunks are processed in order as they fill; the last full chunk fills at send time."""
    fill = c / typing_tps; chunk = max(io_chunk_s, comp_chunk_s); nch = -(-n_tokens // c); fin = 0.0
    for i in range(nch):
        arrive = min((i + 1) * c, n_tokens) / typing_tps; fin = max(fin, arrive) + chunk
    return fin - n_tokens / typing_tps

def fam_s2_13(N125=500, typing_tps=7.0, link=3.1):
    chunks = (8, 16, 32, 64, 128, 500); res = {"measured_ratio": {}, "model125": {}}
    for c in chunks:
        r = []
        for n in TRACES:
            T = toks_of(n); u, w = chunk_unique(T, c); r.append(u / w)
        res["measured_ratio"][str(c)] = r4(mean(r))
    res["trace_tokens"] = [len(toks_of(n)) for n in TRACES]
    whole_gb = lambda: 48 * uniform_unique_per_layer(N125) * SLICE_MB * 1e-3
    cold_io = whole_gb() / link; cold_comp = N125 * cm.FLOP_TOT / 311.0; cold = max(cold_io, cold_comp)
    res["cold_prefill_s"] = dict(io=r4(cold_io), compute=r4(cold_comp), ttft=r4(cold), gb_read=r4(whole_gb()))
    for c in chunks:
        per_layer = uniform_unique_per_layer(c); gb = 48 * per_layer * SLICE_MB * 1e-3; io = gb / link; comp = c * cm.FLOP_TOT / 311.0
        nch = -(-N125 // c); total_gb = gb * nch; ttft = typing_ttft(N125, c, io, comp, typing_tps)
        res["model125"][str(c)] = dict(gb_per_chunk=r4(gb), chunk_io_s=r4(io), chunk_comp_s=r4(comp), fill_s=r4(c / typing_tps), keeps_up=max(io, comp) <= c / typing_tps,
                                       total_gb=r4(total_gb), io_ratio_vs_one_batch=r4(total_gb / whole_gb()), ttft_after_send_s=r4(ttft), ttft_speedup=r4(cold / ttft),
                                       disk_busy_share_of_typing_time=r4(total_gb / link / (N125 / typing_tps)))
    ok = [c for c in chunks if res["model125"][str(c)]["keeps_up"]]
    res["smallest_chunk_that_keeps_up"] = ok[0] if ok else None
    res["best_ttft_speedup"] = max(v["ttft_speedup"] for v in res["model125"].values())
    sens = {}                                                    # slower typists leave the disk idle between chunks
    for tps in (1.0, 2.0, 3.5, 7.0, 14.0):
        best = (0.0, None)
        for c in chunks:
            io = 48 * uniform_unique_per_layer(c) * SLICE_MB * 1e-3 / link; comp = c * cm.FLOP_TOT / 311.0
            sp = cold / typing_ttft(N125, c, io, comp, tps)
            if sp > best[0]: best = (sp, c)
        sens[str(tps)] = dict(best_ttft_speedup=r4(best[0]), best_chunk=best[1])
    res["typing_speed_sensitivity"] = sens
    return res

# ---------------------------------------------------------------- family 14: ARC per-sequence cache vs LRU and Belady
def arc_run(trace, cap):
    """Adaptive Replacement Cache (Megiddo and Modha). Returns (hits, max resident entries). cap <= 0 gives (0, 0)."""
    if cap <= 0: return 0, 0
    t1 = collections.OrderedDict(); t2 = collections.OrderedDict(); b1 = collections.OrderedDict(); b2 = collections.OrderedDict(); p = 0; h = 0; mx = 0
    def replace(in_b2):
        if t1 and (len(t1) > p or (in_b2 and len(t1) == p)): k, _ = t1.popitem(last=False); b1[k] = None
        else: k, _ = t2.popitem(last=False); b2[k] = None
    for x in trace:
        if x in t1: del t1[x]; t2[x] = None; h += 1
        elif x in t2: t2.move_to_end(x); h += 1
        elif x in b1:
            p = min(cap, p + max(len(b2) // len(b1), 1)); replace(False); del b1[x]; t2[x] = None
        elif x in b2:
            p = max(0, p - max(len(b1) // len(b2), 1)); replace(True); del b2[x]; t2[x] = None
        else:
            l1 = len(t1) + len(b1)
            if l1 == cap:
                if len(t1) < cap: b1.popitem(last=False); replace(False)
                else: t1.popitem(last=False)
            elif l1 + len(t2) + len(b2) >= cap:
                if l1 + len(t2) + len(b2) == 2 * cap: b2.popitem(last=False)
                replace(False)
            t1[x] = None
        mx = max(mx, len(t1) + len(t2))
    return h, mx

def arc_hits(trace, cap): return arc_run(trace, cap)[0]
def belady_hits_fast(trace, cap): return len(trace) - cs.belady(trace, cap) if cap > 0 else 0

def tps_from_hit(h): return serial_tps(COMP_S, (1 - h) * cm.EXP_GB / cm.R_EXPL)

def fam_s2_14(fps=(1.0, 2.0, 2.1, 3.0, 4.0, 7.6)):
    res = {"hit": {}, "gap": {}, "tps_gain": {}, "hot_list": {}}
    for fp in fps:
        cap = int(fp * FOOT); row = {}
        for n in TRACES:
            s = stream_of(toks_of(n)); N = len(s); l = cx.lru_hits(s, cap) / N; a = arc_hits(s, cap) / N; b = belady_hits_fast(s, cap) / N
            others = [stream_of(toks_of(o)) for o in TRACES if o != n]; learn = [x for o in others for x in o]
            hot = cx.pinned_hits(s, cap, learn=learn) / N
            row[n] = dict(lru=r4(l), arc=r4(a), belady=r4(b), hot_list_held_out=r4(hot))
        res["hit"][str(fp)] = row
        res["gap"][str(fp)] = dict(arc_minus_lru=r4(mean(row[n]["arc"] - row[n]["lru"] for n in TRACES)), belady_minus_lru=r4(mean(row[n]["belady"] - row[n]["lru"] for n in TRACES)),
                                   arc_minus_lru_min=r4(min(row[n]["arc"] - row[n]["lru"] for n in TRACES)), arc_minus_lru_max=r4(max(row[n]["arc"] - row[n]["lru"] for n in TRACES)))
        hl = mean(row[n]["lru"] for n in TRACES); ha = mean(row[n]["arc"] for n in TRACES); hb = mean(row[n]["belady"] for n in TRACES); hh = mean(row[n]["hot_list_held_out"] for n in TRACES)
        res["tps_gain"][str(fp)] = dict(arc_over_lru_serial=r4(tps_from_hit(ha) / tps_from_hit(hl)), belady_over_lru_serial=r4(tps_from_hit(hb) / tps_from_hit(hl)))
        res["hot_list"][str(fp)] = dict(lru_mean=r4(hl), hot_list_mean=r4(hh))
    res["claim_threshold_arc_gain"] = 0.03
    res["kill_arc_below_0p03_at_2_3_fp"] = max(res["gap"]["2.0"]["arc_minus_lru"], res["gap"]["3.0"]["arc_minus_lru"]) < 0.03
    res["network_tier_note_gbs"] = dict(one_gbe=0.117, ten_gbe=1.15, local_nvme_explicit=cm.R_EXPL)
    return res

# ---------------------------------------------------------------- family 15: all experts in a second machine's RAM (hardware)
LINKS = {"1GbE": 0.117, "2.5GbE": 0.29, "10GbE": 1.15, "TB-class 2.8": 2.8, "RDMA 5": 5.0, "local NVMe explicit": cm.R_EXPL}

def link_tps(rate_gbs, hit, comp_s=COMP_S, exp_gb=cm.EXP_GB):
    io = (1 - hit) * exp_gb / rate_gbs; return serial_tps(comp_s, io), overlap_tps(comp_s, io)

def required_rate(exp_gb, seconds): return exp_gb / seconds

def fam_s2_15():
    res = {"uncached": {}, "lru8gb": {}, "lru_by_machine": {}}
    h = {mc: mean(cx.lru_hits(stream_of(toks_of(n)), int(fp * FOOT)) / len(stream_of(toks_of(n))) for n in TRACES) for mc, fp in FP.items()}
    res["lru_by_machine"] = {k: r4(v) for k, v in h.items()}
    for name, r in LINKS.items():
        s, o = link_tps(r, 0.0); s2, o2 = link_tps(r, h["8 GB"])
        res["uncached"][name] = dict(serial=r4(s), overlap_bound=r4(o)); res["lru8gb"][name] = dict(serial=r4(s2), overlap_bound=r4(o2))
    loc = res["lru8gb"]["local NVMe explicit"]
    res["vs_local_8gb"] = {name: dict(serial=r4(v["serial"] / loc["serial"]), overlap_bound=r4(v["overlap_bound"] / loc["overlap_bound"])) for name, v in res["lru8gb"].items()}
    res["rate_needed_for_1p3GB_in_0p3s"] = r4(required_rate(cm.EXP_GB, 0.3)); res["local_rate_gbs"] = cm.R_EXPL
    res["seconds_for_1p3GB"] = {name: r4(cm.EXP_GB / r) for name, r in LINKS.items()}
    res["compute_ceiling_tps"] = r4(1 / COMP_S)
    res["unmeasured"] = "link throughput and round-trip time between two real machines; remote-RAM expert fetch rate; needs a second host with 69+ GB RAM"
    return res

# ---------------------------------------------------------------- family 16: top-k 10 -> 4 with 2-bit experts
def ceiling_at_k(k, kmax=10): return cm.G_CPU / (FD + FE * k / kmax)

def sim_topm(T, m, cap):
    """LRU over the first m ranked experts of every (token, layer). Returns (misses, accesses)."""
    c = cs.Cache(cap, "lru"); mis = acc = 0
    for tok in T:
        for l in range(NL):
            for e in tok[l][:m]:
                acc += 1
                if not c.touch((l, e)): mis += 1
    return mis, acc

def fam_s2_16():
    res = {"ceiling": {}, "combo_model": {}, "trace": {}}
    for k in (10, 8, 6, 4, 2): res["ceiling"][str(k)] = r4(ceiling_at_k(k))
    res["ceiling_lift_top4"] = r4(ceiling_at_k(4) / ceiling_at_k(10)); res["claimed_lift"] = 2.5; res["claimed_ceiling_tps"] = 11.5
    sc = cm.sc_at(1)
    for mc in FP:
        for ov in (True, False):
            b = dict(K=10, bits=4.3, lowrank=False, route=False, overlap=ov, spec=0, gpu=0, lut=False, exit=1.0); base = cm.tok_s(b, mc, sc)
            row = {}
            for k, bits in ((4, 4.3), (4, 2.5), (6, 4.3)):
                row[f"top{k}_{bits}bit"] = r4(cm.tok_s(dict(b, K=k, bits=bits), mc, sc) / base)
            res["combo_model"][f"{mc} {'overlap' if ov else 'serial'}"] = dict(base_tps=r4(base), **row)
    for mc, fp in FP.items():
        for bits in (4.3, 2.5):
            fpb = fp * 4.3 / bits; cap = int(fpb * FOOT)
            for m in (8, 4, 3):
                hits = []; ser = []; ovl = []
                for n in TRACES:
                    T = toks_of(n); mis, acc = sim_topm(T, m, cap); hits.append(1 - mis / acc)
                    io = mis / len(T) * (10 / 8) * SLICE_MB * 1e-3 * (bits / 4.3) / cm.R_EXPL
                    comp = (FD + FE * (m / 8)) / cm.G_CPU
                    ser.append(serial_tps(comp, io)); ovl.append(overlap_tps(comp, io))
                res["trace"][f"{mc} {bits}bit m{m}"] = dict(hit=r4(mean(hits)), serial=r4(mean(ser)), overlap_bound=r4(mean(ovl)))
    for mc in FP:
        for bits in (4.3, 2.5):
            b = res["trace"][f"{mc} 4.3bit m8"]
            for m in (4, 3):
                t = res["trace"][f"{mc} {bits}bit m{m}"]
                res["trace"][f"{mc} {bits}bit m{m}"].update(serial_gain=r4(t["serial"] / b["serial"]), overlap_gain=r4(t["overlap_bound"] / b["overlap_bound"]))
    res["quality_olmoe_prior"] = dict(top6=cs.PPL["sel"]["top6"], top4=cs.PPL["sel"]["top4"], q2k=cs.PPL["bits"][2.5], top4_and_2bit=r4(cs.ppl_cost("top4", 2.5)), gate=0.05)
    res["passes_5pct_quality_gate"] = res["quality_olmoe_prior"]["top4_and_2bit"] <= 0.05
    res["unmeasured"] = "perplexity and decode speed of the top-k override on a model run (the decisive test); quality of the 125B after top-4 plus 2-bit; fine-tune recovery"
    return res

# ---------------------------------------------------------------- family 17: shared basis + low-rank residual per expert (needs weights)
def lowrank_cost(m, n, r, E=512, k=10, factor_bits=16.0, base_bits=4.3, variant="A"):
    """Per matrix m x n. Variant A: W_e ~ C_e B + U_e V_e (C_e m x r, B r x n shared, U_e m x r, V_e r x n). Variant B: residual only, W_e ~ U_e V_e (the shared part amortised away).
    Returns storage fraction in elements and bytes and the FLOP fraction of the k routed experts versus dense."""
    per_expert = r * (2 * m + n) if variant == "A" else r * (m + n)
    shared = r * n if variant == "A" else 0
    elems = per_expert + shared / E; dense = m * n
    flops = (k * (2 * per_expert) + (2 * r * n if variant == "A" else 0)) / (k * 2 * dense)
    return dict(elem_frac=elems / dense, byte_frac=elems * factor_bits / (dense * base_bits), flop_frac=flops)

def break_even_rank(m, n, factor_bits=16.0, base_bits=4.3, variant="B"):
    """Largest rank whose factor bytes still fit in the dense matrix bytes (no shared basis): r (m+n) bits_f = m n bits_base."""
    return m * n * base_bits / (factor_bits * (m + n))

def svd_energy(M, r, iters=40, seed=0):
    """Fraction of squared Frobenius norm captured by the top-r singular directions, by orthogonal (subspace) iteration. Pure python, small matrices only."""
    rows, cols = len(M), len(M[0]); rng = random.Random(seed)
    Q = [[rng.gauss(0, 1) for _ in range(r)] for _ in range(cols)]
    def orth(A):                                     # modified Gram-Schmidt on the columns of A (list of rows)
        nr, nc = len(A), len(A[0]); cols_ = [[A[i][j] for i in range(nr)] for j in range(nc)]; out = []
        for v in cols_:
            for u in out:
                d = sum(a * b for a, b in zip(v, u)); v = [a - d * b for a, b in zip(v, u)]
            nv = math.sqrt(sum(a * a for a in v)) or 1.0; out.append([a / nv for a in v])
        return [[out[j][i] for j in range(nc)] for i in range(nr)]
    Q = orth(Q)
    for _ in range(iters):
        Y = [[sum(M[i][c] * Q[c][j] for c in range(cols)) for j in range(r)] for i in range(rows)]            # M Q
        Z = [[sum(M[i][c] * Y[i][j] for i in range(rows)) for j in range(r)] for c in range(cols)]            # M^T (M Q)
        Q = orth(Z)
    Y = [[sum(M[i][c] * Q[c][j] for c in range(cols)) for j in range(r)] for i in range(rows)]
    cap = sum(Y[i][j] ** 2 for i in range(rows) for j in range(r)); tot = sum(x * x for row in M for x in row)
    return cap / tot

def gauss_matrix(rows, cols, seed):
    rng = random.Random(seed); return [[rng.gauss(0, 1) for _ in range(cols)] for _ in range(rows)]

def fam_s2_17():
    res = {"cost_30b_gate": {}, "cost_30b_down": {}, "break_even": {}, "null_energy": {}, "usage": {}}
    for r in (16, 32, 64, 128, 256):
        for tag, (m, n) in (("cost_30b_gate", (768, 2048)), ("cost_30b_down", (2048, 768))):
            row = {}
            for var in ("A", "B"):
                for fb in (16.0, 8.0):
                    c = lowrank_cost(m, n, r, factor_bits=fb, variant=var); row[f"{var}_fp{int(fb)}"] = dict(elem_frac=r4(c["elem_frac"]), byte_frac=r4(c["byte_frac"]), flop_frac=r4(c["flop_frac"]))
            res[tag][str(r)] = row
        fl = lowrank_cost(768, 2048, r, variant="A")["flop_frac"]; tot = FD + FE * fl
        res.setdefault("ceiling_lift_variantA", {})[str(r)] = r4(cm.FLOP_TOT / tot)
    for fb in (16.0, 8.0, 4.3): res["break_even"][f"rank_fp{fb}"] = r4(break_even_rank(768, 2048, factor_bits=fb))
    # null: i.i.d. Gaussian matrices have no low-rank structure; captured energy of the top-r is about r / min(m, n)
    for (m, n, r) in ((48, 128, 8), (48, 128, 16), (96, 64, 8)):
        e = svd_energy(gauss_matrix(m, n, 7), r); res["null_energy"][f"{m}x{n}_r{r}"] = dict(captured=r4(e), residual=r4(1 - e), rank_share=r4(r / min(m, n)))
    # routing usage of experts (RFID-MoE gives never-activated experts zero rank): measured on the traces
    nu = []; cov = []
    for n in TRACES:
        T = toks_of(n); never = []; cv = []
        for l in range(NL):
            c = collections.Counter(e for tok in T for e in tok[l]); never.append((E128 - len(c)) / E128)
            top = sum(v for _, v in c.most_common(E128 // 2)); cv.append(top / sum(c.values()))
        nu.append(mean(never)); cov.append(mean(cv))
    res["usage"] = dict(never_used_share_mean=r4(mean(nu)), never_used_share_per_trace=[r4(x) for x in nu], hottest_half_coverage=r4(mean(cov)))
    res["max_ceiling_lift_if_expert_flops_were_free"] = r4(cm.FLOP_TOT / FD)
    res["claimed"] = dict(flop_ratio="12 vs 0.28 GFLOP", residual_energy_le=0.10, rank=64)
    res["unmeasured"] = "residual energy of real expert weights at rank 64 (needs the dequantised tensors; no numpy or gguf package available, pure python cannot take the SVD of a stacked layer); quality after truncation"
    return res

FAMILIES = {
    "s2_10": {"idea": "Second LAN machine as an expert compute server: 256/256 expert split per layer, activations over the wire", "methods": [
        "OD-MoE (arXiv 2512.03927)", "exo blog, DeepSeek V3 on 8x M4 Mac Mini", "Latency-optimized expert placement for distributed edge MoE (arXiv 2508.12851)",
        "Inter-layer expert affinity (arXiv 2401.08383)", "Reddit snapshot on llama.cpp RPC over Thunderbolt"], "run": fam_s2_10, "needs": "local"},
    "s2_11": {"idea": "Layer-split expert store: 10 layers on local NVMe, 38 layers in a peer's RAM over RDMA or 25 GbE", "methods": [
        "exo with RDMA over Thunderbolt 5 on M3 Ultra Mac Studios (Geerling benchmark, via blog write-ups)", "Harvest: Opportunistic Peer-to-Peer GPU Caching for LLM Inference (arXiv 2602.00328)",
        "RMAI: Rethinking memory for AI inference (Noohi et al., Edinburgh)", "llama.cpp RPC backend guides and kjaiswal/llama-cpp-distributed-benchmarks",
        "FlashMoE, SSD offloading (arXiv 2601.17063) and MoE-SpeQ (arXiv 2511.14102)"], "run": fam_s2_11, "needs": "local"},
    "s2_12": {"idea": "Zone-map block sparsity inside each expert: skip gate/up/down column blocks whose SwiGLU activation is negligible", "methods": [
        "TEAL, training-free activation sparsity (ICLR 2025)", "FloE, on-the-fly MoE inference on memory-limited GPU", "TurboSparse / dReLU (Mixtral)", "PowerInfer-2",
        "DualSparse-MoE", "CATS / Enabling Dynamic Sparsity in Quantized LLM Inference"], "run": fam_s2_12, "needs": "model"},
    "s2_13": {"idea": "Chunked prefill of the live prompt prefix while the user types, persisting KV and warmed experts", "methods": [
        "OpenAI forum request for prefill-only requests and cache warm", "Google patent family, fulfillment ahead of selecting an autocomplete suggestion",
        "KV cache disk retrieval vs recompute (photonic-CXL study)", "Speculative Prefill (SpecPrefill)", "LMCache shared MoE KV pool"], "run": fam_s2_13, "needs": "local"},
    "s2_14": {"idea": "Per-sequence ARC expert cache in RAM instead of a global hot list, plus a network RAM tier", "methods": [
        "MoE-Infinity (request/sequence-level activation tracing for expert prefetch and cache)", "DAOP: per-sequence activation pattern from prefill decides expert placement",
        "OD-MoE: distributed edge-node on-demand expert loading", "llama.cpp RPC backend / mac-tensor (experts over network among Macs)"], "run": fam_s2_14, "needs": "local"},
    "s2_15": {"idea": "All 69 GB of experts in a second LAN machine's RAM, fetched per token over a fast link", "methods": [
        "exo 1.0 with RDMA over Thunderbolt 5 on Mac Studio clusters (Jeff Geerling test)", "OD-MoE, distributed on-demand expert loading without caches, ten-node testbed",
        "WDMoE and the Wi-Fi collaborative edge paper, experts on worker devices", "llama.cpp RPC backend (TCP) benchmarks and 10GbE guide"], "run": fam_s2_15, "needs": "hardware"},
    "s2_16": {"idea": "Active experts 10 -> 4 per token plus 2-bit experts: bytes and FLOPs both fall", "methods": [
        "Elbow-based MoE routing (OLMoE)", "HookMoE", "MC-MoE (mixed 2-bit expert quant + dynamic pruning)", "Qwen 3.5 122B 8-to-4 experts post", "Our own first screen R16 (top-k 10 to 6 or 4)"], "run": fam_s2_16, "needs": "model"},
    "s2_17": {"idea": "Per-layer shared SVD basis across 512 experts plus a rank-r residual per expert", "methods": [
        "D2-MoE: delta decompression, shared base plus per-expert SVD-compressed delta", "Sub-MoE: group experts, joint SVD with shared U per group", "MoBE: Mixture-of-Basis-Experts",
        "RFID-MoE: rank allocation by effective rank and routing frequency", "REAL: low-rank compression of the weakest experts"], "run": fam_s2_17, "needs": "model"},
}

if __name__ == "__main__":
    import json
    for k, v in FAMILIES.items(): print(k, json.dumps(v["run"]())[:1500])
