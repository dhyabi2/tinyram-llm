"""Group D published-technique families (second screen, ideas 2-9): deterministic checks on the recorded Qwen3-30B router traces, closed-form arithmetic and hand-computed unit cases.
Each family: determinism, sanity bounds, unit cases of its helpers, the measured number against the screen's claim (range in the check name), and a control that must be able to fail.
Weight-tensor probes run only when PUBTECH_GGUF points at a Q8_0 GGUF; the suite never needs a model."""
import math, os, random, struct, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "tools")); sys.path.insert(0, os.path.join(HERE, "..", "tools", "pubtech"))
from harness import T
import f_D as f
import combo_trace_sim as cs

def between(t, name, v, lo, hi): return t.check(f"{name} in [{lo}, {hi}]", lo <= v <= hi, str(v))

def write_tiny_gguf(path, name, ne0, ne1, ne2, d=0.5):
    """Minimal GGUF v3 file with one Q8_0 tensor (dims ne0 x ne1 x ne2), a string array and a numeric array in the metadata. Quant value of element i of block b: (7*b + 3*i) % 61 - 30."""
    def s(x): b = x.encode(); return struct.pack("<Q", len(b)) + b
    kv = [s("general.architecture") + struct.pack("<I", 8) + s("tiny"), s("tokenizer.ggml.tokens") + struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", 3) + s("a") + s("bb") + s("ccc"),
          s("tokenizer.ggml.scores") + struct.pack("<I", 9) + struct.pack("<I", 6) + struct.pack("<Q", 2) + struct.pack("<ff", 1.0, 2.0), s("general.alignment") + struct.pack("<I", 4) + struct.pack("<I", 32)]
    nblk = ne0 // 32 * ne1 * ne2; data = b""
    for b in range(nblk): data += struct.pack("<e", d) + struct.pack("<32b", *[(7 * b + 3 * i) % 61 - 30 for i in range(32)])
    hdr = b"GGUF" + struct.pack("<IQQ", 3, 1, len(kv)) + b"".join(kv) + s(name) + struct.pack("<I", 3) + struct.pack("<QQQ", ne0, ne1, ne2) + struct.pack("<IQ", 8, 0)
    hdr += b"\0" * ((32 - len(hdr) % 32) % 32)
    with open(path, "wb") as fh: fh.write(hdr + data)

def tiny_tokens(layers=48, ntok=None, sets=None):
    """Synthetic trace: tokens[t][l] = list of 8 expert ids; `sets` gives the id list per token (same for every layer)."""
    return [[list(sets[t]) for _ in range(layers)] for t in range(ntok or len(sets))]

def fam2(t):
    r = f.fam_s2_2(); t.check("s2_2 determinism: two runs give identical dicts", r == f.fam_s2_2())
    L = f.lut_ops(768, 2048, books=1)
    t.check("s2_2 unit: lut_ops 768x2048 one book = hand-computed 3145728 / 1048576 / 393216 ops", (L["base"], L["table"], L["lookups"]) == (3145728, 1048576, 393216), str(L))
    t.check("s2_2 unit: min rows for 2x fewer ops = 2048/3 (one book) and 2048 (two books); at 2048 rows two books give exactly 2.0x", abs(f.min_rows_for_op_ratio(2.0, 1) - 2048 / 3) < 1e-9 and f.min_rows_for_op_ratio(2.0, 2) == 2048 and abs(f.lut_ops(2048, 2048, 2)["ratio"] - 2.0) < 1e-12)
    t.check("s2_2 unit: asymptote 8x needs one book; two books cannot reach 4x or more (None)", f.min_rows_for_op_ratio(8.0, 1) is None and f.min_rows_for_op_ratio(4.0, 2) is None and f.min_rows_for_op_ratio(3.9, 2) is not None)
    t.check("s2_2 unit: table 128 KiB for 2048 columns, fits a 32 KiB L1 only up to 512 columns", r["table_bytes_cols2048_one_book"] == 131072 and r["table_fits_32KiB_L1_up_to_cols"] == 512)
    t.check("s2_2 unit: bits per weight 1.0 (one 256 book), 2.0 (two books), 2.4 with 0.4 of scales", r["bpw"] == {"1x8_index_only": 1.0, "2x8_index_only": 2.0, "2x8_with_0.4_scale": 2.4})
    t.check("s2_2 LUT GEMV equals the dense dot of the same codes (max abs diff < 1e-9) and counts 1024 table MACs, 96 lookups", r["lut_vs_dense_max_abs_diff"] < 1e-9 and r["lut_counters_tiny"] == {"table_macs": 1024, "lookups": 96}, str(r["lut_vs_dense_max_abs_diff"]))
    rng = random.Random(1); cbs = [[[rng.gauss(0, 1) for _ in range(8)] for _ in range(16)]]; codes = [[[rng.randrange(16)] for _ in range(4)] for _ in range(6)]; x = [rng.gauss(0, 1) for _ in range(32)]
    bad = [[w[:] for w in cbs[0]]]; bad[0][codes[0][0][0]][0] += 1.0; y1 = f.lut_gemv(codes, bad, x)[0]; y2 = f.dense_gemv(codes, cbs, x)
    t.check("s2_2 control: a corrupted codeword makes the LUT result differ from the reference (the equality check can fail)", max(abs(a - b) for a, b in zip(y1, y2)) > 1e-3)
    between(t, "s2_2 measured: op reduction of a 2-codebook LUT kernel on a 768x2048 expert matrix (claim: about 8x)", r["op_ratio"]["gate_up_768x2048_books2"], 1.05, 1.15)
    between(t, "s2_2 measured: whole-expert op reduction, 1 codebook, tables not shared", r["op_ratio_whole_expert_gate_up_down_own_tables"]["books1"], 2.4, 2.7)
    between(t, "s2_2 measured: op reduction when 20 gate/up matrices share one table (shared per-layer codebook, 1 book)", r["op_ratio_shared_table_gate_up_20_consumers"]["books1"], 6.5, 7.5)
    between(t, "s2_2 measured: byte cut 4.3 to 2.4 bpw (claim: about 3x)", r["byte_cut_4.3_to_2.4"], 1.78, 1.80)
    g8 = r["throughput"]["8GB"]["bpw2.4_kernel1.0"]
    between(t, "s2_2 measured: 8 GB serial gain of 2.4 bpw with a kernel as fast as the dot kernel (screen: 1.5x)", g8["gain_serial"], 1.45, 1.70)
    between(t, "s2_2 measured: 8 GB overlap gain of 2.4 bpw (screen: 1.1x)", g8["gain_overlap"], 1.05, 1.15)
    for ram in ("16GB", "32GB"): between(t, f"s2_2 measured: {ram} overlap gain from bytes alone (compute-bound, screen: 1.0x)", r["throughput"][ram]["bpw2.4_kernel1.0"]["gain_overlap"], 0.99, 1.01)
    t.check("s2_2 measured: a kernel 0.7x as fast as the dot kernel turns the overlap gain into a loss at every RAM size", all(r["throughput"][k]["bpw2.4_kernel0.7"]["gain_overlap"] < 1.0 for k in r["throughput"]))
    between(t, "s2_2 measured: tokens/s ceiling of the claim (8 x 4.57) vs what a 1.5x kernel gives at 16 GB", r["throughput"]["16GB"]["bpw2.4_kernel1.5"]["overlap"], 6.8, 6.9)
    t.check("s2_2 sanity: hits in [0,1], smaller experts never lower the 8 GB hit rate (4.3 < 2.4 bpw)", all(0 <= r["throughput"][k][c]["hit"] <= 1 for k in r["throughput"] for c in r["throughput"][k] if isinstance(r["throughput"][k][c], dict) and "hit" in r["throughput"][k][c]) and f.lever(2.4, 8)["hit"] >= f.lever(4.3, 8)["hit"])

def fam3(t):
    r = f.fam_s2_3(); t.check("s2_3 determinism: two runs give identical dicts", r == f.fam_s2_3())
    t.check("s2_3 unit: kv_bytes_per_token(2 layers, 1 head, 4 dim) = 2*2*1*4*2 = 32", f.kv_bytes_per_token(2, 1, 4) == 32)
    t.check("s2_3 unit: 30B shape (48 layers, 4 KV heads, 128 dim, fp16) = 98304 bytes per token, 49.152 MB for 500 tokens", r["kv_bytes_per_token_30B_shape"] == 98304 and r["kv_MB_500_tokens_30B_shape"] == 49.152)
    t.check("s2_3 unit: small MoE shape (16 layers, 16 KV heads, 128 dim) = 131072 bytes per token", r["kv_bytes_per_token_small_moe_16layers_16kvheads_128dim"] == 131072)
    between(t, "s2_3 measured: the claimed 1 GB per 500 tokens over the 3x-scaled estimate (claim is too large)", r["claim_over_estimate"]["claim1GB_vs_est_x3"], 6.0, 7.5)
    between(t, "s2_3 measured: prefill compute at the 55 GFLOP/s decode rate (the family's 109 s)", r["prefill_compute_s"]["at_decode_rate_55"], 108, 110)
    between(t, "s2_3 measured: prefill compute at the 311 GFLOP/s batched rate", r["prefill_compute_s"]["at_prefill_rate_311"], 19, 19.6)
    between(t, "s2_3 measured: cold TTFT on the best path (max of compute and expert I/O), seconds (screen: 25-35)", r["cold_ttft_s_best_path_overlap"], 21, 24)
    between(t, "s2_3 measured: TTFT ratio on an exact hit, KV 1x..3x (screen: 40-50x)", min(v["ttft_ratio"] for v in r["hit_case"].values()), 45, 52)
    between(t, "s2_3 measured: whole-request gain for 200 tokens at 2.18 tok/s on a hit (screen: 1.3-1.6x, which assumed a 30 s prefill and 4 tok/s)", r["hit_case"]["kv_x1"]["request_ratio_200_tokens_at_2p18"], 1.2, 1.3)
    t.check("s2_3 sanity: KV load time grows with KV size, hit TTFT ratio shrinks with it", r["hit_case"]["kv_x1"]["load_s"] < r["hit_case"]["kv_x3"]["load_s"] and r["hit_case"]["kv_x1"]["ttft_ratio"] > r["hit_case"]["kv_x3"]["ttft_ratio"])
    cp = r["pinned_hit_cross_prompt_vs_in_sample"]
    t.check("s2_3 sanity: pinned hit rates in [0,1] and grow with the pinned size for both cross-prompt and in-sample", all(0 <= c[k]["cross_prompt"] <= 1 and 0 <= c[k]["in_sample"] <= 1 for c in cp.values() for k in c) and all(c["1.0fp"]["cross_prompt"] <= c["2.1fp"]["cross_prompt"] <= c["4.6fp"]["cross_prompt"] <= c["7.5fp"]["cross_prompt"] for c in cp.values()))
    t.check("s2_3 control: a hot list learned on other prompts is strictly worse than one learned on the prompt itself", all(c[k]["cross_prompt"] < c[k]["in_sample"] for c in cp.values() for k in c))
    between(t, "s2_3 measured: cross-prompt hot-list hit at 1 footprint, all prompts (family text: 5-21%)", min(c["1.0fp"]["cross_prompt"] for c in cp.values()), 0.08, 0.12)
    between(t, "s2_3 measured: cross-prompt hot-list hit at 1 footprint, best prompt", max(c["1.0fp"]["cross_prompt"] for c in cp.values()), 0.15, 0.20)
    between(t, "s2_3 measured: cross-prompt hit at 2.1 footprints, mean (the 8 GB cache size): above the family's 21%", sum(c["2.1fp"]["cross_prompt"] for c in cp.values()) / 3, 0.25, 0.37)
    t.check("s2_3 sanity: decode tokens/s change is exactly 1.0", r["decode_tokps_change"] == 1.0)

def fam4(t):
    r = f.fam_s2_4()
    P = f.predictions("qwen3-30b_db", True); a = f.prefetch_sim("qwen3-30b_db", 2.1, 4, P); b = f.prefetch_sim("qwen3-30b_db", 2.1, 4, P)
    t.check("s2_4 determinism: two simulations of the same configuration are identical", a == b)
    cnt = [[[0] * f.E128 for _ in range(f.E128)] for _ in range(f.NL - 1)]; cnt[0][5][9] = 3; cnt[0][5][7] = 3; cnt[0][6][7] = 1
    t.check("s2_4 unit: predictor scores the sum of co-occurrence rows (hand: expert 7 scores 4, expert 9 scores 3)", f.predict_next(cnt, 0, [5, 6], 2) == [7, 9])
    t.check("s2_4 unit: ties break to the lower expert id", f.predict_next(cnt, 1, [1, 2], 3) == [0, 1, 2])
    toks = tiny_tokens(sets=[range(8)] * 2); st = f.TOK_BYTES / f.FOOT / f.RATE_HI; comp = f.COMP_S
    one = f.prefetch_sim(None, 1.0, 0, tokens=toks[:1]); two = f.prefetch_sim(None, 1.0, 0, tokens=toks)
    t.check("s2_4 unit: one all-miss token takes 48*8 reads + compute (hand-computed)", abs(1 / one["tokps"] - (48 * 8 * st + comp)) < 1e-9, str(1 / one["tokps"]))
    t.check("s2_4 unit: a repeated token is all hits, so two tokens cost one token's reads plus two computes", abs(2 / two["tokps"] - (48 * 8 * st + 2 * comp)) < 1e-9)
    t.check("s2_4 sanity: precision in [0,1]; held-out never above in-sample; top-4 precision at least top-8", all(0 <= r["precision_held_out"][k][m] <= r["precision_in_sample"][k][m] <= 1 for k in r["precision_held_out"] for m in ("top4", "top8")) and all(r["precision_held_out"][k]["top4"] >= r["precision_held_out"][k]["top8"] for k in r["precision_held_out"]))
    for k, (lo, hi) in {"0b_db": (0.40, 0.46), "story": (0.28, 0.33), "_code": (0.34, 0.40)}.items(): between(t, f"s2_4 measured: held-out top-8 precision on {k} (screen: 0.43 / 0.31 / 0.37)", r["precision_held_out"][k]["top8"], lo, hi)
    between(t, "s2_4 measured: in-sample top-8 precision, lowest trace (the idea's 0.55)", min(v["top8"] for v in r["precision_in_sample"].values()), 0.55, 0.60)
    between(t, "s2_4 measured: previous-token persistence, all traces (screen: 0.42-0.46)", min(r["persistence"].values()), 0.41, 0.43); between(t, "s2_4 measured: previous-token persistence, max", max(r["persistence"].values()), 0.44, 0.47)
    s21 = r["sim"]["2.1fp"]
    between(t, "s2_4 measured: top-4 prefetch speedup at 2.1 footprints, min over traces (screen: 1.04-1.07)", min(v["top4"]["speedup"] for v in s21.values()), 1.02, 1.06)
    between(t, "s2_4 measured: top-4 prefetch speedup at 2.1 footprints, max over traces", max(v["top4"]["speedup"] for v in s21.values()), 1.06, 1.10)
    between(t, "s2_4 measured: top-8 prefetch speedup at 2.1 footprints, min (screen: 0.92-1.02)", min(v["top8"]["speedup"] for v in s21.values()), 0.90, 0.97)
    t.check("s2_4 measured: kill criterion met - every held-out speedup at 2.1 footprints is below 1.15", all(v[m]["speedup"] < 1.15 for v in s21.values() for m in ("top4", "top8")))
    t.check("s2_4 measured: at 7.5 footprints prefetch changes tokens/s by under 1% (compute ceiling binds)", all(abs(v[m]["speedup"] - 1) < 0.01 for v in r["sim"]["7.5fp"].values() for m in ("top4", "top8")))
    t.check("s2_4 sanity: no-prefetch tokens/s grows with the cache size and stays under the 4.57 ceiling", all(r["sim"]["2.1fp"][k]["no_prefetch_tokps"] < r["sim"]["4.0fp"][k]["no_prefetch_tokps"] < r["sim"]["7.5fp"][k]["no_prefetch_tokps"] < 4.57 for k in s21))
    t.check("s2_4 measured: prefetching reads more bytes per token than not prefetching (wasted reads), every trace at 2.1 footprints", all(v[m]["gb_tok"] > v["no_prefetch_gb_tok"] for v in s21.values() for m in ("top4", "top8")))
    between(t, "s2_4 measured: even an oracle predictor with a one-layer lead gains at most this at 2.1 footprints (top-8; screen's perfect-overlap bound 1.9x)", max(v["oracle_top8_speedup"] for v in s21.values()), 1.5, 1.7)
    t.check("s2_4 control: the oracle beats the learned predictor on every trace", all(v["oracle_top4_speedup"] > v["top4"]["speedup"] and v["oracle_top8_speedup"] > v["top8"]["speedup"] for v in s21.values()))
    rng = random.Random(0); Prand = [[rng.sample(range(f.E128), 8) for _ in range(f.NL - 1)] for _ in range(len(f.cs.tr("qwen3-30b_db")))]; sr = f.prefetch_sim("qwen3-30b_db", 2.1, 4, Prand)
    t.check("s2_4 control: a random predictor gives no speedup (it must be able to lose) and is worse than the learned one", sr["tokps"] / f.prefetch_sim("qwen3-30b_db", 2.1, 0)["tokps"] < 1.0 and sr["tokps"] < a["tokps"], str(sr))
    t.check("s2_4 sanity: the one-layer lead (4.6 ms) is shorter than reading one layer's experts (8.7 ms)", r["layer_lead_ms"] < r["one_layer_read_ms"])

def fam5(t):
    r = f.fam_s2_5(); t.check("s2_5 determinism: two runs give identical dicts", r == f.fam_s2_5())
    t.check("s2_5 unit: rank-16 basis at 4.3 bpw = 16 x 2.7 MB x 48 layers = 2.0736 GB", abs(f.basis_resident_gb(16) - 2.0736) < 1e-9)
    ev, vec = f.jacobi_eigh([[2.0, 1.0], [1.0, 2.0]])
    t.check("s2_5 unit: Jacobi eigenvalues of [[2,1],[1,2]] are 3 and 1 with orthonormal eigenvectors", abs(ev[0] - 3) < 1e-9 and abs(ev[1] - 1) < 1e-9 and abs(f.dot(vec[0], vec[1])) < 1e-9 and abs(f.dot(vec[0], vec[0]) - 1) < 1e-9)
    rng = random.Random(4); D = 120; B = [[rng.gauss(0, 1) for _ in range(D)] for _ in range(3)]; m = [rng.gauss(0, 1) for _ in range(D)]
    co = [[rng.gauss(0, 1) for _ in range(3)] for _ in range(10)]; X = [[m[i] + sum(co[e][j] * B[j][i] for j in range(3)) for i in range(D)] for e in range(10)]
    a = f.shared_basis_analysis(X, [0, 3], lambda v: v); pr = a["per_rank"]
    t.check("s2_5 unit: experts built from mean + 3 shared basis vectors are captured fully at rank 3 and not at rank 0", pr[3]["energy_captured_of_centred"] > 0.999999 and pr[0]["energy_captured_of_centred"] < 1e-9 and pr[3]["rel_err_lowrank_only"] < 1e-6)
    Y = [[rng.gauss(0, 1) for _ in range(D)] for _ in range(12)]; b = f.shared_basis_analysis(Y, [3], lambda v: v)
    t.check("s2_5 control: independent random experts are NOT captured by 3 shared components (captured share below 0.5 of the centred energy)", b["per_rank"][3]["energy_captured_of_centred"] < 0.5, str(b["per_rank"][3]))
    between(t, "s2_5 measured: residual bpw the claim needs for 0.36 GB per token (a '2-3 bit residual' is not this)", r["implied_residual_bpw_for_claim"], 1.18, 1.21)
    th = r["throughput"]
    between(t, "s2_5 measured: 8 GB serial gain of a free-basis 2.5 bpw residual layout (screen: 1.3x at 1.7x bytes)", th["8GB"]["resid2.5_extraflop0.0_rank0"]["gain_serial"], 1.4, 1.65)
    between(t, "s2_5 measured: 8 GB overlap gain of the same layout (screen: 1.1x)", th["8GB"]["resid2.5_extraflop0.0_rank0"]["gain_overlap"], 1.05, 1.15)
    between(t, "s2_5 measured: 32 GB overlap gain (screen: 1.0x or below)", th["32GB"]["resid2.5_extraflop0.0_rank0"]["gain_overlap"], 0.99, 1.01)
    t.check("s2_5 measured: with 25% extra FLOPs for the basis matmuls the overlap gain is below 1 at every RAM size", all(th[k]["resid2.5_extraflop0.25_rank0"]["gain_overlap"] < 1.0 for k in th))
    t.check("s2_5 measured: a resident basis eats the cache - more rank never raises the 8 GB gain (rank 0 >= 4 >= 16)", th["8GB"]["resid2.5_extraflop0.0_rank0"]["gain_serial"] >= th["8GB"]["resid2.5_extraflop0.0_rank4"]["gain_serial"] >= th["8GB"]["resid2.5_extraflop0.0_rank16"]["gain_serial"])
    t.check("s2_5 sanity: modelled tokens/s never exceeds the 4.57 compute ceiling (the claimed 6.6 is impossible)", all(f.lever(b, ram)["overlap"] <= 4.5701 for b in (1.2, 2.0, 2.5, 3.0) for ram in (8, 16, 32)))
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "tiny.gguf"); write_tiny_gguf(p, "blk.0.ffn_up_exps.weight", 64, 2, 3); g = f.read_gguf(p)
        info = g["tensors"]["blk.0.ffn_up_exps.weight"]; mat = f.read_q8_0(p, info, 1, 1, 1, 32, 32)[0]; blk = (1 * 2 + 1) * 2 + 1   # expert 1, row 1, second block: index (e*ne1 + r)*2 + 1 = 7
        t.check("s2_5 unit: GGUF reader skips string/number arrays and finds the tensor (dims, Q8_0 type)", g["version"] == 3 and info[0] == (64, 2, 3) and info[1] == 8 and g["meta"]["tokenizer.ggml.tokens"] == ("array", 3))
        t.check("s2_5 unit: Q8_0 dequantisation of expert 1, row 1, block 7 = 0.5 * ((7*7 + 3*i) % 61 - 30)", mat == [0.5 * ((7 * 7 + 3 * i) % 61 - 30) for i in range(32)] and blk == 7)
    if f.probe_path() and "weights_probe" in r and r["weights_probe"].get("ran"):
        wp = r["weights_probe"]
        between(t, "s2_5 weights: share of centred energy kept by 16 shared components of 64 experts (chance level 16/63 = 0.25)", wp["per_rank"][16]["energy_captured_of_centred"], 0.20, 0.40)
        between(t, "s2_5 weights: mean share of energy (the cheap shared base)", wp["mean_share_of_energy"], 0.0, 0.1)
    else: t.check("s2_5 weights: probe skipped because PUBTECH_GGUF is unset (nothing faked)", r["weights_probe"]["ran"] is False)

def fam6(t):
    r = f.fam_s2_6(); t.check("s2_6 determinism: two runs give identical dicts", r == f.fam_s2_6())
    t.check("s2_6 unit: expected Jaccard of two random 2-subsets of 4 = 7/18", abs(f.expected_jaccard_random(4, 2) - 7 / 18) < 1e-12)
    same = tiny_tokens(ntok=3, sets=[range(8)] * 3); alt = [[list(range(8)) if l % 2 == 0 else list(range(8, 16)) for l in range(48)] for _ in range(3)]
    t.check("s2_6 unit: identical neighbouring layers give Jaccard 1, disjoint neighbours give 0", all(abs(v - 1) < 1e-12 for v in f.layer_jaccard(None, same)) and all(v == 0 for v in f.layer_jaccard(None, alt)))
    t.check("s2_6 unit: static skip drops exactly round(48*frac) distinct layers inside 1..46; dynamic stream has the right length", len(cs.skipped(0.5)) == 24 and all(1 <= x <= 46 for x in cs.skipped(0.5)) and len(f.dynamic_skip_stream("qwen3-30b_db", 0.25)) == len(f.cs.tr("qwen3-30b_db")) * 36 * 8)
    between(t, "s2_6 measured: chance Jaccard of two random 8-of-128 sets", r["random_expected_jaccard"], 0.03, 0.04)
    jt = r["adjacent_layer_jaccard"]
    t.check("s2_6 measured: no adjacent layer pair has Jaccard above 0.3 (screen expected low overlap); the highest pair is under 0.1", all(v["pairs_above_0p3"] == 0 and v["max"] < 0.1 for v in jt.values()))
    between(t, "s2_6 measured: mean adjacent-layer Jaccard over traces (at chance level)", sum(v["mean"] for v in jt.values()) / 3, 0.030, 0.036)
    t.check("s2_6 control: the redundancy proxy can fire - identical layers reach 1.0 and the measured pairs do not", max(f.layer_jaccard(None, same)) == 1.0 and max(v["max"] for v in jt.values()) < 0.5)
    th = r["throughput_gain_vs_no_skip"]
    for ram in ("16GB", "32GB"): between(t, f"s2_6 measured: {ram} overlap gain at 50% skip equals the compute ratio (static)", th[ram]["skip50"]["static"]["gain_overlap"], 1.99, 2.01)
    between(t, "s2_6 measured: 67% skip (16 of 48 layers kept) overlap gain at 32 GB (claim 3.3x; arithmetic 1/kept = 3.0)", th["32GB"]["skip67"]["static"]["gain_overlap"], 2.99, 3.06)
    between(t, "s2_6 measured: 8 GB serial gain at 67% skip, dynamic per-token layers (screen: 3-4.6x; static sets are optimistic)", th["8GB"]["skip67"]["dynamic"]["gain_serial"], 2.7, 3.0)
    between(t, "s2_6 measured: 8 GB serial gain at 67% skip, static layer set", th["8GB"]["skip67"]["static"]["gain_serial"], 4.5, 5.2)
    t.check("s2_6 sanity: gains grow with the skip fraction and dynamic never beats static on serial I/O at any size", all(th[k]["skip25"][m]["gain_serial"] < th[k]["skip50"][m]["gain_serial"] < th[k]["skip67"][m]["gain_serial"] for k in th for m in ("static", "dynamic")) and all(th[k][s]["dynamic"]["gain_serial"] <= th[k][s]["static"]["gain_serial"] + 1e-9 for k in th for s in th[k]))
    between(t, "s2_6 measured: analytic model 'exit' lever at 33% kept, 8 GB (screen: 4.65x)", r["analytic_model_exit_gain_likely"]["keep0.33"]["8GB"], 4.6, 4.7)
    between(t, "s2_6 measured: realistic 25% skip, 16 GB overlap gain (screen: 1.1-1.4x)", th["16GB"]["skip25"]["static"]["gain_overlap"], 1.32, 1.34)
    zero = [f.skip_tokps(n, 0.0, "static", 8) for n in f.TRACES]; zdyn = [f.skip_tokps(n, 0.0, "dynamic", 8) for n in f.TRACES]
    t.check("s2_6 control: skipping nothing gives gain exactly 1 (static and dynamic agree)", zero == zdyn)

def fam7(t):
    r = f.fam_s2_7(); t.check("s2_7 determinism: two runs give identical dicts", r == f.fam_s2_7())
    A = list(range(8)); B_ = list(range(8, 16)); tk = tiny_tokens(ntok=5, sets=[A, A, B_, B_, B_])
    p, l = f.pin_vs_lru(None, 2, 8, 3, tokens=tk)
    t.check("s2_7 unit: set A pinned, B arrives - pin hits 0, LRU (8 entries) misses once then hits 2 of 3 tokens = 16/24", p == 0.0 and abs(l - 16 / 24) < 1e-12, f"{p} {l}")
    p, l = f.pin_vs_lru(None, 2, 8, 2, tokens=tiny_tokens(ntok=4, sets=[A] * 4))
    t.check("s2_7 unit: an unchanged working set gives pin = LRU = 1", p == 1.0 and l == 1.0)
    t.check("s2_7 control: a zero-size pin and a zero-size LRU both hit nothing", f.pin_vs_lru(None, 2, 0, 2, tokens=tiny_tokens(ntok=4, sets=[A] * 4)) == (0.0, 0.0))
    bt = r["by_trace"]
    for k, (lo, hi) in {"0b_db": (0.79, 0.84), "story": (0.85, 0.89), "_code": (0.80, 0.84)}.items(): between(t, f"s2_7 measured: pin-from-first-64-tokens hit on {k}, 29% pinned, next 20 tokens (screen: 0.81 / 0.87 / 0.82)", bt[k]["W64_29%"]["pin"], lo, hi)
    t.check("s2_7 sanity: all hit rates in [0,1] and pinning more never lowers the pin hit (15% <= 29% <= 50%)", all(0 <= c[k][m] <= 1 for c in bt.values() for k in c for m in ("pin", "lru")) and all(c["W64_15%"]["pin"] <= c["W64_29%"]["pin"] <= c["W64_50%"]["pin"] for c in bt.values()))
    d = [bt[k]["W64_29%"]["diff"] for k in bt]
    between(t, "s2_7 measured: mean pin-minus-LRU over the 3 traces, 29%, W=64, next 20 tokens (screen: within +/-0.03)", sum(d) / 3, -0.02, 0.05)
    t.check("s2_7 measured: kill criterion 'pin - LRU below +0.05 on all 3 traces' holds on 2 of 3 and fails on code (+0.063 over a 20-token window)", sum(x < 0.05 for x in d) == 2 and max(d) > 0.05, str(d))
    t.check("s2_7 measured: over the rest of the trace LRU beats the static pin on every prompt (LRU adapts)", all(c["W64_29%_rest_of_trace"]["diff"] < -0.04 for c in bt.values()))
    t.check("s2_7 measured: a hot set learned on other prompts loses to LRU by more than 0.1 on every prompt", all(c["cross_prompt_29%_W64"]["diff"] < -0.1 for c in bt.values()))
    cur = r["lru_curve_footprints_hit"]; vals = [cur[k] for k in ("1.0", "2.1", "4.0", "7.5", "20.8")]
    t.check("s2_7 sanity: the LRU curve is non-decreasing in footprints and reaches 0.95 at 7.5", vals == sorted(vals) and cur["7.5"] >= 0.95)
    between(t, "s2_7 measured: best-path serial tokens/s at 16 GB (screen: 4.2)", r["best_path_by_ram"]["16GB"]["serial"], 4.1, 4.3)
    between(t, "s2_7 measured: best-path overlap tokens/s at 32 GB equals the 4.57 ceiling", r["best_path_by_ram"]["32GB"]["overlap"], 4.56, 4.58)
    between(t, "s2_7 measured: claimed gain range against a 2-4 tok/s baseline, lower end", r["claimed_gain_vs_baseline_2_4tokps"][0], 1.1, 1.2)

def fam8(t):
    r = f.fam_s2_8(); t.check("s2_8 determinism: two runs give identical dicts", r == f.fam_s2_8())
    t.check("s2_8 unit: uniform 2-bit quantiser reproduces the 4 levels 0..3 exactly and a constant block", f.quant_uniform([0.0, 1.0, 2.0, 3.0], 4, 2) == [0.0, 1.0, 2.0, 3.0] and f.quant_uniform([5.0] * 8, 4, 2) == [5.0] * 8)
    t.check("s2_8 unit: rel_err of identical vectors is 0 and of x vs 0 is 1", f.rel_err([1.0, 2.0], [1.0, 2.0]) == 0.0 and f.rel_err([3.0, 4.0], [0.0, 0.0]) == 1.0)
    sym = ([-1 + 2 * k / 15 for k in range(16)]) * 16; ref = [min((-1.0, -1 / 3, 1 / 3, 1.0), key=lambda L: abs(L - v)) for v in sym]
    t.check("s2_8 unit: Q2_K-like quantiser on symmetric 16-value blocks snaps every value to the 4 hand-computed levels {-1, -1/3, 1/3, 1} (rel err about 0.30)", max(abs(a - b) for a, b in zip(f.quant_q2k_like(sym), ref)) < 1e-9 and 0.29 < f.rel_err(sym, ref) < 0.31, str(f.rel_err(sym, ref)))
    cents = [[0.0, 0.0], [1.0, 1.0]]; codes, err = f.assign([[0.9, 0.8]], cents)
    t.check("s2_8 unit: nearest centroid of (0.9, 0.8) is index 1 with squared error 0.01 + 0.04", codes == [1] and abs(err - 0.05) < 1e-12)
    rng = random.Random(2); bl = [[rng.gauss(0, 1) for _ in range(8)] for _ in range(50)]; ce = [[rng.gauss(0, 1) for _ in range(8)] for _ in range(20)]
    t.check("s2_8 unit: the unrolled 8-dim assignment equals brute-force argmin", f.assign(bl, ce)[0] == [min(range(20), key=lambda i: f.dist_sq(x, ce[i])) for x in bl])
    km = f.kmeans([[0.0], [0.1], [10.0], [10.1]], 2, 6, 0)
    t.check("s2_8 unit: 1-D k-means finds the two obvious clusters (0.05 and 10.05)", sorted(round(c[0], 6) for c in km) == [0.05, 10.05], str(km))
    g = random.Random(9); tr = [[g.gauss(0, 1) for _ in range(8)] for _ in range(2500)]; te = [[g.gauss(0, 1) for _ in range(8)] for _ in range(1500)]
    m1 = f.vq_eval(tr, te, 64, 5, stages=2); m0 = f.vq_eval(tr, te, 64, 0, stages=1)
    between(t, "s2_8 measured: held-out normalised MSE of an 8-dim, 64-word (0.75 bpw) VQ on Gaussian data (Shannon bound 0.354)", m1[0], 0.354 * 0.95, 0.65)
    t.check("s2_8 sanity: a second additive stage lowers the error; trained codebooks beat the random ones (control that can fail)", m1[1] < m1[0] < m0[0], f"{m1} {m0}")
    rng = random.Random(5); W = [[rng.gauss(0, 1) for _ in range(16)] for _ in range(8)]; Wq = [[v + rng.gauss(0, 0.3) for v in row] for row in W]; num = den = 0.0
    for _ in range(3000):
        x = [rng.gauss(0, 1) for _ in range(16)]; y = [f.dot(r, x) for r in W]; yq = [f.dot(r, x) for r in Wq]; num += sum((a - b) ** 2 for a, b in zip(y, yq)); den += sum(a * a for a in y)
    fro = math.sqrt(sum((a - b) ** 2 for r, q in zip(W, Wq) for a, b in zip(r, q)) / sum(a * a for r in W for a in r))
    between(t, "s2_8 unit: output error for Gaussian x equals the weight Frobenius error (ratio)", math.sqrt(num / den) / fro, 0.95, 1.05)
    t.check("s2_8 measured: an 8-bit index per 8 weights is 1.0 bpw; the codebook adds under 0.03 bpw; the claimed 1.3 is not derived (0.3 bpw unexplained)", r["bpw_index_only_256_entries"] == 1.0 and r["codebook_overhead_bpw_per_1.5M_weights"] < 0.03 and r["bpw_claimed"] - r["bpw_index_only_256_entries"] > 0.29)
    th = r["throughput"]
    between(t, "s2_8 measured: 8 GB serial gain at the claimed 1.3 bpw (screen: 2.0x)", th["8GB"]["bpw1.3"]["gain_serial"], 1.9, 2.1)
    between(t, "s2_8 measured: 8 GB overlap gain at 1.3 bpw (screen: 1.1x)", th["8GB"]["bpw1.3"]["gain_overlap"], 1.05, 1.15)
    between(t, "s2_8 measured: 16 GB serial gain at 1.3 bpw (screen: 1.05x; cold-start misses of a 250-token trace make our baseline hit 0.95)", th["16GB"]["bpw1.3"]["gain_serial"], 1.05, 1.12)
    t.check("s2_8 sanity: 8 GB serial gain falls as bits per weight rise (1.0 > 1.3 > 2.0 > 2.4 > 2.625)", [th["8GB"][f"bpw{b}"]["gain_serial"] for b in (1.0, 1.3, 2.0, 2.4, 2.625)] == sorted([th["8GB"][f"bpw{b}"]["gain_serial"] for b in (1.0, 1.3, 2.0, 2.4, 2.625)], reverse=True))
    if f.probe_path() and r["weights_probe"].get("ran"):
        w = r["weights_probe"]
        between(t, "s2_8 weights: relative error of 8-dim 256-word VQ (1.0 bpw) on held-out experts (screen expected 0.45-0.6)", w["vq_1x8_256_rel_err"], 0.45, 0.65)
        t.check("s2_8 weights: kill criterion - the 256-entry VQ error is above the Q2_K-like error", w["vq_1x8_256_rel_err"] > w["q2k_like_rel_err"])
        between(t, "s2_8 weights: two additive 256-word codebooks (2.0 bpw) vs the Q2_K-like scalar quantiser at 2.625 bpw, error ratio", w["vq_2x8_256_rel_err"] / w["q2k_like_rel_err"], 0.9, 1.15)
        between(t, "s2_8 weights: Gaussian-x output error vs weight error of the same matrix, ratio", w["q2k_like_output_rel_err_gaussian_x"] / w["q2k_like_weight_rel_err_same_matrix"], 0.9, 1.1)
    else: t.check("s2_8 weights: probe skipped because PUBTECH_GGUF is unset (nothing faked)", r["weights_probe"]["ran"] is False)

def fam9(t):
    r = f.fam_s2_9(); t.check("s2_9 determinism: two runs give identical dicts", r == f.fam_s2_9())
    one = [[list(range(8))], [list(range(8))]]
    t.check("s2_9 unit: 2 identical tokens, C=8: Q4 misses 8 of 16 accesses; base+first-4-enhancement misses 12 half-size entries -> 0.375 bytes/access (ratio 1.333)", abs(f.scheme_bytes(one, 8, "first4") - 0.375) < 1e-12)
    t.check("s2_9 unit: enhancement on all 8 gives 16 half-size misses -> 0.5 = the Q4 miss rate (ratio exactly 1)", abs(f.scheme_bytes(one, 8, "all8") - 0.5) < 1e-12)
    t.check("s2_9 unit: no enhancement (pure 2-bit base): 8 misses at half size = 0.25 bytes/access", abs(f.scheme_bytes(one, 8, "none") - 0.25) < 1e-12)
    rb = r["bytes_per_access_replay"]
    for k, (lo, hi) in {"0b_db": (1.45, 1.55), "story": (1.47, 1.57), "_code": (1.38, 1.48)}.items(): between(t, f"s2_9 measured: bytes ratio Q4/scheme at 2.2 footprints on {k} (screen: 1.50 / 1.52 / 1.43)", rb["k2.2"]["first4"][k]["ratio"], lo, hi)
    between(t, "s2_9 measured: max bytes ratio over traces at 4.4 footprints (screen: 1.72-1.88)", max(v["ratio"] for v in rb["k4.4"]["first4"].values()), 1.85, 1.92)
    between(t, "s2_9 measured: min bytes ratio over traces at 8 footprints (screen: 1.16-1.46)", min(v["ratio"] for v in rb["k8.0"]["first4"].values()), 1.14, 1.2)
    t.check("s2_9 measured: the claimed 2.4x bytes cut is never reached - max over every trace and footprint is below 2.0", max(v["ratio"] for k in rb for v in rb[k]["first4"].values()) < 2.0)
    t.check("s2_9 control: enhancing all 8 experts gives exactly 1.00x on every trace (no gain from the layout alone)", all(abs(v["ratio"] - 1.0) < 0.005 for v in rb["k2.2"]["all8"].values()))
    t.check("s2_9 control: enhancing the LAST four (lowest-ranked) experts is worse than the first four at every footprint and trace", all(rb[k]["last4"][n]["ratio"] < rb[k]["first4"][n]["ratio"] for k in rb for n in rb[k]["last4"]))
    t.check("s2_9 sanity: miss rates and scheme bytes per access are in [0,1]", all(0 <= v["q4"] <= 1 and 0 <= v["scheme"] <= 1 for k in rb for e in rb[k] for v in rb[k][e].values()))
    t.check("s2_9 measured: pure 2-bit base with no enhancement is the cheap lossy extreme (ratio above 3.5 on every trace at 2.2 footprints)", all(v["ratio"] > 3.5 for v in rb["k2.2"]["none"].values()))
    th = r["throughput_at_real_footprints"]
    between(t, "s2_9 measured: 8 GB serial gain at the real 8 GB footprint count (screen: 1.4-1.5x bytes, 1.0-1.3x tok/s)", th["8GB"]["gain_serial"], 1.1, 1.25)
    t.check("s2_9 measured: overlap gain is exactly 1.00 at 8, 16 and 32 GB (compute ceiling binds); 16/32 GB serial gain under 1.05", all(th[k]["gain_overlap"] == 1.0 for k in th) and th["16GB"]["gain_serial"] < 1.05 and th["32GB"]["gain_serial"] < 1.05)
    ro = r["rank_order_check"]
    t.check("s2_9 measured: indirect gate-rank check - hotness and persistence decrease monotonically with the recorded position (consistent with, not proof of, gate-rank order)", ro["mean_expert_hotness_by_position"] == sorted(ro["mean_expert_hotness_by_position"], reverse=True) and ro["persistence_by_position"] == sorted(ro["persistence_by_position"], reverse=True))
    if f.probe_path() and r["weights_probe"].get("ran"):
        w = r["weights_probe"]
        between(t, "s2_9 weights: relative error of a Q2_K-like base alone (what the non-top-4 experts would run at)", w["base_q2k_like_rel_err"], 0.25, 0.4)
        t.check("s2_9 weights: base + residual (5.25 bpw) has lower error than base alone but not lower than a plain 4-bit scalar quantiser (5.0 bpw)", w["base_plus_q2k_like_residual_rel_err"] < w["base_q2k_like_rel_err"] and w["base_plus_q2k_like_residual_rel_err"] > w["q4_like_rel_err"])
    else: t.check("s2_9 weights: probe skipped because PUBTECH_GGUF is unset (nothing faked)", r["weights_probe"]["ran"] is False)

def run():
    t = T("pubtech group D (screen 2 #2-#9)")
    for fn in (fam2, fam3, fam4, fam5, fam6, fam7, fam8, fam9): fn(t)
    return t
if __name__ == "__main__": sys.exit(0 if run().summary() else 1)
