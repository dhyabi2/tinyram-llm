"""Group F published-technique family tests (screen 2 families 18..25) on the three real Qwen3-30B router traces, closed-form arithmetic and, when present, the real 30B weights.
Every measured range below was derived from the run and is stated in the check name. Where the recomputation disagrees with the screen's claim, the check passes on the measured value
and the disagreement is written up in the group report."""
import json, math, os, random, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from harness import T
from pubtech import f_F as F

RAMS = (8, 16, 32)
def same(a, b): return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)
def inr(x, lo, hi): return x is not None and lo <= x <= hi


def run():
    t = T("Group F published techniques (screen 2 #18-#25)")
    R = {}
    for k, fam in F.FAMILIES.items(): R[k] = fam["run"]()

    # ---------------------------------------------------------------- registry
    counts = {"s2_18": 3, "s2_19": 6, "s2_20": 4, "s2_21": 5, "s2_22": 4, "s2_23": 6, "s2_24": 6, "s2_25": 3}
    t.check("registry: eight families s2_18..s2_25", sorted(F.FAMILIES) == sorted(counts))
    for k, n in counts.items():
        fam = F.FAMILIES[k]
        t.check(f"{k}: registry entry has idea, run, needs in {{local, hardware, model}} and {n} catalog method names", callable(fam["run"]) and fam["needs"] in ("local", "hardware", "model") and len(fam["methods"]) == n and all(isinstance(m, str) and m for m in fam["methods"]) and fam["idea"])
    t.check("registry: 37 catalog method rows in total", sum(len(f["methods"]) for f in F.FAMILIES.values()) == 37)
    for fn in (os.path.join(HERE, "..", "tools", "pubtech", "f_F.py"), os.path.abspath(__file__)):
        s = open(fn).read(); bad = ["Internal" + "_not_public", "divide" + "AndConcur", "divide" + "-and-" + "concur", "/Users/" + "mac/", "172." + "86.85.83", "root" + "@", "BEGIN " + "PRIVATE KEY"]
        t.check(f"privacy: {os.path.basename(fn)} has no private strings", not [b for b in bad if b in s])

    # ---------------------------------------------------------------- shared helpers
    t.check("serial_toks hand case: 1 s of reads + 0.5 s compute = 0.6667 tok/s", abs(F.serial_toks(0.0, 1.296, comp=0.5) - 1 / 1.5) < 1e-9)
    t.check("overlap_toks hand case: 0.5 s of reads vs 0.2 s compute = 2.0 tok/s", abs(F.overlap_toks(0.5, 1.296, comp=0.2) - 2.0) < 1e-9)
    t.check("overlap_toks hand case: compute-bound = 1/comp", abs(F.overlap_toks(1.0, 1.0, comp=0.25) - 4.0) < 1e-9)
    t.check("footprints: 8/16/32 GB = 2.07/7.58/20.84 (within 0.5%)", all(abs(F.footprints(r) - v) / v < 0.005 for r, v in zip(RAMS, (2.0718, 7.5826, 20.8419))))
    fp = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]; hh = [F.mean_hit(x) for x in fp]
    t.check("mean LRU hit: in [0,1] and monotone in cache size", all(0 <= h <= 1 for h in hh) and all(a <= b + 1e-12 for a, b in zip(hh, hh[1:])), str(hh))
    t.check("mean LRU hit: reproduces the vault curve at 2 footprints (0.549 +-0.03) and 7.5 (0.950 +-0.03)", abs(F.mean_hit(2.0) - 0.549) < 0.03 and abs(F.mean_hit(7.5) - 0.950) < 0.03, f"{F.mean_hit(2.0):.3f} {F.mean_hit(7.5):.3f}")
    t.check("pass_cost: serial >= overlap > 0 for k=1..16 at every RAM size", all(F.pass_cost(r, k)["serial"] >= F.pass_cost(r, k)["overlap"] > 0 for r in RAMS for k in range(1, 17)))
    t.check("misses per pass non-decreasing in k (union of more tokens); with an LRU of 2-21 footprints, grouping k=8 tokens does NOT cut misses per token (ratio to k=1 in [0.9,1.1], measured 0.97-1.02)", all(F.misses_per_pass(r, k + 1) >= F.misses_per_pass(r, k) - 1e-9 for r in RAMS for k in range(1, 8)) and all(0.9 <= F.misses_per_pass(r, 8) / 8 / F.misses_per_pass(r, 1) <= 1.1 for r in RAMS), str([F.misses_per_pass(r, 8) / 8 / F.misses_per_pass(r, 1) for r in RAMS]))
    t.check("cost_of_sizes: sum of singles and splitting above 16", abs(F.cost_of_sizes([1, 1], 8) - 2 * F.pass_cost(8, 1)["serial"]) < 1e-9 and abs(F.cost_of_sizes([20], 8) - (F.pass_cost(8, 16)["serial"] + F.pass_cost(8, 4)["serial"])) < 1e-9)

    # ================================================================ family 18
    r = R["s2_18"]
    t.check("s2_18: deterministic (two runs identical)", same(r, F.fam_s2_18()))
    t.check("s2_18: hit rates in [0,1] and monotone in RAM", all(0 <= r["hit"][x] <= 1 for x in RAMS) and r["hit"][8] <= r["hit"][16] <= r["hit"][32], str(r["hit"]))
    t.check("s2_18: every serial tok/s positive and every overlap tok/s <= the 4.5833 compute ceiling", all(0 < v for x in RAMS for v in r["serial"][x].values()) and all(0 < v <= 4.5834 for x in RAMS for v in r["overlap"][x].values()))
    t.check("s2_18: serial <= overlap at the same rate", all(r["serial"][x][n] <= r["overlap"][x][n] + 1e-4 for x in RAMS for n in r["serial"][x]))
    order = ["mmap_paging", "tcp_10gbe_1.1", "nvme_screen_2.4", "nvme_explicit_3.11", "rdma_tb5_10", "rdma_100gbe_11"]
    t.check("s2_18: serial tok/s monotone in read rate (mmap < 10GbE < NVMe 2.4 < NVMe 3.11 < TB5 < 100GbE) at every RAM size", all(all(r["serial"][x][a] <= r["serial"][x][b] + 1e-4 for a, b in zip(order, order[1:])) for x in RAMS))
    t.check("s2_18: closed form vs combination model (LRU table instead of traces) agree within 3% at 3.11 and 11 GB/s", all(abs(r["combo_model_serial"][x][k] / r["serial"][x][n] - 1) < 0.03 for x in RAMS for k, n in (("3.11", "nvme_explicit_3.11"), ("11.0", "rdma_100gbe_11"))), str(r["combo_model_serial"]))
    c = r["claim"]
    t.check("s2_18: claim arithmetic: 117.8 ms vs 540 ms per token = 4.58x on I/O only", abs(c["io_ms_peer_nocache"] - 117.8) < 0.2 and abs(c["io_ms_nvme24_nocache"] - 540) < 0.5 and abs(c["io_ratio"] - 4.583) < 0.01)
    t.check("s2_18: the claimed ~8 tok/s (pure I/O 8.49 tok/s) exceeds the 4.58 tok/s compute ceiling", c["exceeds_compute_ceiling"] and inr(c["pure_io_tok_s_at_11GBs"], 8.4, 8.6))
    t.check("s2_18: control: at 5 GB/s the pure-I/O rate no longer exceeds the ceiling (flag can be False)", not (5.0 / F.EXP_GB > 1 / F.COMP_DEC))
    g = r["gain_vs_nvme311"]
    t.check("s2_18: 8 GB serial gain vs local NVMe 3.11 in [1.40,1.60] (screen: 1.5-1.7x)", inr(g[8]["serial"], 1.40, 1.60), str(g[8]))
    t.check("s2_18: 8 GB serial gain vs the screen's 2.4 GB/s path in [1.55,1.80]", inr(r["gain_vs_nvme24"][8]["serial"], 1.55, 1.80), str(r["gain_vs_nvme24"][8]))
    t.check("s2_18: 16 GB serial gain in [1.03,1.12] and 32 GB in [1.02,1.10] (screen: 1.06x and 1.0x)", inr(g[16]["serial"], 1.03, 1.12) and inr(g[32]["serial"], 1.02, 1.10), f"{g[16]} {g[32]}")
    t.check("s2_18: overlap gain vs local NVMe 3.11 is exactly 1.0 at every RAM size (compute ceiling binds); vs 2.4 GB/s at 8 GB in [1.05,1.12]", all(g[x]["overlap"] == 1.0 for x in RAMS) and inr(r["gain_vs_nvme24"][8]["overlap"], 1.05, 1.12))
    t.check("s2_18: vs plain mmap at 8 GB: serial in [6.5,7.5], overlap in [7.0,8.2] (screen: 7.0x / 7.7x)", inr(r["gain_vs_mmap"][8]["serial"], 6.5, 7.5) and inr(r["gain_vs_mmap"][8]["overlap"], 7.0, 8.2), str(r["gain_vs_mmap"][8]))
    t.check("s2_18: control: a 10GbE-class peer (1.1 GB/s) is SLOWER than local NVMe at 8 GB", r["serial"][8]["tcp_10gbe_1.1"] < r["serial"][8]["nvme_explicit_3.11"] and r["overlap"][8]["tcp_10gbe_1.1"] < r["overlap"][8]["nvme_explicit_3.11"])
    t.check("s2_18: read rate needed to reach the compute ceiling under perfect overlap: 8 GB in [2.4,2.8] GB/s, 16 GB below 0.5", inr(r["rate_to_hit_ceiling_GBs"][8], 2.4, 2.8) and r["rate_to_hit_ceiling_GBs"][16] < 0.5, str(r["rate_to_hit_ceiling_GBs"]))
    t.check("s2_18: per-fetch latency cost is small: RDMA < 3 ms per token and 100 us TCP < 25 ms per token at 8 GB", r["fetch_latency_ms_per_token"][8]["rdma_10us"] < 3 and r["fetch_latency_ms_per_token"][8]["tcp_100us"] < 25)

    # ================================================================ family 19
    r = R["s2_19"]
    t.check("s2_19: deterministic (two runs identical)", same(r, F.fam_s2_19()))
    t.check("s2_19: bytes_bound hand cases: r=1 -> 1.0, r=4,p=0.55 -> 1.7021, r=inf -> 2.2222, p=0 -> 1.0", F.bytes_bound(1.0) == 1.0 and abs(F.bytes_bound(4.0) - 1 / 0.5875) < 1e-9 and abs(F.bytes_bound(1e12) - 1 / 0.45) < 1e-6 and F.bytes_bound(7.0, 0.0) == 1.0)
    t.check("s2_19: bits_saved / comp_ratio hand cases", F.bits_saved(0.5) == 1.0 and F.bits_saved(1.0) == 0.0 and F.bits_saved(1.5) == 0.0 and abs(F.comp_ratio(1.0, 4.5) - 4.5 / 3.5) < 1e-12 and F.comp_ratio(5.0) == float("inf"))
    cs_ = F.cos_and_ratio([2, 4, 6], [1, 2, 3]); co = F.cos_and_ratio([1, 0], [0, 1]); d45 = F.cos_and_ratio([1, 1], [1, 0])
    t.check("s2_19: cos_and_ratio hand cases: parallel -> (1,0), orthogonal -> (0,1), 45 degrees -> (0.7071, 0.7071)", abs(cs_[0] - 1) < 1e-12 and cs_[1] < 1e-7 and co == (0.0, 1.0) and abs(d45[0] - 0.5 ** 0.5) < 1e-12 and abs(d45[1] - 0.5 ** 0.5) < 1e-12)
    rng = random.Random(7); b = [rng.gauss(0, 1) for _ in range(20000)]; nz = [rng.gauss(0, 1) for _ in range(20000)]; a_ = [0.8 * x + 0.6 * y for x, y in zip(b, nz)]
    cc, rr = F.cos_and_ratio(a_, b); ci, ri = F.cos_and_ratio(nz, b)
    t.check("s2_19: control: the instrument detects a planted correlation (rho 0.8 -> ratio 0.60 +-0.02, 0.74 bit saved) and none in independent noise (ratio > 0.99)", abs(rr - 0.6) < 0.02 and abs(F.bits_saved(rr) - 0.737) < 0.05 and ri > 0.99, f"{rr:.3f} {ri:.4f}")
    import struct
    blk = struct.pack("<ee", 2.0, 1.0) + bytes([3] * 4 + [1] * 4 + [0] * 4) + bytes([0x21] * 128); deq = F.dequant_q4k(blk)
    t.check("s2_19: Q4_K dequantizer hand block (d=2, dmin=1, scales 3, mins 1, nibbles 1/2): 32x5, 32x11, 32x5, 32x11, 128x0", deq == [5.0] * 32 + [11.0] * 32 + [5.0] * 32 + [11.0] * 32 + [0.0] * 128, str(deq[:5]))
    raw = struct.pack("<ee", 1.0, 0.0) + bytes([1] * 4 + [0] * 4 + [1] * 4)[:12] + bytes(128)
    tmp = tempfile.NamedTemporaryFile(suffix=".gguf", delete=False); tmp.write(F.build_gguf([("blk.0.ffn_gate_exps.weight", [256, 1, 1], 12, raw), ("other", [8], 0, bytes(32))])); tmp.close()
    try:
        h = F.read_gguf(tmp.name); ti = h["tensors"]["blk.0.ffn_gate_exps.weight"]
        with open(tmp.name, "rb") as fh: fh.seek(h["data_start"] + ti["offset"]); back = fh.read(144)
        t.check("s2_19: GGUF writer/reader round trip: dims, type, alignment, data offset and bytes", ti["dims"] == [256, 1, 1] and ti["type"] == 12 and h["data_start"] % 32 == 0 and back == raw and h["tensors"]["other"]["type"] == 0 and h["version"] == 3)
        open(tmp.name, "wb").write(b"NOPE" + bytes(100))
        try: F.read_gguf(tmp.name); bad = False
        except ValueError: bad = True
        t.check("s2_19: control: a file without the GGUF magic is rejected", bad)
    finally: os.unlink(tmp.name)
    pi, ph = r["predictor_precision_in_sample"], r["predictor_precision_held_out"]
    t.check("s2_19: predictor precision in [0,1], held-out <= in-sample (the latter leaks)", 0 <= ph <= pi <= 1, f"{ph} {pi}")
    t.check("s2_19: held-out cross-layer precision in [0.45,0.62] and in-sample in [0.50,0.66] (screen: 0.55 in-prompt, 0.31-0.43 held out for the earlier predictor)", inr(ph, 0.45, 0.62) and inr(pi, 0.50, 0.66), f"{ph} {pi}")
    bt = r["bound_table_p055"]
    t.check("s2_19: bound table: r=1 -> 1.0, 1.9 -> 1.35 (+-0.01), 4 -> 1.70 (+-0.01), inf -> 2.22 (+-0.01); monotone in r", bt["1.0"] == 1.0 and inr(bt["1.9"], 1.34, 1.36) and inr(bt["4.0"], 1.69, 1.71) and inr(bt["1000000000.0"], 2.21, 2.23) and bt["1.0"] <= bt["1.05"] <= bt["1.9"] <= bt["4.0"])
    t.check("s2_19: held-out precision lowers every bound", all(r["bound_table_p_held_out"][k] <= r["bound_table_p055"][k] + 1e-9 for k in bt))
    ca = r["claim_arith"]
    t.check("s2_19: claim arithmetic: 1300/1.9 = 684 MB not 331 MB; 331 MB implies 3.93x; 3.93 x 2.2 = 8.65 (the 8.7x headline)", inr(ca["mb_if_ratio_1.9"], 684, 684.5) and inr(ca["ratio_implied_by_331mb"], 3.92, 3.93) and inr(ca["stack_8.7x_decomposition"], 8.6, 8.7))
    w = r["weights"]
    if w is None: t.check("s2_19: weights file absent: weight-level checks skipped (informational pass)", True)
    else:
        cl, sl = w["cross_layer_random"], w["same_layer_control"]
        t.check("s2_19: real weights: 192 cross-layer pairs, median |cos| in [0.001,0.012] (measured 0.0047), max < 0.03", cl["n"] == 192 and inr(cl["median_abs_cos"], 0.001, 0.012) and cl["max_abs_cos"] < 0.03, str(cl))
        t.check("s2_19: real weights: median residual/target RMS ratio >= 0.999 (kill threshold of the screen was > 0.85) and best-of-24 median >= 0.999", cl["median_ratio"] >= 0.999 and w["best_of_candidates_median_ratio"] >= 0.999 and cl["median_ratio"] > 0.85)
        t.check("s2_19: real weights: same-layer control and same-expert-id and trace-coactivation bases are equally uncorrelated (|cos| medians < 0.012, ratios >= 0.999)", sl["median_abs_cos"] < 0.012 and w["same_expert_id_prev_layer"]["median_abs_cos"] < 0.012 and w["coactivation_base"]["median_abs_cos"] < 0.012 and min(sl["median_ratio"], w["same_expert_id_prev_layer"]["median_ratio"], w["coactivation_base"]["median_ratio"]) >= 0.999)
        t.check("s2_19: real weights: bits saved per weight < 0.001 (screen expected < 0.05) so the residual compression factor r rounds to 1.0 and the bytes bound is 1.00x", w["saved_bits_median"] < 0.001 and w["saved_bits_best_of_candidates"] < 0.001 and r["residual_ratio_r_best_of_candidates"] < 1.001 and r["bytes_gain_bound_measured"] < 1.001, str(r.get("bytes_gain_bound_measured")))
        t.check("s2_19: real weights: tensor is Q4_K (type 12) and 32,768 weights sampled per expert", w["tensor_type"] == 12 and w["n_weights_per_expert"] == 32768)
        t.check("s2_19: coactivation_base returns a layer-10 expert id in 0..127, deterministically", (lambda e: F.coactivation_base(11, e) == F.coactivation_base(11, e) and 0 <= F.coactivation_base(11, e) < 128)(F.cs.tr(F.TRACES[0])[0][11][0]))

    # ================================================================ family 20
    r = R["s2_20"]
    t.check("s2_20: deterministic (two runs identical)", same(r, F.fam_s2_20()))
    t.check("s2_20: uniform_frac hand cases: 1 token of 8/128 = 0.0625, 64 tokens of 10/512 = 0.717 (+-0.001), 500 tokens -> 1", F.uniform_frac(1, 128, 8) == 0.0625 and abs(F.uniform_frac(64, 512, 10) - 0.71699) < 1e-3 and F.uniform_frac(500, 512, 10) > 0.9999)
    t.check("s2_20: a single decode token touches exactly 8/128 = 0.0625 of the experts on every trace", all(F.union_frac(tr, 1) == 0.0625 for tr in F.TRACES))
    tab = r["union_table"]; ns = sorted(tab)
    t.check("s2_20: measured union share is monotone in window length, <= 1 and <= n*0.0625, and below uniform routing for n >= 4", all(tab[a]["measured30"][k] <= tab[b]["measured30"][k] + 1e-9 for a, b in zip(ns, ns[1:]) for k in ("db", "story", "code")) and all(tab[n]["measured30"][k] <= min(1.0, n * 0.0625) + 1e-9 for n in ns for k in ("db", "story", "code")) and all(tab[n]["measured30"][k] <= tab[n]["uniform30"] for n in ns if n >= 4 for k in ("db", "story", "code")))
    t.check("s2_20: prefill_pass_s hand cases: 64 tokens with no bytes = 2.4695 s of compute; 500 tokens over the full store is I/O-bound at 21.34 s", abs(F.prefill_pass_s(64, 0.0) - 64 * 12 / 311) < 1e-9 and abs(F.prefill_pass_s(500, 1.0) - F.ALL_EXPERT_GB / 3.11) < 1e-9 and F.prefill_pass_s(500, 1.0) > 500 * 12 / 311)
    cs_ = r["compute_s"]
    t.check("s2_20: the claim's 14 s / 109 s come from the 55 GFLOP/s decode rate; at 311 GFLOP/s block 1 is 2.47 s and 500 tokens 19.29 s", abs(cs_["claimed_block_s"] - 14.0) < 0.1 and abs(cs_["claimed_full_s"] - 109.1) < 0.1 and abs(cs_["block64"] - 2.47) < 0.01 and abs(cs_["prompt500"] - 19.29) < 0.01)
    t.check("s2_20: measured share of experts a 64-token window touches in [0.42,0.50] (decode proxy; uniform would be 0.98 on the 30B)", inr(r["frac64"]["measured30"], 0.42, 0.50) and inr(r["frac64"]["uniform125"], 0.71, 0.72), str(r["frac64"]))
    t.check("s2_20: longest measured window (249 tokens) touches [0.55,0.68] of the experts; uniform 500 tokens of 512 experts touches 0.9999", inr(r["frac500"]["measured30"], 0.55, 0.68) and r["frac500"]["uniform125"] > 0.9998 and r["tokens_per_trace_min"] == 249)
    b = r["block"]
    t.check("s2_20: block-1 time over full-prompt time: measured proxy in [0.45,0.58], scaled to 125B in [0.33,0.42], uniform routing in [0.67,0.77]", inr(b["measured30"]["block1_over_full"], 0.45, 0.58) and inr(b["scaled125"]["block1_over_full"], 0.33, 0.42) and inr(b["uniform125"]["block1_over_full"], 0.67, 0.77), str({k: v["block1_over_full"] for k, v in b.items()}))
    t.check("s2_20: claimed 6-8x earlier first block is refuted: full/block1 time ratio is below 3 under every routing model", all(1 / v["block1_over_full"] < 3.0 for v in b.values()))
    t.check("s2_20: token-chunked prefill re-reads experts: 8 chunk passes cost 4-7x the bytes of one pass under every model", all(inr(v["chunk_passes_8x64_bytes_over_one_pass"], 4.0, 7.0) for v in b.values()), str({k: v["chunk_passes_8x64_bytes_over_one_pass"] for k, v in b.items()}))
    sp = r["spec_prefill"]
    t.check("s2_20: speculative prefill bound: speedup grows as less of the prompt is kept (1.0 < 1.48 < 1.80 < 2.77, +-0.1) because bytes saturate before compute", sp["1.0"]["speedup_vs_full"] == 1.0 and inr(sp["0.5"]["speedup_vs_full"], 1.38, 1.58) and inr(sp["0.25"]["speedup_vs_full"], 1.70, 1.90) and inr(sp["0.125"]["speedup_vs_full"], 2.67, 2.87))
    t.check("s2_20: kill criteria are sensitive to the routing model: uniform routing touches over half of the experts, the decode proxy does not", b["uniform125"]["touches_over_half_of_experts"] and not b["measured30"]["touches_over_half_of_experts"] and not b["scaled125"]["touches_over_half_of_experts"])
    t.check("s2_20: control: a 1-token 'block' touches 6% of the experts, far below the 50% kill line", F.union_frac(F.TRACES[0], 1) < 0.5)
    rows = F.parse_llama_bench_json('ggml_metal: init\n[{"n_prompt": 64, "n_gen": 0, "n_threads": 4, "avg_ts": 75.8}, {"n_prompt": 500, "n_gen": 0, "n_threads": 4, "avg_ts": 115.3}]')
    t.check("s2_20: llama-bench JSON parser skips log lines and returns (n_prompt, n_gen, threads, tok/s)", rows == [(64, 0, 4, 75.8), (500, 0, 4, 115.3)] and F.parse_llama_bench_json("no json here") == [])

    # ================================================================ family 21
    r = R["s2_21"]
    t.check("s2_21: deterministic (two runs identical)", same(r, F.fam_s2_21()))
    t.check("s2_21: tok_text hand case 'a b,c' -> ['a', ' b', ',', 'c'] and joins back to the text", F.tok_text("a b,c") == ["a", " b", ",", "c"] and "".join(F.tok_text("x  y\n z")) == "x  y\n z")
    t.check("s2_21: pld_run hand case 1: [5,6,7,8,5,6,7,8], prompt 4 -> forwards [1,1,1,2], 1 accepted", (lambda x: x["sizes"] == [1, 1, 1, 2] and x["accepted"] == 1 and x["out"] == 4)(F.pld_run([5, 6, 7, 8, 5, 6, 7, 8], 4, 3, 3, 8)))
    x = F.pld_run([1, 2, 3, 4, 5] * 4, 5, 3, 3, 8)
    t.check("s2_21: pld_run hand case 2: a 5-cycle repeated 4x with the first as prompt -> forwards [1,1,1,6,6], 10 of 15 accepted", x["sizes"] == [1, 1, 1, 6, 6] and x["accepted"] == 10 and x["drafted"] == 10 and sum(a_ for a_ in x["sizes"]) - len(x["sizes"]) == x["drafted"])
    x = F.pld_run(list(range(50)), 10, 3, 4, 8)
    t.check("s2_21: control: a sequence with no repeated n-gram yields f_accepted exactly 0 and one forward per token", x["accepted"] == 0 and x["forwards"] == x["out"])
    cps = r["corpora"]
    t.check("s2_21: f_accepted and acceptance in [0,1] for all corpora; random control is exactly 0", all(0 <= c["f_accepted"] <= 1 and (c["acceptance_of_drafted"] is None or 0 <= c["acceptance_of_drafted"] <= 1) for c in cps.values()) and cps["random_control"]["f_accepted"] == 0.0)
    t.check("s2_21: accounting: each forward yields accepted+1 tokens, so forwards per token = 1 - f_accepted (within 2e-3)", all(abs(c["forwards_per_token"] - (1 - c["f_accepted"])) < 2e-3 for c in cps.values()), str({k: (c["forwards_per_token"], c["f_accepted"]) for k, c in cps.items()}))
    t.check("s2_21: prose f_accepted in [0.02,0.25] (measured 0.107 on the Python docs text) and code in [0.08,0.25] (0.158): both below the screen's 0.3 kill line", inr(cps["prose"]["f_accepted"], 0.02, 0.25) and inr(cps["code"]["f_accepted"], 0.08, 0.25) and cps["prose"]["f_accepted"] < 0.3 and cps["code"]["f_accepted"] < 0.3, f"{cps['prose']['f_accepted']} {cps['code']['f_accepted']}")
    t.check("s2_21: control: a 10%-edited copy of its prompt reaches f in [0.5,0.7] (measured 0.593) and structured JSON rows f in [0.45,0.7] (0.590), far above prose", inr(cps["edit_control_10pct"]["f_accepted"], 0.5, 0.7) and inr(cps["json_rows"]["f_accepted"], 0.45, 0.7) and cps["edit_control_10pct"]["f_accepted"] > 3 * cps["prose"]["f_accepted"])
    un, uc = r["union_experts_per_layer"], r["union_claimed"]
    t.check("s2_21: unique experts per layer for k=2/3/4/8 consecutive tokens reproduce the vault figures 12.6/16.3/19.6/29.1 within 2%", all(abs(un[str(k) if str(k) in un else k] / uc[str(k) if str(k) in uc else k] - 1) < 0.02 for k in (2, 3, 4, 8)), f"{un} {uc}")
    t.check("s2_21: bytes per verified token at k=8 are 0.454 of single-token decode (range [0.44,0.47]), never zero", inr(r["bytes_per_token_vs_single_k8"], 0.44, 0.47))
    sc = r["screen_curve_100pct_accept"]
    t.check("s2_21: screen formula at 100% acceptance: f=0.15/0.3/0.6 at 8 GB in [1.03,1.12]/[1.10,1.20]/[1.30,1.70] and 32 GB f=0.6 in [1.7,2.0]; f=0.85 stays below 3x at every RAM size", inr(sc["0.15"][8], 1.03, 1.12) and inr(sc["0.3"][8], 1.10, 1.20) and inr(sc["0.6"][8], 1.30, 1.70) and inr(sc["0.6"][32], 1.7, 2.0) and all(sc["0.85"][x] < 3.0 for x in RAMS), str(sc))
    t.check("s2_21: speedup grows with f at every RAM size and with the conservative batch model is lower than the 311 GFLOP/s model", all(sc["0.15"][x] <= sc["0.3"][x] <= sc["0.6"][x] <= sc["0.85"][x] for x in RAMS) and all(r["screen_curve_combo_batch"][f][x] <= sc[f][x] + 1e-9 for f in sc for x in RAMS))
    t.check("s2_21: typical text (prose, code) gets no gain at 8 GB under either compute model (speedup <= 1.0) and at most 1.10x at 16/32 GB", all(cps[n]["speedup_serial"][8] <= 1.0 and cps[n]["speedup_serial_combo_batch"][x] <= 1.0 and cps[n]["speedup_serial"][x] <= 1.10 for n in ("prose", "code") for x in RAMS), str({n: cps[n]["speedup_serial"] for n in ("prose", "code")}))
    t.check("s2_21: even the repetitive cases stay far below the claimed 3x: max speedup over all corpora/RAM/models < 2.2", max(v for c in cps.values() for key in ("speedup_serial", "speedup_serial_combo_batch", "speedup_overlap") for v in c[key].values()) < 2.2)
    t.check("s2_21: control: a draft that is always rejected costs time (random-control speedup is exactly 1.0 only because no draft is ever proposed; a junk draft of 8 at 8 GB would cost more than one single pass)", cps["random_control"]["speedup_serial"][8] == 1.0 and F.pass_cost(8, 9)["serial"] > F.pass_cost(8, 1)["serial"])

    # ================================================================ family 22
    r = R["s2_22"]
    t.check("s2_22: deterministic (two runs identical)", same(r, F.fam_s2_22()))
    t.check("s2_22: common_prefix_len hand cases (KV reuse after an edit)", F.common_prefix_len([1, 2, 3, 4], [1, 2, 9, 4]) == 2 and F.common_prefix_len([1, 2], [1, 2, 3]) == 2 and F.common_prefix_len([], [1]) == 0 and F.common_prefix_len([7], [8]) == 0)
    cf = lambda n: n * 0.1
    t.check("s2_22: typing_ttft hand case (paste, 128 tokens, 2 chunks of 6.4 s) = 12.8 s", abs(F.typing_ttft(128, 64, float("inf"), cf) - 12.8) < 1e-9)
    t.check("s2_22: typing_ttft hand case (2 tok/s: chunk 1 done at 38.4 s, chunk 2 ready at 64 s) = 6.4 s after Enter; fast typing (20 tok/s) queues up = 9.6 s", abs(F.typing_ttft(128, 64, 2.0, cf) - 6.4) < 1e-9 and abs(F.typing_ttft(128, 64, 20.0, cf) - 9.6) < 1e-9)
    t.check("s2_22: typing_ttft hand case with a tail: 100 tokens pasted = 6.4 + 3.6 s", abs(F.typing_ttft(100, 64, float("inf"), cf) - 10.0) < 1e-9)
    u = r["unique_experts_64_window"]
    t.check("s2_22: unique experts in a 64-token window on the db/story/code traces: [53,61] / [51,59] / [62,70] of 128 (screen expected >= 90%)", inr(u["db"]["measured_of_128"], 53, 61) and inr(u["story"]["measured_of_128"], 51, 59) and inr(u["code"]["measured_of_128"], 62, 70), str(u))
    t.check("s2_22: the screen's premise (> 85% of experts per 64-token window) is NOT met by the decode proxy, but is under uniform routing (0.984 on the 30B)", not r["premise_over_85pct_measured"] and inr(r["uniform_share_64_30B"], 0.98, 0.99) and inr(r["uniform_share_64_125B"], 0.71, 0.72))
    m = r["models"]
    t.check("s2_22: one 64-token chunk pass costs [9,11] s (decode proxy), [6,8] s (scaled), [14,17] s (uniform); full 500-token pass 19.3 / 19.3 / 21.3 s", inr(m["measured30"]["chunk64_pass_s"], 9, 11) and inr(m["scaled125"]["chunk64_pass_s"], 6, 8) and inr(m["uniform125"]["chunk64_pass_s"], 14, 17) and inr(m["measured30"]["full500_pass_s"], 19, 20) and inr(m["uniform125"]["full500_pass_s"], 20.5, 22), str({k: v["chunk64_pass_s"] for k, v in m.items()}))
    t.check("s2_22: the screen's 14 s kill line is model dependent: not exceeded by the decode proxy or the scaled model, exceeded by uniform routing", not r["kill"]["chunk_pass_over_14s_scaled125"] and m["uniform125"]["chunk64_pass_s"] > 14.0 and m["measured30"]["chunk64_pass_s"] < 14.0)
    ty = lambda mn, rate: m[mn]["typing"][rate]["ttft_gain_vs_single_pass"]
    t.check("s2_22: typed prompt (1-3 tok/s): TTFT gain in [1.9,2.2] (proxy), [2.9,3.3] (scaled), [1.4,1.7] (uniform); the claimed 7.8x is not reached under any model", inr(ty("measured30", "1.0"), 1.9, 2.2) and inr(ty("scaled125", "3.0"), 2.9, 3.3) and inr(ty("uniform125", "2.0"), 1.4, 1.7) and all(ty(mn, k) < 7.8 for mn in m for k in m[mn]["typing"]))
    t.check("s2_22: pasted prompt chunked anyway is worse than one pass (gain 0.15-0.40) under every model", all(inr(ty(mn, "paste"), 0.15, 0.40) for mn in m), str({mn: ty(mn, "paste") for mn in m}))
    t.check("s2_22: control: under uniform routing, fast typing (5 tok/s) queues the chunk passes behind the typist and the gain falls below 1 (0.635), unlike slow typing", ty("uniform125", "5.0") < 1.0 < ty("uniform125", "1.0") and all(ty(mn, "5.0") <= ty(mn, "1.0") + 1e-9 for mn in m))
    t.check("s2_22: chunked prefill moves 4.0-6.5x the expert bytes of one pass under every model", all(inr(v["disk_bytes_chunked_over_single"], 4.0, 6.5) for v in m.values()), str({k: v["disk_bytes_chunked_over_single"] for k, v in m.items()}))
    cold = r["decode_cold_start_8gb"]
    t.check("s2_22: decode starts cold: LRU hit over the first 10 tokens is below the later hit (cold < warm, both in [0,1]) and the lost reads cost under 1 s", 0 <= cold["hit_first10_tokens"] < cold["hit_after_50"] <= 1 and 0 < cold["extra_io_s_first10_tokens"] < 1.0, str(cold))
    ph = F.per_token_hit(F.TRACES[0], F.footprints(8))
    t.check("s2_22: per_token_hit: one rate per token in [0,1], first token has hit 0 (empty cache)", len(ph) == len(F.cs.tr(F.TRACES[0])) and all(0 <= h <= 1 for h in ph) and ph[0] == 0.0)

    # ================================================================ family 23
    r = R["s2_23"]
    t.check("s2_23: deterministic (two runs identical)", same(r, F.fam_s2_23()))
    t.check("s2_23: amdahl hand cases: 0 -> 1.0, 0.5 -> 2.0, 0.9 -> 10", F.amdahl(0.0) == 1.0 and F.amdahl(0.5) == 2.0 and abs(F.amdahl(0.9) - 10) < 1e-9)
    fl = r["flops"]
    t.check("s2_23: 500 tokens x 12 GFLOP = 6 TFLOP: 109.1 s at the 55 GFLOP/s decode rate, 19.3 s at 311; the 109 s base is the decode rate applied to prefill", abs(fl["prefill500_s_at_55"] - 109.1) < 0.1 and abs(fl["prefill500_s_at_311"] - 19.3) < 0.05 and inr(fl["tflop_500_tokens"], 5.99, 6.01))
    t.check("s2_23: 311 GFLOP/s x 109 s = 33.9 TFLOP, 5.6x the 6 TFLOP of a 500-token prompt (the claim's baseline is not self-consistent); claimed cut 109 s -> 45 s = 0.587", inr(fl["tflop_in_109s_at_311"], 33.8, 34.0) and inr(fl["tflop_in_109s_at_311"] / fl["tflop_500_tokens"], 5.6, 5.7) and inr(fl["claimed_cut"], 0.58, 0.60))
    o = r["ops"]
    t.check("s2_23: operation counts: 144 graph matmuls, 1,440 slice matmuls per token, 73,728 expert matmuls per prefill pass", o["graph_ops_per_token"] == 144 and o["slice_matmuls_per_token"] == 1440 and o["prefill_expert_matmuls"] == 73728)
    sh = r["dispatch_us_scenarios"]
    t.check("s2_23: dispatch share grows with per-op cost; at 2/5/10 us decode share 1.3/3.3/6.6% and Amdahl bound <= 1.071; prefill <= 1.040", inr(sh["2"]["decode_share"], 0.012, 0.015) and inr(sh["5"]["decode_share"], 0.03, 0.036) and inr(sh["10"]["decode_share"], 0.06, 0.07) and sh["10"]["decode_amdahl"] <= 1.072 and sh["10"]["prefill_amdahl"] <= 1.041 and sh["2"]["decode_share"] < sh["5"]["decode_share"] < sh["10"]["decode_share"] < sh["50"]["decode_share"])
    nf = r["needed_for_claim"]
    t.check("s2_23: removing 60% needs 91 us per slice matmul in decode (18x the 5 us scenario), 909 us per graph op, 157 us per expert matmul in prefill; implied rate multiplier 2.5x", inr(nf["us_per_slice_matmul_decode"], 90, 92) and inr(nf["us_per_graph_op_decode"], 905, 915) and inr(nf["us_per_expert_matmul_prefill_19s"], 155, 159) and nf["implied_rate_multiplier"] == 2.5)
    t.check("s2_23: control: no dispatch scenario up to 50 us per op reaches the claimed 2.5x (largest Amdahl bound is the 50 us decode case, 1.49x)", max(v["decode_amdahl"] for v in sh.values()) < 2.5 and max(v["prefill_amdahl"] for v in sh.values()) < 2.5 and inr(sh["50"]["decode_amdahl"], 1.45, 1.55))
    io = r["decode_if_compute_doubled"]
    t.check("s2_23: for context, if compute could be doubled by any means the serial gain is [1.30,1.45] at 8 GB and [1.80,1.95] at 16/32 GB, so the only open question is the dispatch share", inr(io[8]["gain_x2_compute"], 1.30, 1.45) and inr(io[16]["gain_x2_compute"], 1.80, 1.95) and inr(io[32]["gain_x2_compute"], 1.80, 1.95), str({x: io[x]["gain_x2_compute"] for x in RAMS}))
    t.check("s2_23: decode hit in [0,1], io_s >= 0, serial tok/s below 4.58 at every RAM size", all(0 <= io[x]["hit"] <= 1 and io[x]["io_s"] >= 0 and 0 < io[x]["serial_tok_s"] < 4.5834 for x in RAMS))
    lb = F.layer_bound(500, 2)
    t.check("s2_23: layer_bound hand case: 500 tok/s on 2 layers -> 1 ms/layer -> 48 ms over 48 layers = 22% of a 218 ms token", abs(lb["per_layer_ms"] - 1.0) < 1e-9 and abs(lb["scaled_ms"] - 48.0) < 1e-9 and abs(lb["share"] - 0.22) < 1e-3 and abs(lb["amdahl"] - 1 / 0.78) < 1e-3)
    t.check("s2_23: control: the measured-bound helper can exceed the claim (a model at 50 tok/s over 2 layers would give a share above 100% and Amdahl capped)", F.layer_bound(50, 2)["share"] > 1.0 and F.layer_bound(50, 2)["amdahl"] > 100)

    # ================================================================ family 24
    r = R["s2_24"]
    t.check("s2_24: deterministic (two runs identical)", same(r, F.fam_s2_24()))
    t.check("s2_24: enum_flags hand case: 'reject' among approve/reject/revise -> choice at r, e forced, j|v choice, rest forced", F.enum_flags(["approve", "reject", "revise"], "reject") == [False, True, False, True, True, True, True])
    t.check("s2_24: enum_flags hand case: option that is a prefix of another keeps the end a choice", F.enum_flags(["a", "ab"], "a") == [True, False] and F.enum_flags(["a", "ab"], "ab") == [True, False, True])
    sc = ("obj", [("a", ("int",), True), ("b", ("enum", ["x", "y"]), True)]); o_ = F.Chars(); F.emit(sc, {"a": 5, "b": "x"}, o_)
    t.check('s2_24: emit hand case {"a":5,"b":"x"}: forced flags TTTTTFFTTTTTFTT (digit free, comma after a number free, x|y choice free)', F.render(o_.c) == '{"a":5,"b":"x"}' and "".join("T" if c[1] else "F" for c in o_.c) == "TTTTTFFTTTTTFTT")
    sc2 = ("obj", [("a", ("int",), True), ("note", ("str",), False)]); o1 = F.Chars(); F.emit(sc2, {"a": 1}, o1); o2 = F.Chars(); F.emit(sc2, {"a": 1, "note": "hi"}, o2)
    t.check("s2_24: optional property: the closing brace is a free choice while an optional property is absent, forced once it is present", o1.c[-1][1] is False and o2.c[-1][1] is True and F.render(o2.c) == '{"a":1,"note":"hi"}')
    outs = F.build_outputs()
    ok = True
    for name, v, ch in outs:
        try: ok = ok and json.loads(F.render(ch)) == v
        except Exception: ok = False
    t.check("s2_24: all 20 emitted outputs are valid JSON and equal the generated values (5 schemas x 4)", len(outs) == 20 and len({n for n, _, _ in outs}) == 5 and ok)
    t.check("s2_24: forced_runs / forward_sizes hand cases", F.forced_runs([True, True, False, True, False, False, True, True, True]) == [2, 1, 3] and F.forward_sizes([True, True, False, True, False]) == [3, 2, 1] and F.forward_sizes([]) == [1])
    fl = [True, False, True, True, False]
    t.check("s2_24: sum of forward sizes = tokens + EOS (every token is processed once)", sum(F.forward_sizes(fl)) == len(fl) + 1)
    v = r["by_variant"]; a_, b_ = v["unmerged_punct"], v["merged_punct"]
    t.check("s2_24: tokenization: merging punctuation gives fewer tokens (992 vs 1542) and a lower forced share; f in [0.40,0.48] merged and [0.56,0.65] unmerged (screen expected 0.2-0.4)", b_["tokens"] < a_["tokens"] and b_["f"] < a_["f"] and inr(b_["f"], 0.40, 0.48) and inr(a_["f"], 0.56, 0.65), f"{a_['f']} {b_['f']}")
    t.check("s2_24: mean forced run in [2.9,3.7] merged and [4.8,6.0] unmerged (screen expected 2-3); histogram counts add up to runs and run lengths to forced tokens", inr(b_["mean_run"], 2.9, 3.7) and inr(a_["mean_run"], 4.8, 6.0) and all(sum(x["run_hist"].values()) == x["n_runs"] and sum(int(k) * c for k, c in x["run_hist"].items()) == x["forced"] for x in (a_, b_)))
    t.check("s2_24: per-output f spread is wide (merged [0.2,0.3] to [0.55,0.62]); forwards per token = 1 - f x (something): forwards < tokens", inr(b_["f_per_output_min"], 0.2, 0.3) and inr(b_["f_per_output_max"], 0.55, 0.62) and b_["forwards"] < b_["tokens"] and a_["forwards"] < a_["tokens"])
    t.check("s2_24: serial speedup with the conservative batch model in [1.0,1.35] merged / [1.1,1.65] unmerged at every RAM size, never above the ideal 1/(1-f)", all(inr(b_["speedup_serial_combo_batch"][x], 1.0, 1.35) and inr(a_["speedup_serial_combo_batch"][x], 1.1, 1.65) and b_["speedup_serial_combo_batch"][x] <= b_["ideal_1_over_1_minus_f"] and a_["speedup_serial_combo_batch"][x] <= a_["ideal_1_over_1_minus_f"] for x in RAMS), str((a_["speedup_serial_combo_batch"], b_["speedup_serial_combo_batch"])))
    t.check("s2_24: with the screens' 311 GFLOP/s batch model the best case is [2.1,2.4] at 16/32 GB, still under the claimed 3.3x; 8 GB gain in [1.2,1.5]", all(inr(a_["speedup_serial"][x], 2.1, 2.4) for x in (16, 32)) and inr(b_["speedup_serial"][8], 1.2, 1.5) and a_["speedup_serial"][32] < r["claim"]["claimed_speedup"], str(a_["speedup_serial"]))
    t.check("s2_24: at 8 GB with perfect read/compute overlap the merged-token gain is below 1.05 (extra experts of a batch cost bytes); >= 0.9", b_["speedup_overlap"][8] < 1.05 and b_["speedup_overlap"][8] >= 0.9, str(b_["speedup_overlap"]))
    t.check("s2_24: claimed f=0.7 -> 3.3x; the screen's own bound at f=0.7 is 1.3-1.6x (8 GB) and 1.9-2.4x (16/32 GB), below the claim", inr(r["screen_bound_f07_serial"][8], 1.3, 1.6) and all(inr(r["screen_bound_f07_serial"][x], 1.9, 2.4) for x in (16, 32)) and r["claim"]["claimed_speedup"] > 3.3)
    free = F.Chars(); free.val("some free running prose without any grammar at all", [False] * 49); ff = F.tokenize_flagged(free.c, True)
    t.check("s2_24: control: free text has f = 0 and jump-forward changes nothing (all forwards size 1, speedup exactly 1.0)", not any(ff) and set(F.forward_sizes(ff)) == {1} and abs(len(ff) * F.pass_cost(8, 1)["serial"] / F.cost_of_sizes([1] * len(ff), 8) - 1.0) < 1e-12 and F.forward_sizes(ff)[:-1] == [1] * len(ff))
    t.check("s2_24: kill flags: f < 0.5 fires for merged tokens and not for unmerged; mean run >= 3 for merged", r["kill"]["f_below_0.5_merged"] and not r["kill"]["f_below_0.5_unmerged"] and not r["kill"]["mean_run_below_3_merged"])

    # ================================================================ family 25
    r = R["s2_25"]
    t.check("s2_25: deterministic (two runs identical)", same(r, F.fam_s2_25()))
    tm = F.tile_model(0.5, 0.0, 1.296, "none", comp=0.5, base_bpw=4.3)
    t.check("s2_25: tile_model hand case (hit 0.5, no tiles, 4.3 bpw, 1.296 GB/s, 0.5 s compute): 0.648 GB, serial 1.0, overlap 2.0", abs(tm["bytes_gb"] - 0.648) < 1e-9 and abs(tm["serial"] - 1.0) < 1e-9 and abs(tm["overlap"] - 2.0) < 1e-9)
    e1 = F.tile_model(0.5, 0.1, 1.296, "every_token", comp=0.5); e2 = F.tile_model(0.5, 0.1, 1.296, "on_miss", comp=0.5)
    t.check("s2_25: tile_model hand cases: 10% tiles every token add 0.1206 GB; on misses only add 0.0603 GB", abs(e1["bytes_gb"] - (0.648 * 2.2 / 4.3 * 0 + 0.5 * 1.296 * 2.2 / 4.3 + 0.1 * 1.296 * 4 / 4.3)) < 1e-9 and abs(e2["bytes_gb"] - (0.5 * 1.296 * 2.2 / 4.3 + 0.1 * 0.5 * 1.296 * 4 / 4.3)) < 1e-9)
    t.check("s2_25: file grows to 96 GB at 6 bpw (vs 69) and the cache holds 1.955x more base entries", inr(r["file_size_gb_6bpw"], 96, 97) and inr(r["cap_ratio"], 1.95, 1.96))
    h = r["hit"]
    t.check("s2_25: hit rates in [0,1]; the 2.2 bpw base hits at least as often as the 4.3 bpw slice (inclusion) and both rise with RAM; 8 GB 0.56 -> 0.80", all(0 <= h[x][k] <= 1 for x in RAMS for k in h[x]) and all(h[x]["b22"] >= h[x]["b43"] - 1e-9 for x in RAMS) and h[8]["b43"] <= h[16]["b43"] <= h[32]["b43"] and inr(h[8]["b43"], 0.54, 0.58) and inr(h[8]["b22"], 0.77, 0.82), str(h))
    ts = r["tok_s"]
    t.check("s2_25: 8 GB baseline at 2.4 GB/s: serial [2.15,2.25] and overlap [4.1,4.3] (screen 2.19 / 4.20)", inr(ts[8]["2.4"]["lru_4.3bit"]["serial"], 2.15, 2.25) and inr(ts[8]["2.4"]["lru_4.3bit"]["overlap"], 4.1, 4.3), str(ts[8]["2.4"]["lru_4.3bit"]))
    t.check("s2_25: all tok/s positive and overlap <= 4.5834 ceiling; serial <= overlap", all(0 < v["serial"] <= v["overlap"] + 1e-4 and v["overlap"] <= 4.5834 for x in RAMS for rt in ts[x] for v in ts[x][rt].values()))
    g8 = ts[8]["2.4"]
    t.check("s2_25: 8 GB serial gain at 2.4 GB/s: pure 2.2 bpw [1.55,1.75], tiles on miss [1.50,1.70], tiles every token [1.30,1.50] (screen: 1.4-1.6x)", inr(g8["pure_2.2bit"]["gain_serial"], 1.55, 1.75) and inr(g8["tiles_on_miss_10pct"]["gain_serial"], 1.50, 1.70) and inr(g8["tiles_every_token_10pct"]["gain_serial"], 1.30, 1.50), str({k: v["gain_serial"] for k, v in g8.items() if "gain_serial" in v}))
    t.check("s2_25: 8 GB overlap gain [1.05,1.12] at 2.4 GB/s (screen 1.09x) and exactly 1.0 at 3.11 GB/s (already at the ceiling)", inr(g8["pure_2.2bit"]["gain_overlap"], 1.05, 1.12) and ts[8]["3.11"]["pure_2.2bit"]["gain_overlap"] == 1.0)
    t.check("s2_25: 16/32 GB: tiles on miss gain in [0.98,1.08] and pure 2.2 bpw in [1.0,1.08] (screen 1.03-1.07x)", all(inr(ts[x][rt]["tiles_on_miss_10pct"]["gain_serial"], 0.98, 1.08) and inr(ts[x][rt]["pure_2.2bit"]["gain_serial"], 1.0, 1.08) for x in (16, 32) for rt in ts[x]))
    t.check("s2_25: control: re-reading the tiles every token LOSES at 16/32 GB (gain 0.82-0.90) and is far worse than reading them on misses", all(inr(ts[x][rt]["tiles_every_token_10pct"]["gain_serial"], 0.80, 0.91) and ts[x][rt]["tiles_every_token_10pct"]["gain_serial"] < ts[x][rt]["tiles_on_miss_10pct"]["gain_serial"] for x in (16, 32) for rt in ts[x]))
    t.check("s2_25: the screen's kill rule: 8 GB serial gain >= 1.5 survives but the 16 GB gain is < 1.1, so it is an 8 GB-only trick", g8["pure_2.2bit"]["gain_serial"] >= 1.5 and ts[16]["2.4"]["pure_2.2bit"]["gain_serial"] < 1.1)
    t.check("s2_25: q2_groupwise hand cases: a 4-level group is exact, a constant group is exact", F.q2_groupwise([0, 1, 2, 3] * 4) == [0, 1, 2, 3] * 4 and F.q2_groupwise([2.5] * 16) == [2.5] * 16 and len(F.q2_groupwise([0.0, 0.3, 0.9, 1.0] * 8)) == 32)
    rng = random.Random(25); gs = [[rng.gauss(0, 1) for _ in range(512)] for _ in range(16)]; mixed = gs[:12] + [[rng.gauss(0, 1) ** 3 for _ in range(512)] for _ in range(4)]
    ea, eb = F.residual_energy(gs), F.residual_energy(mixed)
    t.check("s2_25: control: the selection statistic detects heterogeneity: homogeneous Gaussians put ~14% of residual energy in the top 10% of slices, 4 heavy-tailed slices put > 35%", ea["top10pct_slices_energy_share"] < 0.20 and eb["top10pct_slices_energy_share"] > 0.35, f"{ea['top10pct_slices_energy_share']} {eb['top10pct_slices_energy_share']}")
    t.check("s2_25: top_share hand cases: [1,1,1,1,6] top 20% = 0.6; uniform 10 values top 10% = 0.1", abs(F.top_share([1, 1, 1, 1, 6], 0.2) - 0.6) < 1e-9 and abs(F.top_share([1.0] * 10, 0.1) - 0.1) < 1e-9)
    for key, lo_hi in (("residual_layer10", None), ("residual_layer30", None)):
        s = r[key]
        if s is None: t.check(f"s2_25: {key}: weights file absent, skipped (informational pass)", True)
        else: t.check(f"s2_25: real weights {key}: relative 2-bit residual in [0.25,0.40], CV < 0.05 (screen: nothing to select if < 0.2), top 10% of slices hold [0.10,0.20] and top 10% of groups [0.15,0.25] of the residual energy", inr(s["rel_residual_mean"], 0.25, 0.40) and s["rel_residual_cv"] < 0.05 and inr(s["top10pct_slices_energy_share"], 0.10, 0.20) and inr(s["top10pct_groups_energy_share"], 0.15, 0.25), str(s))
    return t


if __name__ == "__main__": sys.exit(0 if run().summary() else 1)
