"""Published-technique families, group A (brainstorm screen 1, ideas #0..#7): each family re-computes the decisive free test of its idea on the real Qwen3-30B traces
and checks determinism, sanity bounds, hand-computed helper cases, the measured result against the screen's claim, and a control that can fail.
Ranges named in a check were derived from the run on the three recorded traces (db / story / code); a check fails if the recomputation leaves its range."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "tools")); sys.path.insert(0, os.path.join(HERE, "..", "tools", "pubtech"))
from harness import T
import f_A as fa
import cachesim as cs

def inr(x, lo, hi): return lo <= x <= hi
def run():
    t = T("pubtech A (screen 1 #0..#7)"); t0 = time.time()
    def rng(name, x, lo, hi): t.check(f"{name} in [{lo}, {hi}]", inr(x, lo, hi), f"{x:.4f}")
    def det(fam):
        fa.clear_cache(); a = fam(quick=True); fa.clear_cache(); b = fam(quick=True)
        t.check(f"{fam.__name__}: two runs with a cleared cache are identical (deterministic)", a == b)
        t.check(f"{fam.__name__}: result names its family and lists what is unmeasured", isinstance(a.get("family"), str) and len(a.get("unmeasured", [])) >= 2)
    # ---- helpers: hand-computed unit cases
    t.check("expected_accepted(8, 0.5) = (1 - 2^-8) / 0.5 = 1.9921875", abs(fa.expected_accepted(8, 0.5) - 1.9921875) < 1e-12)
    t.check("expected_accepted: a=1 gives k, k=1 gives 1", fa.expected_accepted(4, 1.0) == 4 and fa.expected_accepted(1, 0.3) == 1.0)
    t.check("union_layer keeps first-seen order and drops repeats", fa.union_layer([[1, 2, 3], [3, 4], [1]]) == [1, 2, 3, 4])
    t.check("uniform_union_fraction: k=1 is exactly 1, k=2 of 8-of-128 is 0.96875", fa.uniform_union_fraction(128, 8, 1) == 1.0 and abs(fa.uniform_union_fraction(128, 8, 2) - 0.96875) < 1e-12)
    c = fa.LRU(2); res = [c.touch(0, e) for e in (1, 2, 1, 3, 2)]
    t.check("LRU unit: hand-computed hits [F,F,T,F,F]", res == [False, False, True, False, False], str(res))
    c = fa.LRU(96, per_layer=True)
    t.check("per-layer LRU: 96 entries = 2 per layer; layers do not share slots", [c.touch(0, 1), c.touch(0, 2), c.touch(0, 3), c.touch(1, 1), c.touch(0, 1)] == [False, False, False, False, False] and c.touch(0, 3))
    t.check("LRU with 0 entries never hits", not any(fa.LRU(0).touch(0, 1) for _ in range(3)))
    t.check("cache_footprints: 2.5 GiB of 4.3 bpw experts = 2.07 footprints; halving the bits doubles it", inr(fa.cache_footprints(2.5), 2.06, 2.08) and abs(fa.cache_footprints(2.5, 2.15) / fa.cache_footprints(2.5) - 2) < 1e-9)
    t.check("compute ceiling is 12 GFLOP at 55 GFLOP/s = 4.583 tok/s", abs(1 / fa.COMP_S - 4.58333) < 1e-4)
    t.check("serial 0.5 s io + compute vs overlap", abs(fa.tps_serial(0.5) - 1 / (0.5 + fa.COMP_S)) < 1e-12 and fa.tps_overlap(0.5) == 2.0 and abs(fa.tps_overlap(0.0) - 1 / fa.COMP_S) < 1e-12)
    t.check("breakeven_gflops: k=4 all accepted at the same bytes needs exactly the decode rate (55 GFLOP/s)", abs(fa.breakeven_gflops(4, 1.0, 0.4, 0.4) - 55.0) < 1e-9)
    t.check("breakeven_gflops: k=2, a=0.5 (E=1.5) at equal bytes needs 55 * 2 / 1.5 = 73.33 GFLOP/s", abs(fa.breakeven_gflops(2, 0.5, 0.4, 0.4) - 110 / 1.5) < 1e-9)
    t.check("breakeven_gflops: None when the extra bytes alone cost more than the whole baseline token", fa.breakeven_gflops(8, 0.7, 2.0, 0.4) is None)
    # synthetic traces for the block simulator (hand-computed)
    fa._TOK["syn_same"] = [[[0, 1, 2, 3, 4, 5, 6, 7] for _ in range(48)] for _ in range(6)]
    fa._TOK["syn_disj"] = [[[8 * i + j for j in range(8)] for _ in range(48)] for i in range(4)]
    s1 = fa.spec_sim("syn_same", 1, 1.0, 100); s3 = fa.spec_sim("syn_same", 3, 1.0, 100)
    t.check("spec_sim unit: identical tokens, huge cache -> only the 384 cold misses; k=1 and k=3 both give 64 misses per token", s1["misses"] == 384 and s3["misses"] == 384 and s1["mpa"] == 64.0 and s3["mpa"] == 64.0 and s3["blocks"] == 2)
    z1 = fa.spec_sim("syn_same", 1, 1.0, 0); z3 = fa.spec_sim("syn_same", 3, 1.0, 0)
    t.check("spec_sim unit: identical tokens, no cache -> k=1 misses 384 per token, k=3 reads the union once (128 per token)", z1["mpa"] == 384.0 and z3["mpa"] == 128.0)
    d4 = fa.spec_sim("syn_disj", 4, 1.0, 0)
    t.check("spec_sim unit: disjoint tokens share nothing, so grouping saves nothing (384 per token)", d4["mpa"] == 384.0 and d4["accesses"] == 4 * 384)
    # differential: the block simulator at k=1 must equal the repo's independent LRU simulator
    for nm in fa.TRACES:
        s = [(l, e) for tk in fa.tokens(nm) for l in range(48) for e in tk[l]]; cap = int(2.07 * 384)
        t.check(f"{nm[-5:]}: spec_sim(k=1) misses equal cachesim.lru_hits misses on the same stream (differential)", fa.spec_sim(nm, 1, 1.0, 2.07)["misses"] == len(s) - cs.lru_hits(s, cap))
    t.check("trace loader: three traces of 249-299 tokens, 48 layers, 8 distinct ids per layer", all(240 <= len(fa.tokens(n)) <= 300 and all(len(set(x)) == 8 for tk in fa.tokens(n) for x in tk) for n in fa.TRACES))

    # ---------------- family 0
    det(fa.fam_s1_0); r = fa.fam_s1_0(); R = r["lru_relative_bytes"]
    rng("s1#0 unique experts per layer, k=8 (screen 29.1)", r["union_per_layer"]["k8"], 28.5, 29.7); rng("s1#0 unique per layer, k=2 (screen 12.6)", r["union_per_layer"]["k2"], 12.3, 12.9)
    t.check("s1#0 union fraction falls with k: k2 > k4 > k8, all in (0,1]", 1 >= r["union_fraction"]["k2"] > r["union_fraction"]["k4"] > r["union_fraction"]["k8"] > 0)
    rng("s1#0 uncached GB per accepted token, k=8, a=1 (screen 0.59)", r["uncached_gb_per_accepted_k8"]["a1.0"], 0.57, 0.61); rng("s1#0 uncached GB per accepted token, k=8, a=0.5 (screen 2.38)", r["uncached_gb_per_accepted_k8"]["a0.5"], 2.30, 2.43)
    g = r["uncached_gb_per_accepted_k8"]; t.check("s1#0 uncached GB per accepted token falls as acceptance rises", g["a0.5"] > g["a0.7"] > g["a0.8"] > g["a0.9"] > g["a1.0"] > 0)
    rng("s1#0 uncached break-even acceptance vs 1.296 GB plain (screen: >= 0.75)", r["uncached_break_even_alpha_k8"], 0.74, 0.78)
    t.check("s1#0 every LRU-relative ratio is finite and positive", all(0 < v < 50 for v in R.values()))
    t.check("s1#0 wrong-branch loads never make the verify cheaper than the optimistic true-token union (tolerance 2%)", all(R[k.replace("|oracle", "|wrong")] >= v * 0.98 for k, v in R.items() if k.endswith("|oracle") and k.replace("|oracle", "|wrong") in R))
    t.check("s1#0 wrong-branch ratio falls as acceptance rises (k=8, both cache sizes)", all(R[f"fp{f}|k8|a0.6|wrong"] > R[f"fp{f}|k8|a0.8|wrong"] > R[f"fp{f}|k8|a0.9|wrong"] for f in (2.1, 4.0)))
    rng("s1#0 MEASURED vs claim: smallest LRU-relative bytes at acceptance <= 0.9 (screen: never below 1.0)", r["lru_relative_bytes_min_alpha_le_0p9"], 0.98, 1.05)
    rng("s1#0 k=8 at 2.1 footprints, a=0.9, true tokens (screen 1.34-1.38 at E=5.7; recomputed value lies a little lower)", R["fp2.1|k8|a0.9|oracle"], 1.20, 1.40)
    rng("s1#0 k=4 at 2.1 footprints, a=0.8, wrong-branch (screen 1.61 at E=3)", R["fp2.1|k4|a0.8|wrong"], 1.55, 1.80)
    rng("s1#0 DISAGREES with 'the union never beats the cache': all-accepted k=8 at 2.1 footprints beats k=1 LRU by", R["fp2.1|k8|a1.0|oracle"], 0.85, 0.95)
    t.check("s1#0 that is the only cell below 1.0 of 42", r["n_cells_below_1"] == 1 and r["n_cells"] == 42 and r["lru_relative_bytes_min_cell"] == "fp2.1|k8|a1.0|oracle")
    rng("s1#0 compute ceiling of a k=8 verify at 311 GFLOP/s, a=0.8 (screen ~13)", r["compute_ceiling_tps_k8_a0p8_at_311"], 13.2, 13.8)
    ctl = fa.spec_sim("qwen3-30b_db", 8, 1.0, 0)["mpa"] / fa.spec_sim("qwen3-30b_db", 1, 1.0, 0)["mpa"]
    rng("s1#0 control that can fail: with NO cache grouping k=8 does cut bytes (ratio ~ the 0.45 union fraction)", ctl, 0.40, 0.50)

    # ---------------- family 1
    det(fa.fam_s1_1); r = fa.fam_s1_1(); O = r["misses_per_token_in_footprints_oracle"]; D = r["misses_per_accepted_in_footprints_drafted"]
    t.check("s1#1 misses per token are fractions of a footprint in (0,1] and fall as the cache grows (k=1)", all(0 < v <= 1.0 for v in O.values()) and O["fp2.1|k1"] > O["fp3.0|k1"] > O["fp4.0|k1"] > O["fp8.0|k1"] > 0)
    t.check("s1#1 cold union fraction falls with k: 1 > k2 > k4 > k8", r["cold_union_fraction"]["k1"] == 1.0 and r["cold_union_fraction"]["k2"] > r["cold_union_fraction"]["k4"] > r["cold_union_fraction"]["k8"])
    rng("s1#1 cold union fraction at k=8 (screen 0.43-0.47)", r["cold_union_fraction"]["k8"], 0.43, 0.48); rng("s1#1 k=1 LRU misses per token at 2.1 footprints (screen 0.40-0.46)", O["fp2.1|k1"], 0.40, 0.47)
    rng("s1#1 MEASURED best all-accepted grouping gain over k=1 LRU in the whole grid (screen ~1.0x; kill if < 1.3)", r["best_gain_oracle"], 1.00, 1.09)
    rng("s1#1 MEASURED best drafted gain over k=1 LRU (screen: worse, 0.4-0.6x at a=0.7)", r["best_gain_drafted"], 0.90, 1.00)
    t.check("s1#1 kill rule holds: no cap/k/acceptance gives 1.3x fewer misses per accepted token", r["killed"] is True and r["kill_threshold"] == 1.3)
    rng("s1#1 drafted k=8, a=0.7, 2.1 footprints: footprints missed per accepted token (screen 1.19)", D["fp2.1|k8|a0.7"], 1.15, 1.45); rng("s1#1 k=1 baseline at 2.1 footprints (screen 0.44)", O["fp2.1|k1"], 0.40, 0.47)
    t.check("s1#1 drafted cost falls as acceptance rises for k>=4 at 2.1 and 4 footprints", all(D[f"fp{f}|k{k}|a0.5"] > D[f"fp{f}|k{k}|a0.7"] > D[f"fp{f}|k{k}|a0.9"] for f in (2.1, 4.0) for k in (4, 8)))
    t.check("s1#1 at 8 footprints grouping is irrelevant (all-accepted gain within 1% of 1.0)", all(abs(r["gain_vs_k1_oracle"][f"fp8.0|k{k}"] - 1) < 0.01 for k in (2, 4, 8)))
    t.check("s1#1 control that can fail: with no cache the oracle gain is > 2 (the metric can show a gain)", 1.0 / r["cold_union_fraction"]["k8"] > 2.0)

    # ---------------- family 2
    det(fa.fam_s1_2); r = fa.fam_s1_2(); U = r["window_union_fraction"]; W = [f"W{w}" for w in (1, 8, 16, 32, 64, 128, 249)]
    t.check("s1#2 window union fraction rises with the window and W=1 is exactly 8/128", all(U[a] < U[b] for a, b in zip(W, W[1:])) and U["W1"] == 8 / 128 and all(0 < v <= 1 for v in U.values()))
    rng("s1#2 PROXY (decode traces): experts touched in a 128-token window (screen 0.52-0.61)", U["W128"], 0.52, 0.59); rng("s1#2 PROXY: window 16 (screen 0.28-0.33)", U["W16"], 0.28, 0.33)
    t.check("s1#2 marginal-iid estimate at W=128 agrees with the measured union within 0.03, and W=500 is larger", abs(r["iid_marginal_union_fraction"]["W128"] - U["W128"]) < 0.03 and r["iid_marginal_union_fraction"]["W500"] > r["iid_marginal_union_fraction"]["W128"])
    t.check("s1#2 uniform routing would touch ~everything, so real routing is far from uniform", r["uniform_routing_union_fraction"]["qwen3_128_top8_W128"] > 0.999 and r["uniform_routing_union_fraction"]["target_512_top10_W500"] > 0.999)
    t.check("s1#2 prefill compute floor for 500 tokens = 6 TFLOP / 311 GFLOP/s = 19.3 s", inr(r["prefill_compute_s_500"], 19.2, 19.4))
    t.check("s1#2 token-major no-sharing baseline is exactly 500 x 1.296 = 648 GB (the screen's 'wrong baseline')", abs(r["uncached_token_major_gb_500"] - 648.0) < 1e-6)
    tt_ = r["ttft"]; t.check("s1#2 TTFT arithmetic: read GB rises with the union, overlapped TTFT >= compute floor, mmap slower than both", all(tt_[a]["read_gb"] < tt_[b]["read_gb"] for a, b in (("union0.5", "union0.6"), ("union0.6", "union0.75"), ("union0.75", "union1.0"))) and all(v["explicit_overlap_s"] >= 19.29 and v["mmap_s"] > v["explicit_serial_s"] >= v["explicit_overlap_s"] for v in tt_.values()))
    rng("s1#2 TTFT gain vs mmap at union 0.75, serial (screen 3.5-6)", tt_["union0.75"]["gain_vs_mmap_serial"], 3.5, 3.8); rng("s1#2 TTFT gain vs mmap at union 0.95, overlap at 1.56 GB/s read size (screen 3.5-6)", tt_["union0.95"]["gain_vs_mmap_overlap_at_1p56"], 4.5, 4.7)
    rng("s1#2 DISAGREES at the top: overlapped at 2.4 GB/s the gain vs mmap is ~7x (read-size ratio 2.4/0.34), above the screen's 6x", tt_["union0.95"]["gain_vs_mmap_overlap"], 6.9, 7.2)
    lm = r["token_vs_layer_major"]
    rng("s1#2 layer-major (W=128) vs token-major LRU misses at 2.07 footprints", lm["W128|fp2.07"]["ratio"], 5.5, 6.6); rng("s1#2 at 7.58 footprints a 128-token layer-major batch is WORSE than token-major LRU (union exceeds the cache)", lm["W128|fp7.58"]["ratio"], 0.70, 0.82)
    t.check("s1#2 once the whole expert set fits (20.8 footprints) both orders miss only the cold set (ratio 1.0)", abs(lm["W128|fp20.84"]["ratio"] - 1) < 1e-9 and abs(lm["W64|fp20.84"]["ratio"] - 1) < 1e-9)
    t.check("s1#2 control that can fail: with a cache larger than the whole expert set, layer-major batching of 2 and token-major miss the same cold set", fa.spec_sim("qwen3-30b_db", 2, 1.0, 100)["misses"] == fa.spec_sim("qwen3-30b_db", 1, 1.0, 100)["misses"])
    t.check("s1#2 needs a model: the real prefill trace is declared unmeasured", any("500-token prefill" in x for x in r["unmeasured"]) and fa.FAMILIES["fam_s1_2"]["needs"] == "model")

    # ---------------- family 3
    det(fa.fam_s1_3); r = fa.fam_s1_3(); P = r["per_layer_lru_24_entries_misses_per_token"]; G = r["per_layer_miss_gain_vs_K8"]
    rng("s1#3 K=8 misses/token, db, 24 entries per layer (screen 122.8)", P["K8"]["0b_db"], 121, 125); rng("s1#3 K=8 misses/token, story (screen 103.1)", P["K8"]["story"], 101, 105); rng("s1#3 K=8 misses/token, code (screen 136.6)", P["K8"]["_code"], 134.5, 139)
    rng("s1#3 K=6 misses/token, db (screen 80.6)", P["K6"]["0b_db"], 79, 82); rng("s1#3 MEASURED K=6 vs K=8 miss gain, db (screen 1.52)", G["K6"]["0b_db"], 1.49, 1.56); rng("s1#3 MEASURED K=6 vs K=8 miss gain, story (screen 1.49)", G["K6"]["story"], 1.45, 1.52); rng("s1#3 MEASURED K=6 vs K=8 miss gain, code (screen 1.46)", G["K6"]["_code"], 1.43, 1.50)
    t.check("s1#3 misses fall monotonically with K (8 > 6 > 5) on every trace, so gain(K=5) > gain(K=6) > 1", all(P["K8"][x] > P["K6"][x] > P["K5"][x] for x in P["K8"]) and all(G["K5"][x] > G["K6"][x] > 1 for x in G["K6"]))
    gl = r["global_lru_misses_per_token"]; mg = {x: gl[f"{x}GB|K8"] / gl[f"{x}GB|K6"] for x in (8, 16, 32)}
    t.check("s1#3 global-LRU miss gain of K=6 shrinks as the cache grows (8 GB > 16 GB > 32 GB), all > 1", mg[8] > mg[16] > mg[32] > 1, str(mg)); rng("s1#3 K=6 miss gain at the 8 GB cache", mg[8], 1.38, 1.48)
    A = r["amdahl_compute_ceiling_gain"]; rng("s1#3 compute ceiling gain for 6-of-10 (expert share 40%) (screen 1.18)", A["6of10"], 1.18, 1.21); rng("s1#3 compute gain 6-of-8", A["6of8"], 1.10, 1.13); rng("s1#3 compute gain 4-of-10", A["4of10"], 1.30, 1.34)
    X = r["throughput_rows"]; rng("s1#3 MEASURED 8 GB serial gain, 6-of-10 analogue (K=5), top-k cut alone (screen 1.9 incl. 2.5 bpw requant)", X["8GB|K5"]["serial_gain"], 1.38, 1.50); rng("s1#3 MEASURED 8 GB overlap gain, K=5 analogue (screen 1.3)", X["8GB|K5"]["overlap_gain"], 1.25, 1.33)
    t.check("s1#3 at 32 GB the overlap gain equals the pure compute gain (I/O is hidden) for K=6", abs(X["32GB|K6"]["overlap_gain"] - A["6of8"]) < 0.005 and all(v["serial_gain"] >= v["overlap_gain"] - 1e-9 for k, v in X.items() if k.startswith("8GB")))
    pj = r["project_simulator_cross_check"]; t.check("s1#3 the repo's combination simulator agrees on top-6 at 8 GB within 0.05, and cache-aware 6-of-8 beats rank-order top-6", abs(pj["8GB"]["top6_vs_full"] - mg[8]) < 0.05 and all(v["aware6_vs_top6"] > 1 for v in pj.values()))
    t.check("s1#3 quoted OLMoE perplexity loss: top-6 +8.8% > the 5% quality gate, top-4 +45.6% (quoted, not re-measured)", r["earlier_olmoe_relative_ppl_loss"]["top6"] == 0.088 and r["earlier_olmoe_relative_ppl_loss"]["top4"] == 0.456 and r["earlier_olmoe_relative_ppl_loss"]["top6"] > 0.05)
    big = fa._kcut_misses("qwen3-30b_db", 8, 10 ** 6) / fa._kcut_misses("qwen3-30b_db", 6, 10 ** 6)
    t.check("s1#3 control that can fail: with an unbounded cache the K=6 gain is smaller than at 24 entries per layer (gain comes from capacity pressure)", big < G["K6"]["0b_db"] and big >= 1.0, f"{big:.3f}")

    # ---------------- family 4
    det(fa.fam_s1_4); r = fa.fam_s1_4(); M = r["miss_fraction_of_full_token_read"]; Q = r["relative_to_k1"]
    t.check("s1#4 miss fraction in [0,1], k=1 uncached = 1.0 exactly, and uncached falls with k", all(0 <= v <= 1 for v in M.values()) and M["fp0|k1"] == 1.0 and M["fp0|k1"] > M["fp0|k2"] > M["fp0|k4"] > M["fp0|k8"])
    rng("s1#4 uncached k=8 fraction (screen 0.43-0.47)", M["fp0|k8"], 0.43, 0.48); rng("s1#4 uncached gain at k=8 (screen ~2.2x)", r["cold_gain_k8"], 2.1, 2.3)
    rng("s1#4 MEASURED k=8 vs k=1 with a 2-footprint LRU (screen: identical, 1.0x)", Q["fp2|k8"], 0.95, 1.07); rng("s1#4 MEASURED k=4 vs k=1 at 2 footprints (screen: 1.3x worse)", Q["fp2|k4"], 1.25, 1.45)
    t.check("s1#4 at 4 footprints k=2/4/8 are all within 5% of k=1 (flat)", all(abs(Q[f"fp4|k{k}"] - 1) < 0.05 for k in (2, 4, 8)))
    t.check("s1#4 drafted (a=0.7, wrong-branch) is never cheaper in bytes per accepted token than k=1", all(v > 1.0 for v in r["drafted_relative_to_k1"].values()), str(min(r["drafted_relative_to_k1"].values())))
    rng("s1#4 union fraction at k=8 (screen 29/64 = 0.455)", r["union_fraction_k8_measured"], 0.44, 0.47)
    u = r["uniform_union_fraction"]; rng("s1#4 uniform-routing bound for 128-choose-8 at k=8", u["qwen3_128_top8_k8"], 0.80, 0.81); rng("s1#4 same bound for the 512-choose-10 target (a larger union than the 30B traces suggest)", u["target_512_top10_k8"], 0.93, 0.94)
    B = r["batch_compute_breakeven"]; t.check("s1#4 break-even compute rate is never below the decode rate (batching must at least match 55 GFLOP/s)", all(v["breakeven_gflops"] is None or v["breakeven_gflops"] >= 55 for v in B.values()))
    t.check("s1#4 at 8 GB, k=8 and a=0.7 no compute rate can recover the extra bytes (None)", B["8GB|k8|a0.7"]["breakeven_gflops"] is None and B["8GB|k4|a0.7"]["breakeven_gflops"] is None)
    t.check("s1#4 at 32 GB (bytes equal) break-even = (k/E) x 55: k=2, a=0.7 -> 64.7 GFLOP/s", abs(B["32GB|k2|a0.7"]["breakeven_gflops"] - 2 / fa.expected_accepted(2, 0.7) * 55) < 1e-6)
    t.check("s1#4 control that can fail: batch-2 break-even is lowest at k=2 a=0.85 (needs < 75 GFLOP/s at 8 GB), k=4 needs far more (> 400)", B["8GB|k2|a0.85"]["breakeven_gflops"] < 75 and B["8GB|k4|a0.85"]["breakeven_gflops"] > 400)

    # ---------------- family 5
    det(fa.fam_s1_5); r = fa.fam_s1_5(); Uq = r["unique_per_layer"]; C = r["cached"]
    rng("s1#5 unique per layer, k=2 (screen 12.6)", Uq["k2"], 12.3, 12.9); rng("s1#5 unique per layer, k=3 (screen 16.3)", Uq["k3"], 16.0, 16.6); rng("s1#5 unique per layer, k=4 (screen 19.6)", Uq["k4"], 19.4, 19.9); rng("s1#5 unique per layer, k=8 (screen 29.1)", Uq["k8"], 28.8, 29.4)
    rng("s1#5 plain uncached explicit-read decode, tok/s (screen 1.31)", r["plain_uncached_tps"], 1.30, 1.34)
    un = r["uncached"]; rng("s1#5 uncached k=2, a=0.7 tok/s with the bonus-token reading of acceptance (screen 2.35)", un["k2|a0.7"]["tps_with_bonus_token"], 2.30, 2.42); rng("s1#5 uncached k=8, a=0.7 (screen 1.40)", un["k8|a0.7"]["tps_with_bonus_token"], 1.36, 1.46)
    t.check("s1#5 uncached bytes per accepted token fall with acceptance and k=8 at a=0.5 is slower than plain decode", all(un[f"k{k}|a0.5"]["gb_per_accepted"] > un[f"k{k}|a0.85"]["gb_per_accepted"] for k in (2, 3, 4, 8)) and un["k8|a0.5"]["tps"] < r["plain_uncached_tps"])
    rng("s1#5 MEASURED best LRU-cached bytes gain at a=0.7 over every k (kill if < 1.3; screen expected 1.1-1.4 at k=2-3)", r["best_bytes_gain_a0p7"], 0.95, 1.02)
    t.check("s1#5 kill rule holds: bytes gain at a=0.7 is below 1.3 at every k, cache size and mode", r["best_bytes_gain_a0p7"] < r["kill_threshold"] and all(v["bytes_gain_vs_k1"] < 1.3 for v in C.values()))
    t.check("s1#5 pessimistic wrong-branch bytes gain <= optimistic one in every cell (tolerance 2%)", all(C[k.replace("|wrong", "|oracle")]["bytes_gain_vs_k1"] >= v["bytes_gain_vs_k1"] * 0.98 for k, v in C.items() if k.endswith("|wrong")))
    t.check("s1#5 bytes gain rises with acceptance for k=3 and k=4 (optimistic, 2.1 footprints)", all(C[f"fp2.1|k{k}|a0.5|oracle"]["bytes_gain_vs_k1"] < C[f"fp2.1|k{k}|a0.7|oracle"]["bytes_gain_vs_k1"] < C[f"fp2.1|k{k}|a0.85|oracle"]["bytes_gain_vs_k1"] for k in (3, 4)))
    rng("s1#5 tok/s gain at k=2, a=0.7, 2.1 fp, optimistic, IF batch-2 compute runs at 311 GFLOP/s (unmeasured assumption)", r["tps_gain_a0p7_fp2p1"]["fp2.1|k2|a0.7|oracle"], 1.5, 1.7); rng("s1#5 same with wrong-branch loads", r["tps_gain_a0p7_fp2p1"]["fp2.1|k2|a0.7|wrong"], 1.2, 1.4)
    t.check("s1#5 k>=4 loses tok/s at a=0.7 even with the 311 GFLOP/s assumption", r["tps_gain_a0p7_fp2p1"]["fp2.1|k4|a0.7|oracle"] < 1 and r["tps_gain_a0p7_fp2p1"]["fp2.1|k8|a0.7|oracle"] < 1)
    gt = r["gate"]; rng("s1#5 gate: compute-only ceiling at 60% skipped (screen 11.5)", gt["compute_only_ceiling_tps"], 11.4, 11.5); rng("s1#5 gate: uncached serial tok/s (screen 3.3)", gt["uncached_serial_tps"], 3.25, 3.35)
    t.check("s1#5 control that can fail: the compute-only gate ceiling is more than 2x the serial figure that includes I/O (the 11 tok/s counts compute only)", gt["compute_only_ceiling_tps"] > 2 * gt["uncached_serial_tps"])

    # ---------------- family 6
    det(fa.fam_s1_6); r = fa.fam_s1_6(); H = r["hit"]
    t.check("s1#6 hit rates in [0,1], non-decreasing in stickiness f and in cache size", all(0 <= v <= 1 for v in H.values()) and all(H[f"f{a}|fp{p}"] <= H[f"f{b}|fp{p}"] + 1e-9 for p in (1.5, 2.0, 3.0) for a, b in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0))) and all(H[f"f{f}|fp1.5"] <= H[f"f{f}|fp2.0"] <= H[f"f{f}|fp3.0"] for f in (0.0, 0.25, 0.5, 0.75, 1.0)))
    rng("s1#6 baseline hit at 2 footprints (screen 0.54)", H["f0.0|fp2.0"], 0.53, 0.57); rng("s1#6 MEASURED hit at f=0.5, 2 footprints (screen 0.76)", H["f0.5|fp2.0"], 0.745, 0.78)
    rng("s1#6 baseline hit at 1.5 footprints (screen 0.45)", H["f0.0|fp1.5"], 0.44, 0.47); rng("s1#6 hit at f=0.5, 1.5 footprints (screen 0.71)", H["f0.5|fp1.5"], 0.70, 0.74); rng("s1#6 hit at f=0.5, 3 footprints (screen 0.83)", H["f0.5|fp3.0"], 0.82, 0.85)
    o = r["token_overlap"]; rng("s1#6 token-to-token overlap, f=0 (screen 0.42)", o["f0.0"], 0.42, 0.445); rng("s1#6 token-to-token overlap, f=0.5 (screen 0.69)", o["f0.5"], 0.68, 0.70)
    t.check("s1#6 overlap rises with f and f=1 means every token repeats the first (overlap 1.0, hit > 0.99: degenerate, no quality)", o["f0.0"] < o["f0.25"] < o["f0.5"] < o["f0.75"] < o["f1.0"] == 1.0 and H["f1.0|fp2.0"] > 0.99)
    X = r["throughput"]; rng("s1#6 MEASURED serial gain at f=0.5, 2 fp (screen 1.31)", X["f0.5|fp2.0"]["serial_gain"], 1.30, 1.37); rng("s1#6 MEASURED overlap gain at f=0.5, 2 fp (screen 1.10)", X["f0.5|fp2.0"]["overlap_gain"], 1.09, 1.13)
    t.check("s1#6 every throughput stays at or under the 4.583 tok/s compute ceiling, and the claimed 5-8 tok/s lies above it", all(v["overlap_tps"] <= 4.5834 and v["serial_tps"] <= 4.5834 for v in X.values()) and r["claimed_range_above_ceiling"] is True)
    Wn = r["window_union_fraction"]; t.check("s1#6 window union falls with f, but at f=0.5 a 128-token window still touches ~49% of experts, nowhere near the claimed 64 of 512 (12.5%)", Wn["f0.0|W128"] > Wn["f0.25|W128"] > Wn["f0.5|W128"] > Wn["f0.75|W128"] > Wn["f1.0|W128"] and inr(Wn["f0.5|W128"], 0.46, 0.52))
    rng("s1#6 even at f=0.75 a 128-token window touches", Wn["f0.75|W128"], 0.41, 0.46)
    b = fa.tokens("qwen3-30b_db"); s0 = fa.stick_tokens("qwen3-30b_db", 0.0); s5 = fa.stick_tokens("qwen3-30b_db", 0.5); s1_ = fa.stick_tokens("qwen3-30b_db", 1.0)
    t.check("s1#6 stick_tokens: f=0 reproduces the trace, every layer keeps 8 distinct ids, f=1 repeats token 0's expert sets, same seed gives the same tokens", s0 == b and all(len(set(x)) == 8 for tk in s5 for x in tk) and all([sorted(x) for x in tk] == [sorted(x) for x in s1_[0]] for tk in s1_) and s5 == fa.stick_tokens("qwen3-30b_db", 0.5))
    t.check("s1#6 control that can fail: f=0 hit equals the independent cachesim LRU on the raw trace", abs(fa._hit_for(b, 2.0) - cs.lru_hits([(l, e) for tk in b for l in range(48) for e in tk[l]], 768) / (len(b) * 384)) < 1e-12)

    # ---------------- family 7
    det(fa.fam_s1_7); r = fa.fam_s1_7(); Cc = r["cache"]
    t.check("s1#7 hit rates in [0,1] and gb read per token >= 0", all(0 <= v["hit"] <= 1 and v["gb_per_token_read"] >= 0 for v in Cc.values()))
    t.check("s1#7 fewer bits never lower the hit (inclusion) and a bigger machine never lowers it", all(Cc[f"{ram}GB|4.3"]["hit"] <= Cc[f"{ram}GB|3.0"]["hit"] <= Cc[f"{ram}GB|2.6"]["hit"] <= Cc[f"{ram}GB|2.06"]["hit"] for ram in (8, 16, 32)) and all(Cc[f"8GB|{b_}"]["hit"] <= Cc[f"16GB|{b_}"]["hit"] <= Cc[f"32GB|{b_}"]["hit"] for b_ in (4.3, 3.0, 2.6, 2.06)))
    rng("s1#7 footprints at 8 GB, 4.3 bpw (brief 2.1)", Cc["8GB|4.3"]["footprints"], 2.05, 2.1); rng("s1#7 footprints at 8 GB, 2.6 bpw", Cc["8GB|2.6"]["footprints"], 3.40, 3.45); rng("s1#7 4-bit hit at 8 GB (screen table 0.55)", Cc["8GB|4.3"]["hit"], 0.54, 0.58)
    Gg = r["gain_vs_4bit"]; rng("s1#7 MEASURED 8 GB serial gain, 2.6 bpw (screen 1.8 at a 4.5 GB cache)", Gg["8GB|2.6"]["serial"], 1.42, 1.56); rng("s1#7 8 GB overlap gain, 2.6 bpw (screen 1.1)", Gg["8GB|2.6"]["overlap"], 1.06, 1.11)
    Sg = r["screen_budget_gain_2p6_vs_4p3"]; rng("s1#7 MEASURED serial gain at the screen's own 4.5 GB cache budget (screen 1.8)", Sg["4.5GB"]["serial"], 1.40, 1.50); rng("s1#7 serial gain at 12 GB (screen 1.07)", Sg["12.0GB"]["serial"], 1.02, 1.06); rng("s1#7 serial gain at 26 GB (screen ~1.0)", Sg["26.0GB"]["serial"], 1.02, 1.05)
    t.check("s1#7 16 and 32 GB: overlap gain is 1.0 (compute bound) for every width, serial gain under 1.08", all(Gg[f"{ram}GB|{b_}"]["overlap"] == 1.0 and Gg[f"{ram}GB|{b_}"]["serial"] < 1.08 for ram in (16, 32) for b_ in (3.0, 2.6, 2.06)))
    t.check("s1#7 CAVEAT checked: 16/32 GB hit equals the trace's cold-start ceiling (0.962), so the large-cache serial gains are cold-miss artefacts of a 250-token trace", abs(Cc["32GB|4.3"]["hit"] - r["trace_cold_start_hit_ceiling"]) < 1e-12 and abs(Cc["16GB|2.6"]["hit"] - r["trace_cold_start_hit_ceiling"]) < 1e-12)
    E_ = r["expert_file_gb"]; rng("s1#7 expert file at 2.6 bpw (screen 42)", E_["2.6"], 41, 42.5); rng("s1#7 expert file at 3.0 bpw (screen 48)", E_["3.0"], 47.5, 49); rng("s1#7 expert file at 2.06 bpw (screen 33-34)", E_["2.06"], 32.5, 34)
    t.check("s1#7 REFUTES 'nearly fits in 32 GB': even 2.06 bpw (33 GB) exceeds the 27 GB expert cache", not any(r["whole_file_fits_32gb_cache"].values()) and E_["2.06"] > r["cache_32gb_gb"])
    rng("s1#7 k=2 speculation on 2.6 bpw at 8 GB: bytes per accepted token vs k=1 (>1 = worse; screen: 1.58x bytes per forward)", r["spec_k2_on_2p6bit_8gb"]["k2_a0p7_bytes_per_accepted_vs_k1"], 1.25, 1.45)
    import combo_trace_sim as cts
    t.check("s1#7 quoted perplexity losses match the repo's recorded OLMoE numbers (Q3_K_S +10.4%, Q2_K +26%)", r["relative_ppl_loss_quoted"]["Q3_K_S"] == cts.PPL["bits"][3.0] and r["relative_ppl_loss_quoted"]["Q2_K"] == cts.PPL["bits"][2.5])
    t.check("s1#7 control that can fail: a cache of zero entries hits nothing, and the 8 GB 4-bit hit is below the 2.06 bpw hit by > 0.2", cs.lru_hits([(0, 1), (0, 1)], 0) == 0 and Cc["8GB|2.06"]["hit"] - Cc["8GB|4.3"]["hit"] > 0.2)

    # ---------------- registry
    t.check("FAMILIES has 8 entries fam_s1_0..7, each runnable, with 7/5/7/7/4/5/5/5 catalog method rows (45 total)", sorted(fa.FAMILIES) == [f"fam_s1_{i}" for i in range(8)] and [len(fa.FAMILIES[f"fam_s1_{i}"]["methods"]) for i in range(8)] == [7, 5, 7, 7, 4, 5, 5, 5] and all(callable(v["run"]) and v["needs"] in ("local", "hardware", "model") and v["idea"] for v in fa.FAMILIES.values()))
    t.check("suite finishes within 600 s (runaway guard, not a speed test)", time.time() - t0 < 600, f"{time.time() - t0:.0f} s")
    return t
if __name__ == "__main__": sys.exit(0 if run().summary() else 1)
