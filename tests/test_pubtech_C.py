"""Published-technique families, group C (screen 1 #16-#21, screen 2 #0-#1): deterministic checks on the real Qwen3-30B router traces.
Per family: determinism, bounds, monotonicity, hand-computed unit cases, the recomputed result against the screen's claimed number (range in the check name),
and a control that must be able to fail. Where the recomputation disagrees with the screen, the check passes on the TRUE measured range and the name says DISAGREES."""
import os, sys, tempfile, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "tools", "pubtech"))
from harness import T
import f_C as f

def inr(x, lo, hi): return lo <= x <= hi

def run():
    t = T("pubtech C: screen 1 #16-#21, screen 2 #0-#1")
    R = {}
    for k, v in f.FAMILIES.items(): R[k] = v["run"]()

    # ------------------------------------------------------------------ shared helpers (hand-computed)
    tk = f.tokens("qwen3-30b_db")
    t.check("trace shape: 48 layers x 8 ids per token, 249+ tokens", len(tk) >= 249 and all(len(x) == 48 and all(len(y) == 8 for y in x) for x in tk[:5]), str(len(tk)))
    t.check("runs_of hand case: [1,2,3,7,9] -> (1,3),(7,7),(9,9); gap 1 joins 7 and 9", f.runs_of([9, 1, 2, 3, 7]) == [(1, 3), (7, 7), (9, 9)] and f.runs_of([9, 1, 2, 3, 7], 1) == [(1, 3), (7, 9)])
    t.check("popcount hand case", f.popcount(0b101101) == 4 and f.popcount(0) == 0)
    t.check("tok_s hand case: hit 0 and io 1.0 s -> serial 1/(1+compute), overlap 1.0; hit 1 -> 1/compute both", abs(f.tok_s(0.0, f.GB_TOK)[0] - 1 / (1 + f.T_COMP)) < 1e-9 and abs(f.tok_s(0.0, f.GB_TOK)[1] - 1.0) < 1e-9 and abs(f.tok_s(1.0, 3.0)[0] - 1 / f.T_COMP) < 1e-9)
    t.check("rate_for_size: end points 1.5 and 2.38, clamped, increasing in between", f.rate_for_size(0.5) == 1.5 and abs(f.rate_for_size(4.194304) - 2.38) < 1e-9 and f.rate_for_size(8) == 2.38 and f.rate_for_size(1.0) < f.rate_for_size(2.0) < f.rate_for_size(4.0))
    t.check("unique_random hand case: one token -> k; n tokens never above E or k*n", f.unique_random(128, 8, 1) == 8 and f.unique_random(128, 8, 3) <= 24 and f.unique_random(128, 8, 1000) <= 128)
    t.check("size_power hand case: rate doubles when size quadruples -> exponent 0.5", abs(f.size_power(1, 2, 1, 4) - 0.5) < 1e-9)

    # ------------------------------------------------------------------ screen 1 #16: K independent prompts in lock-step
    r = R["s1_16"]; K = r["K"]
    t.check("s1_16 determinism: two calls give equal dicts", f.fam_s1_16() == r)
    allv = list(r["sharing_same_domain_offsets"].values()) + list(r["sharing_mixed_domains"].values()) + list(r["sharing_random_128"].values())
    t.check("s1_16 sharing is within [1, K] for every K and stream kind", all(1.0 <= v <= k + 1e-9 for d in (r["sharing_same_domain_offsets"], r["sharing_mixed_domains"], r["sharing_random_128"]) for k, v in d.items()), str(allv))
    t.check("s1_16 sharing is non-decreasing in K (same-domain and mixed)", all(r["sharing_same_domain_offsets"][a] <= r["sharing_same_domain_offsets"][b] and r["sharing_mixed_domains"][a] <= r["sharing_mixed_domains"][b] for a, b in zip(K, K[1:])))
    t.check("s1_16 screen's 2 unrelated aligned streams: sharing in [1.04, 1.08] (screen 1.06)", inr(r["streams_aligned_2_unrelated"], 1.04, 1.08), str(r["streams_aligned_2_unrelated"]))
    t.check("s1_16 screen's 3 unrelated aligned streams: sharing in [1.09, 1.15] (screen 1.12)", inr(r["streams_aligned_3_unrelated"], 1.09, 1.15), str(r["streams_aligned_3_unrelated"]))
    t.check("s1_16 mixed-domain sharing never above same-domain (same-domain is the optimistic bound), K=4..32", all(r["sharing_mixed_domains"][k] <= r["sharing_same_domain_offsets"][k] + 1e-9 for k in (4, 8, 16, 32)))
    t.check("s1_16 real traces share MORE than independent random routing on 128 experts at K=8 (range same-domain [1.7, 2.1] vs random [1.2, 1.3])", inr(r["sharing_same_domain_offsets"][8], 1.7, 2.1) and inr(r["sharing_random_128"][8], 1.2, 1.3))
    t.check("s1_16 kill rule 'same-domain K=8 sharing below 1.5' is NOT triggered on 128 experts (measured 1.9; screen expected 1.2-1.5: DISAGREES, higher; one-prompt optimistic bound)", r["kill_same_domain_k8_below_1p5"] is False)
    e = r["extrapolation_125B"]
    t.check("s1_16 125B K=1 reads exactly 10 experts per layer = 1.296 GB; unique <= min(512, 10K) for all K", e[1]["unique_per_layer"] == 10.0 and abs(e[1]["step_gb"] - 1.296) < 1e-3 and all(v["unique_per_layer"] <= min(512, 10 * k) + 1e-6 for k, v in e.items()))
    t.check("s1_16 125B aggregate gain K=8 in [1.05, 1.09] (screen 1.07), K=32 in [1.30, 1.37] (screen ~1.35), K=128 in [2.6, 2.8] (screen ~2.7)",
            inr(r["best_path_aggregate_gain_k8"], 1.05, 1.09) and inr(r["best_path_aggregate_gain_k32"], 1.30, 1.37) and inr(e[128]["agg_gain_vs_single_stream_same_disk"], 2.6, 2.8))
    t.check("s1_16 per-user speed at K=32 is in [0.03, 0.08] of single-user (screen 0.03-0.08)", inr(r["per_user_fraction_k32"], 0.03, 0.08), str(r["per_user_fraction_k32"]))
    t.check("s1_16 aggregate tok/s rises with K but never reaches the K-fold claim: K=32 gain < 2 (claim was 6x/100x)", e[32]["agg_gain_vs_single_stream_same_disk"] < 2.0 and all(e[a]["agg_tok_s"] <= e[b]["agg_tok_s"] for a, b in zip((1, 2, 4, 8, 16, 32), (2, 4, 8, 16, 32, 64))))
    f._MASK["__idA"] = [[0xFF] * 48] * 5; f._MASK["__disjA"] = [[0xFF] * 48] * 5; f._MASK["__disjB"] = [[0xFF00] * 48] * 5
    t.check("s1_16 control: the statistic can fail - identical streams give unique 8 (sharing 2), disjoint streams give unique 16 (sharing 1)",
            f.union_mean([("__idA", 0), ("__idA", 0)], 3) == 8.0 and f.union_mean([("__disjA", 0), ("__disjB", 0)], 3) == 16.0)

    # ------------------------------------------------------------------ screen 1 #17: B tokens together, layer by layer
    r = R["s1_17"]
    t.check("s1_17 determinism: two calls give equal dicts", f.fam_s1_17() == r)
    fs, fm, fr = r["factor_same_domain"], r["factor_mixed"], r["factor_random_128"]
    t.check("s1_17 union factor in (0, 1] and strictly decreasing in B (same-domain and mixed)", all(0 < v <= 1 for v in list(fs.values()) + list(fm.values())) and all(fs[a] > fs[b] and fm[a] > fm[b] for a, b in zip(r["B"], r["B"][1:])))
    t.check("s1_17 same-domain <= mixed <= independent-random(128) at B=2,4,8 (shared structure lowers the union)", all(fs[b] <= fm[b] <= fr[b] + 0.005 for b in (2, 4, 8)), f"{fs} {fm} {fr}")
    t.check("s1_17 screen's same-domain factors: B=2 in [0.82, 0.90] (screen 0.86-0.88), B=4 in [0.66, 0.75] (0.68-0.73), B=8 in [0.47, 0.58] (0.49-0.55)", inr(fs[2], 0.82, 0.90) and inr(fs[4], 0.66, 0.75) and inr(fs[8], 0.47, 0.58), str(fs))
    t.check("s1_17 screen's mixed-trace factors: B=2 in [0.91, 0.97] (0.94), B=4 in [0.77, 0.85] (0.81), B=8 in [0.57, 0.66] (0.61)", inr(fm[2], 0.91, 0.97) and inr(fm[4], 0.77, 0.85) and inr(fm[8], 0.57, 0.66), str(fm))
    t.check("s1_17 consecutive tokens share most: B=4 in [0.57, 0.66] (screen 0.62), B=8 in [0.41, 0.50] (0.46)", inr(r["factor_consecutive"][4], 0.57, 0.66) and inr(r["factor_consecutive"][8], 0.41, 0.50))
    t.check("s1_17 kill rule 'B=4 same-domain factor above 0.85' is not triggered; measured 0.70 vs the screen's expected 0.8-0.95: DISAGREES (more sharing on 128 experts, one-prompt bound)", r["killed_by_b4_same_above_0p85"] is False and inr(fs[4], 0.66, 0.75))
    t.check("s1_17 independent random on 512 experts top-10: B=8 factor in [0.93, 0.94] (hand: 512(1-(1-10/512)^8)/80)", inr(r["factor_random_512"][8], 0.93, 0.94), str(r["factor_random_512"][8]))
    s, o, gs, go = f.batch_throughput(1, 1.0, 0.0)
    t.check("s1_17 hand case: B=1, factor 1, hit 0 reproduces single-token serial tok/s exactly and gain 1.0", abs(s - f.tok_s(0.0, f.RATE_4MIB)[0]) < 1e-9 and abs(gs - 1.0) < 1e-9 and abs(go - 1.0) < 1e-9)
    th = r["throughput"]
    t.check("s1_17 aggregate tok/s never exceeds the flat compute ceiling 4.58 and serial <= overlap", all(v["agg_overlap"] <= 1 / f.T_COMP + 1e-6 and v["agg_serial"] <= v["agg_overlap"] + 1e-9 for v in th.values()))
    t.check("s1_17 batching vs the best path at 8 GB (hit 0 batch vs LRU single): gain below 1 for every B<=8 (range [0.4, 0.95]) - the cache is lost", all(inr(v["gain_vs_best_serial"], 0.4, 0.95) for v in th.values()))
    cg = r["cached_gain_8gb"]
    t.check("s1_17 with the SAME LRU cache for batch and single stream: aggregate gain at B=2 in [0.95, 1.1], B=4 in [0.5, 0.85], B=8 in [0.65, 1.0] (below the screen's 1.3-1.7x at B=8: DISAGREES)",
            inr(cg["B2_same"]["serial_gain_same_cache"], 0.95, 1.1) and inr(cg["B4_same"]["serial_gain_same_cache"], 0.5, 0.85) and inr(cg["B8_same"]["serial_gain_same_cache"], 0.65, 1.0) and cg["B8_same"]["overlap_gain_same_cache"] < 1.3)
    t.check("s1_17 control: batching a stream with itself shares perfectly (B identical streams -> factor 1/B), a statistic that can fail", abs(f.union_mean([("__idA", 0)] * 4, 2) / (8 * 4) - 0.25) < 1e-9)

    # ------------------------------------------------------------------ screen 1 #18: explicit pread, no cache
    r = R["s1_18"]; rec = r["by_rate"]["recorded_pread"]; asm = r["by_rate"]["screen_assumed"]
    t.check("s1_18 determinism: two calls give equal dicts", f.fam_s1_18() == r)
    t.check("s1_18 all hit rates, throughputs and ratios are in range (hit in [0,1], tok/s > 0)", all(0 <= v["lru_hit"] <= 1 and v["nocache_serial"] > 0 and v["mmap_serial"] > 0 for by in r["by_rate"].values() for v in by["machines"].values()))
    t.check("s1_18 merging adjacent experts saves reads in [0.04, 0.08] (screen 5-6%): reads per slice 0.93-0.95 for each trace", inr(r["merge_saving"], 0.04, 0.08) and all(inr(v, 0.92, 0.96) for v in r["reads_per_slice_after_merge"].values()), str(r["reads_per_slice_after_merge"]))
    h = r["lru_hit_by_footprints"]
    t.check("s1_18 LRU hit: 0 below one footprint (0.9), 1.0 -> [0.28, 0.33], 2.0 -> [0.52, 0.59], 3.0 -> [0.64, 0.73], 7.5 -> [0.93, 0.96] (screen's values), non-decreasing", h[0.9] == 0.0 and inr(h[1.0], 0.28, 0.33) and inr(h[2.0], 0.52, 0.59) and inr(h[3.0], 0.64, 0.73) and inr(h[7.5], 0.93, 0.96) and all(a <= b for a, b in zip(sorted(h.values()), sorted(h.values())[1:])))
    t.check("s1_18 bytes per token with no cache = 1.296 GB (480 slices x 2.7 MB)", abs(r["disk_gb_per_token_nocache"] - 1.296) < 1e-3)
    m8a = asm["machines"]["8GB"]
    t.check("s1_18 with the screen's 1.5 GB/s: 8 GB mmap serial in [0.50, 0.57] (screen 0.53); no-cache serial in [0.90, 0.95] (0.92); gain over mmap serial [1.6, 1.85] (1.7x), overlap [1.8, 2.05] (1.9x)",
            inr(m8a["mmap_serial"], 0.50, 0.57) and inr(m8a["nocache_serial"], 0.90, 0.95) and inr(m8a["nocache_over_mmap_serial"], 1.6, 1.85) and inr(m8a["nocache_over_mmap_overlap"], 1.8, 2.05))
    t.check("s1_18 at 16 and 32 GB no-cache pread loses to mmap-as-LRU (ratio < 1) with either read rate (screen 0.55x / 0.27x)", all(by["machines"][g]["nocache_over_mmap_serial"] < 1.0 for by in r["by_rate"].values() for g in ("16GB", "32GB")))
    t.check("s1_18 with the RECORDED per-layer pread rate (2.48 GB/s, higher than the 1.5 assumed): 8 GB gain over mmap serial in [2.3, 2.7], 'DISAGREES' upward with the screen's 1.7x", inr(rec["machines"]["8GB"]["nocache_over_mmap_serial"], 2.3, 2.7))
    t.check("s1_18 without a cache the idea stays below our best path (explicit reads + LRU) on every machine and rate", all(v["nocache_over_best_serial"] < 1.0 for by in r["by_rate"].values() for v in by["machines"].values()))
    t.check("s1_18 recorded mmap cold replay (0.277 GB/s) is the same order as the 0.34 GB/s paging figure (range [0.2, 0.4])", inr(r["recorded"]["mmap_cold"], 0.2, 0.4))
    d = tempfile.mkdtemp()
    try:
        gp = f.make_synthetic_gguf(os.path.join(d, "syn.gguf"), layers=48, experts=8, slice_bytes=4096)
        ts = f.read_gguf_tensors(gp); tab = f.expert_table(ts)
        t.check("s1_18 GGUF parser on a synthetic file: 144 tensors, expert dim 8, slice 4096 B, offsets 32-aligned and increasing", len(ts) == 144 and tab[(0, "gate")][2] == 8 and tab[(0, "gate")][1] == 4096 and all(v["offset"] % 32 == 0 for v in ts.values()) and ts["blk.1.ffn_up_exps.weight"]["offset"] > ts["blk.0.ffn_down_exps.weight"]["offset"])
        off = tab[(3, "down")][0] + 5 * tab[(3, "down")][1]
        with open(gp, "rb") as fh: fh.seek(off); b = fh.read(4)
        d0 = ts["blk.0.ffn_gate_exps.weight"]["offset"]
        t.check("s1_18 GGUF parser: the byte at a slice offset equals the planted pattern", list(b) == [((off - d0) + i) % 251 for i in range(4)], str(list(b)))
        toks = [[[1, 2, 5]] + [[0, 4, 6]] * 47]
        pm = f.plan_reads(toks, tab, True); pu = f.plan_reads(toks, tab, False)
        t.check("s1_18 plan_reads hand case: layer 0 ids {1,2,5} merge to 2 runs x 3 tensors = 6 reads vs 9 unmerged; bytes equal", len(pm[0]) == 6 and len(pu[0]) == 9 and f.plan_stats(pm)["bytes"] == f.plan_stats(pu)["bytes"] == 48 * 3 * 3 * 4096)
        rp = f.replay_pread(gp, pm, 4); rm = f.replay_mmap(gp, pm)
        t.check("s1_18 replay_pread and replay_mmap account exactly the requested bytes on a synthetic file", rp["bytes"] == f.plan_stats(pm)["bytes"] and rp["reads"] == f.plan_stats(pm)["reads"] and rm["bytes"] == f.plan_stats(pm)["bytes"] and rp["gb_s"] > 0)
    finally: shutil.rmtree(d, ignore_errors=True)
    t.check("s1_18 control: more cache never lowers tok/s, and a hit rate of 1 reaches the compute ceiling (the model can fail)", f.tok_s(0.6, 2.0)[0] > f.tok_s(0.3, 2.0)[0] and abs(f.tok_s(1.0, 2.0)[1] - 1 / f.T_COMP) < 1e-9)

    # ------------------------------------------------------------------ screen 1 #19: repack triples + cluster
    r = R["s1_19"]; sc = r["scenarios"]
    t.check("s1_19 determinism: two calls give equal dicts", f.fam_s1_19() == r)
    t.check("s1_19 reads per layer are in (0, 8] for every scenario and trace", all(0 < v <= 8 for s in sc.values() for v in s["per_trace"].values()))
    t.check("s1_19 ordering: in-sample < same-prompt halves < held-out(2) < cross(1) < identity (mean reads)", sc["in_sample_not_valid"]["mean"] < sc["same_prompt_first_half_to_second"]["mean"] < sc["held_out_two_others"]["mean"] < sc["cross_prompt_one_other"]["mean"] < sc["identity"]["mean"])
    t.check("s1_19 stock layout reads per layer in [7.4, 7.65] (screen 7.5)", inr(r["identity_reads"], 7.4, 7.65))
    cp = sc["cross_prompt_one_other"]["per_trace"]
    t.check("s1_19 cross-prompt reads per layer: each trace in [5.5, 6.6] (screen 6.1 / 6.4 / 5.6), mean in [5.9, 6.4]", all(inr(v, 5.5, 6.6) for v in cp.values()) and inr(r["cross_prompt_reads"], 5.9, 6.4), str(cp))
    t.check("s1_19 clustering cuts reads by [0.15, 0.25] across prompts (screen 15-25%); measured 18.7%", inr(r["read_reduction_cross"], 0.15, 0.25), str(r["read_reduction_cross"]))
    t.check("s1_19 same-prompt first-half -> second-half reads in [4.7, 5.5] (screen 4.6 / 4.6 / 5.1; measured slightly higher: DISAGREES by about 0.3)", inr(sc["same_prompt_first_half_to_second"]["mean"], 4.7, 5.5))
    t.check("s1_19 kill rule 'cross-prompt clustering leaves more than 6 reads per layer' IS triggered, marginally (6.125 > 6.0)", r["kill_cross_prompt_reads_above_6"] is True and inr(r["cross_prompt_reads"], 6.0, 6.3))
    scr = r["by_rates"]["screen_expectation"]["machines"]; rc = r["by_rates"]["recorded"]["machines"]
    t.check("s1_19 repack (3 reads -> 1) with the screen's rates (1.7 -> 2.38 GB/s): 8 GB serial gain in [1.15, 1.25], overlap in [1.35, 1.45] (screen 1.2-1.4x over the best path)", inr(scr["8GB"]["repack_gain_serial"], 1.15, 1.25) and inr(scr["8GB"]["repack_gain_overlap"], 1.35, 1.45))
    t.check("s1_19 repack with the RECORDED rates (2.9 -> 3.8 GB/s): 8 GB serial gain in [1.08, 1.17], overlap exactly 1.0 (compute-capped): DISAGREES with the screen's 1.2-1.4x", inr(rc["8GB"]["repack_gain_serial"], 1.08, 1.17) and rc["8GB"]["repack_gain_overlap"] == 1.0)
    t.check("s1_19 at 16 and 32 GB the repack gain is below 1.05 under both rate sets (screen 1.0-1.1x)", all(m[g]["repack_gain_serial"] < 1.05 for m in (scr, rc) for g in ("16GB", "32GB")))
    t.check("s1_19 cluster-only extra gain is below 1.08 everywhere (screen up to 1.1x)", all(m[g]["cluster_extra_serial"] < 1.08 and m[g]["cluster_extra_overlap"] < 1.08 for m in (scr, rc) for g in ("8GB", "16GB", "32GB")))
    toks = [[[3, 4, 7]]]; pos = [[0, 1, 2, 3, 7, 6, 5, 4]]
    t.check("s1_19 hand case: ids {3,4,7} on slots 3,7,4 -> runs {3-4},{7} = 2 reads; with the reorder slots {7,4,3}: reads 2 at gap 0 and 1 at gap 3", f.layout_reads(toks, None)[0] == 2 and f.layout_reads(toks, None, 3)[0] == 1)
    ch = f.coact_chain([[[[0, 2]]] * 5 + [[[1, 3]]] * 3], 0, 4)
    t.check("s1_19 coact_chain hand case: always-together experts 0,2 end adjacent, as do 1,3 (order 0,2,1,3)", ch == [0, 2, 1, 3] or (abs(ch[0] - ch[2]) == 1 and abs(ch[1] - ch[3]) == 1), str(ch))
    rng = __import__("random").Random(5); perm = []
    for l in range(48): p = list(range(128)); rng.shuffle(p); perm.append(p)
    rnd = sum(f.layout_reads(f.tokens(n), perm)[0] for n in f.TRACES) / 3
    t.check("s1_19 control: a RANDOM expert order does not cluster - reads in [7.2, 7.8], well above the held-out layout's 5.9", inr(rnd, 7.2, 7.8) and rnd > r["held_out_reads"] + 1.0, str(rnd))
    d = tempfile.mkdtemp()
    try:
        gp = f.make_synthetic_gguf(os.path.join(d, "syn.gguf"), layers=48, experts=128, slice_bytes=4096)
        res = f.bench_layout(gp, pairs=40, threads=2, scratch_dir=d, seed=1, reps=2)
        t.check("s1_19 bench_layout dry run on a synthetic file: three scattered reads and one contiguous read move the same bytes, scratch file removed", res["triple_mb"] > 0 and res["scattered_3reads_gb_s"] > 0 and res["contiguous_1read_gb_s"] > 0 and os.listdir(d) == ["syn.gguf"], str(os.listdir(d)))
    finally: shutil.rmtree(d, ignore_errors=True)

    # ------------------------------------------------------------------ screen 1 #20: reorder GGUF by co-activation
    r = R["s1_20"]
    t.check("s1_20 determinism: two calls give equal dicts", f.fam_s1_20() == r)
    t.check("s1_20 reads per layer in (0, 8] and run length = 8 / reads for identity, held-out, in-sample", all(0 < r[k] <= 8 for k in ("reads_identity", "reads_held_out", "reads_in_sample")) and abs(r["run_length_held_out"] - 8 / r["reads_held_out"]) < 0.01)
    t.check("s1_20 stock reads per layer per trace in [7.4, 7.65] (screen 7.48 / 7.58 / 7.54)", all(inr(v, 7.4, 7.65) for v in r["per_trace_reads_identity"].values()))
    t.check("s1_20 held-out reads per trace in [5.5, 6.3] (screen 5.77 / 6.21 / 6.06)", all(inr(v, 5.5, 6.3) for v in r["per_trace_reads_held_out"].values()), str(r["per_trace_reads_held_out"]))
    t.check("s1_20 run lengths: identity in [1.04, 1.08] (1.06), held-out in [1.28, 1.42] (screen 1.29-1.39), in-sample in [1.7, 1.9] (1.77-1.81)", inr(r["run_length_identity"], 1.04, 1.08) and inr(r["run_length_held_out"], 1.28, 1.42) and inr(r["run_length_in_sample"], 1.7, 1.9))
    g = r["gap_merge_no_reorder"]
    t.check("s1_20 merging across gaps without reorder: gap 1 run length in [1.10, 1.15] (screen 1.12-1.13), gap 3 in [1.23, 1.29] (1.25-1.26); slots read grow with the gap", inr(g[1]["run_length"], 1.10, 1.15) and inr(g[3]["run_length"], 1.23, 1.29) and g[0]["slots_per_layer"] < g[1]["slots_per_layer"] < g[3]["slots_per_layer"])
    t.check("s1_20 kill rule 'held-out run length below 1.5' IS triggered (1.35 on 30B)", r["kill_held_out_run_length_below_1p5"] is True)
    ac = r["after_cache"]
    t.check("s1_20 reads that remain after the LRU: miss run length reordered >= stock >= 1.0 on all machines; stock misses are almost never adjacent (<= 1.05)", all(ac[g_]["miss_run_length_reordered"] >= ac[g_]["miss_run_length_stock"] >= 1.0 and ac[g_]["miss_run_length_stock"] <= 1.05 for g_ in ac))
    t.check("s1_20 tok/s gain over explicit reads + LRU at 8 GB: serial in [1.02, 1.08], overlap in [1.03, 1.10] (screen ~1.1x; measured smaller: DISAGREES slightly)", inr(ac["8GB"]["serial_gain"], 1.02, 1.08) and inr(ac["8GB"]["overlap_gain"], 1.03, 1.10), str(ac["8GB"]))
    t.check("s1_20 at 16 and 32 GB the gain is below 1.01 (screen 1.0-1.05x)", all(ac[g_]["serial_gain"] < 1.01 and ac[g_]["overlap_gain"] < 1.01 for g_ in ("16GB", "32GB")))
    t.check("s1_20 disk-rate gain is at most 1.1 on every machine (screen 1.1-1.15x of disk rate)", all(ac[g_]["disk_rate_gain"] <= 1.1 for g_ in ac))
    t.check("s1_20 control: a random expert order gives no gain (reads within 0.3 of stock) while the held-out chain gains more than 1.0 read", abs(rnd - r["reads_identity"]) < 0.3 and r["reads_identity"] - r["reads_held_out"] > 1.0)
    t.check("s1_20 lru_miss_sets hand case: listed misses = counted misses <= accesses, and a cache of 0 entries misses every access", (lambda ms: ms[2] <= ms[1] and sum(len(m) for _, m in ms[0]) == ms[2])(f.lru_miss_sets("qwen3-30b_db", 2.0)) and f.lru_miss_sets("qwen3-30b_db", 0.0)[2] == f.lru_miss_sets("qwen3-30b_db", 0.0)[1])

    # ------------------------------------------------------------------ screen 1 #21: draft + verify
    r = R["s1_21"]; rows = r["rows"]
    t.check("s1_21 determinism: two calls give equal dicts", f.fam_s1_21() == r)
    t.check("s1_21 all bytes ratios and speedups are positive and finite, 36 scenarios x 3 cache sizes", len(rows) == 4 * 3 * 3 and all(0 < v["bytes_ratio"] < 10 and 0 < v["speedup_flat"] < 5 and 0 < v["speedup_fast_small_batch"] < 10 for v in rows.values()))
    t.check("s1_21 K=1 bytes ratio in [0.96, 1.01] (screen 0.96-1.0): the IO kill condition holds", inr(r["bytes_ratio_k1_range"][0], 0.96, 1.01) and inr(r["bytes_ratio_k1_range"][1], 0.99, 1.01), str(r["bytes_ratio_k1_range"]))
    t.check("s1_21 K>=2 bytes ratio reaches [3.9, 4.4] at the small cache (screen 'up to 4.17')", inr(r["bytes_ratio_k_ge2_range"][1], 3.9, 4.4), str(r["bytes_ratio_k_ge2_range"]))
    t.check("s1_21 at 2 footprints, a=0.7: bytes ratio grows with K (K=1 < K=2 < K=3) - wider windows read more", rows["fp2.0_K1_a0.7"]["bytes_ratio"] < rows["fp2.0_K2_a0.7"]["bytes_ratio"] < rows["fp2.0_K3_a0.7"]["bytes_ratio"])
    t.check("s1_21 a bigger cache never makes the window cost more: ratio at 7.5 footprints <= at 2 footprints for K=3, 7 (a=0.7, 0.9)", all(rows[f"fp7.5_K{k}_a{a}"]["bytes_ratio"] <= rows[f"fp2.0_K{k}_a{a}"]["bytes_ratio"] + 1e-9 for k in (3, 7) for a in (0.7, 0.9)))
    t.check("s1_21 with flat compute (a K+1 pass costs K+1 decodes) speedup never exceeds 1.0: max in [0.95, 1.0] (screen 0.24-1.0)", inr(r["speedup_flat_max"], 0.95, 1.0) and inr(r["speedup_flat_min"], 0.2, 0.3), str((r["speedup_flat_min"], r["speedup_flat_max"])))
    t.check("s1_21 unproven fast-small-batch bound (311 GFLOP/s) tops out in [3.4, 3.9] at 7.5 footprints (screen 'up to 3.9x') and 1.6-1.7 at 2 footprints (screen 1.68)", inr(r["speedup_fast_max"], 3.4, 3.9) and inr(rows["fp2.0_K1_a0.9"]["speedup_fast_small_batch"], 1.5, 1.8))
    t.check("s1_21 higher acceptance never lowers the flat-compute speedup (K=3, 2 footprints: a=0.5 <= 0.7 <= 0.9)", rows["fp2.0_K3_a0.5"]["speedup_flat"] <= rows["fp2.0_K3_a0.7"]["speedup_flat"] <= rows["fp2.0_K3_a0.9"]["speedup_flat"])
    f._TOK["__s"] = [[[l, l + 1, l + 2, l + 3, l + 4, l + 5, l + 6, l + 7] for l in range(48)] for _ in range(10)]
    s0 = f.spec_cycle_sim("__s", 100.0, 3, 0.0, 1); s1 = f.spec_cycle_sim("__s", 100.0, 3, 1.0, 1)
    t.check("s1_21 hand case: acceptance 0 advances one token per pass (10 passes); acceptance 1 with K=3 advances 4 (3 passes for 10 tokens); identical tokens -> misses = 48 x 8 distinct", s0["passes"] == 10 and s1["passes"] == 3 and s0["misses"] == s1["misses"] == 48 * 8, str((s0, s1)))
    t.check("s1_21 spec_time hand case: no misses, K=0, one pass per token -> speedup exactly 1.0", abs(f.spec_time(0.0, dict(tokens=10, passes=10, misses=0), 0, None)[0] - 1.0) < 1e-9)
    t.check("s1_21 bench_ratio hand case: decode 20 t/s, batch-4 prompt 40 t/s -> relative per-token cost 0.5", f.bench_ratio([dict(n_prompt=0, n_gen=32, avg_ts=20.0), dict(n_prompt=4, n_gen=0, avg_ts=40.0)]) == {4: 0.5})
    sa = f.spec_cycle_sim("qwen3-30b_db", 2.0, 3, 0.0, 1)
    t.check("s1_21 control: with acceptance 0 the window cost is pure waste - flat-compute speedup below 0.3 at K=3 (the model can fail)", f.spec_time(f.lru_miss_sets_count("qwen3-30b_db", 2.0), sa, 3, None)[0] < 0.3)

    # ------------------------------------------------------------------ screen 2 #0: NPU for prefill
    r = R["s2_0"]
    t.check("s2_0 determinism: two calls give equal dicts", f.fam_s2_0() == r)
    t.check("s2_0 compute time of a 500-token prefill at 311 GFLOP/s in [19.0, 19.6] s (screen 19.3); at the 55 GFLOP/s decode rate 109 s (the claim's fictional baseline)", inr(r["compute_s_at_311"], 19.0, 19.6) and inr(r["compute_s_at_55_decode_rate"], 108.5, 109.6))
    t.check("s2_0 I/O time of 69 GB: 28.8 s at 2.4, 22.3 s at 3.1, 44.2 s at 1.56 GB/s (screen), ranges [28.5, 29], [22.0, 22.5], [44, 44.5]", inr(r["io_s"]["2.4"], 28.5, 29) and inr(r["io_s"]["3.1"], 22.0, 22.5) and inr(r["io_s"]["1.56"], 44.0, 44.5))
    tt = r["ttft"]
    t.check("s2_0 overlapped TTFT: the NPU speedup is exactly 1.0 at all three disk rates (I/O hides compute)", all(v["overlap_speedup"] == 1.0 for v in tt.values()))
    t.check("s2_0 I/O time over compute time in [1.1, 2.4] (screen 1.2-2.3)", all(inr(v["io_over_compute"], 1.1, 2.4) for v in tt.values()))
    t.check("s2_0 baseline error: the claimed 109 s uses the decode rate, 5.5-5.8x too slow for prefill; claimed ratio 109/29 = 3.76", inr(r["baseline_error_factor"], 5.5, 5.8) and inr(r["claim_ratio_109_over_29"], 3.7, 3.8))
    t.check("s2_0 without overlap (serial) the NPU would give [1.2, 1.5]x - the only way the idea helps at 500 tokens", all(inr(v["serial_speedup"], 1.2, 1.5) for v in tt.values()))
    t.check("s2_0 break-even prompt length where compute equals I/O: [570, 1150] tokens; at 2000 tokens speedup [2.5, 2.8], at 8000 [2.7, 3.0]", all(inr(v["break_even_prompt_tokens"], 570, 1150) for v in tt.values()) and inr(r["ttft_by_prompt_length_2p4"][2000]["speedup"], 2.5, 2.8) and inr(r["ttft_by_prompt_length_2p4"][8000]["speedup"], 2.7, 3.0))
    cv = r["coverage"]
    t.check("s2_0 experts touched by n consecutive real tokens grow with n and stay in [0.2, 0.7] up to 200 tokens, far below the random model (1.0 at n=100 on 128 experts): DISAGREES with 'nearly every expert once'",
            all(cv[10]["real_30b"][k] < cv[50]["real_30b"][k] < cv[100]["real_30b"][k] < cv[200]["real_30b"][k] for k in cv[10]["real_30b"]) and all(inr(v, 0.2, 0.7) for n in (10, 50, 100, 200) for v in cv[n]["real_30b"].values()) and cv[100]["random_128"] > 0.99)
    cs_ = r["coverage_scenarios"]
    t.check("s2_0 if a prompt touched only 60% of experts and reads ran at 3.1 GB/s, prefill would turn compute-bound and an NPU would gain [1.4, 1.6]x; at full coverage 1.0x", inr(cs_["cov0.6_rate3.1"]["speedup"], 1.4, 1.6) and cs_["cov1.0_rate2.4"]["speedup"] == 1.0)
    a = f.ttft(1000, 12.0, 10.0, 1.0)
    t.check("s2_0 ttft hand case: 1000 tokens at 12 GFLOP/s of 12 GFLOP/token = 1000 s compute, 10 GB at 1 GB/s = 10 s I/O -> overlap 1000 s, serial 1010 s", abs(a[0] - 1000) < 1e-6 and abs(f.ttft(1000, 12.0, 10.0, 1.0, False)[0] - 1010) < 1e-6)
    t.check("s2_0 control: the kill criterion can fail - if the CPU prefill were only 50 GFLOP/s the NPU gain at 500 tokens would be above 4x", f.ttft(500, 50.0, 69.0, 2.4)[0] / f.ttft(500, 900.0, 69.0, 2.4)[0] > 4.0)

    # ------------------------------------------------------------------ screen 2 #1: computational-storage drives
    r = R["s2_1"]
    t.check("s2_1 determinism: two calls give equal dicts", f.fam_s2_1() == r)
    t.check("s2_1 arithmetic: 1 NVMe in [2.38, 2.40] tok/s, 2 NVMe in [4.77, 4.80] (screen 2.4 / 4.8), compute ceiling 55/12 in [4.58, 4.59], capped value = ceiling", inr(r["one_nvme_tok_s"], 2.38, 2.40) and inr(r["two_nvme_tok_s"], 4.77, 4.80) and inr(r["compute_ceiling_tok_s"], 4.58, 4.59) and r["two_nvme_capped_tok_s"] == r["compute_ceiling_tok_s"])
    t.check("s2_1 per-drive in-device I/O 0.65 GB per token = 0.21 s, just under the 0.218 s compute time (screen: the same)", inr(r["csd_per_drive_gb_per_token"], 0.64, 0.66) and inr(r["csd_per_drive_io_s"], 0.20, 0.215) and r["csd_per_drive_io_s"] < r["compute_s"])
    t.check("s2_1 two striped NVMe drives equal the ideal CSD I/O time (no FPGA gain over striping)", r["striping_equals_csd_ideal"] is True)
    ib = r["imbalance_real_128_top8"]
    t.check("s2_1 per-layer imbalance (busiest drive over mean) is >= 1 and in [1.20, 1.33] on the real traces for half/mod/hash placement, [1.18, 1.30] for random 512 top-10", all(inr(v["mean_layer_max_over_mean"], 1.20, 1.33) for v in ib.values()) and inr(r["imbalance_random_512_top10"], 1.18, 1.30))
    t.check("s2_1 with imbalance the CSD pair reaches [3.6, 4.2] tok/s overlapped and [1.9, 2.3] serial (screen 'realistic 3-4'), below the ideal 4.58", inr(r["csd"]["tok_s_overlap_imbalanced"], 3.6, 4.2) and inr(r["csd"]["tok_s_serial_imbalanced"], 1.9, 2.3) and r["csd"]["tok_s_overlap_ideal"] > r["csd"]["tok_s_overlap_imbalanced"])
    t.check("s2_1 ideal gain over our best path at 8 GB: serial in [2.0, 2.2] (screen 2.1x), overlapped in [1.05, 1.15] (screen 1.1x)", inr(r["csd_ideal_gain_over_best_8gb_serial"], 2.0, 2.2) and inr(r["csd_ideal_gain_over_best_8gb_overlap"], 1.05, 1.15))
    pd = r["per_drive_rate_scenarios"]
    t.check("s2_1 even at 7 GB/s per drive (the screen's survival threshold) the ideal rate is the same compute ceiling as at 4.14 GB/s: 4.58 tok/s both", pd["7.0"]["tok_s_overlap_ideal"] == pd["4.14"]["tok_s_overlap_ideal"] == r["compute_ceiling_tok_s"])
    t.check("s2_1 best path at 16 and 32 GB already reaches the ceiling overlapped (4.58): a CSD adds 1.0x there", r["best_path"]["16GB"]["overlap"] == r["compute_ceiling_tok_s"] and r["best_path"]["32GB"]["overlap"] == r["compute_ceiling_tok_s"])
    th = r["recorded_nvme_threads"]
    t.check("s2_1 recorded read rate keeps rising past 4 in flight (16 in flight > 4 in flight by more than 10%): DISAGREES with the screen's 'flat beyond 4'", th[16] > th[4] * 1.1 and th[1] < th[4] < th[8])
    f._TOK["__b"] = [[[0, 1, 2, 3, 4, 5, 6, 7]] * 48] * 3; f._TOK["__c"] = [[[0, 1, 2, 3, 64, 65, 66, 67]] * 48] * 3
    t.check("s2_1 drive_balance hand cases: all 8 experts on drive 0 -> ratio 2.0; 4 + 4 -> 1.0", f.drive_balance("__b", 2, "half")[0] == 2.0 and f.drive_balance("__c", 2, "half")[0] == 1.0)
    t.check("s2_1 random_balance hand case: when k = E every drive holds exactly its share -> 1.0", f.random_balance(4, 4, 2, 50, 1) == 1.0)
    d = tempfile.mkdtemp()
    try:
        gp = f.make_synthetic_gguf(os.path.join(d, "syn.gguf"), layers=48, experts=8, slice_bytes=4096)
        bt = f.bench_threads(gp, 0.01, (1, 2), 0.15)
        t.check("s2_1 bench_threads dry run on a synthetic file returns a positive rate per thread count", set(bt) == {1, 2} and all(v > 0 for v in bt.values()), str(bt))
    finally: shutil.rmtree(d, ignore_errors=True)
    t.check("s2_1 control: a placement with every expert on one drive (imbalance 2.0) would drop the overlapped rate below 3.0 tok/s, so the 3.6-4.2 above is not forced", 1 / max(r["csd"]["io_s_ideal"] * 2.0, r["compute_s"]) < 3.0)

    # ------------------------------------------------------------------ catalog
    names = [m for v in f.FAMILIES.values() for m in v["methods"]]
    t.check("FAMILIES lists 8 families and 29 distinct method names, each with a run function and needs=local", len(f.FAMILIES) == 8 and len(names) == 29 and len(set(names)) == len(names) and all(callable(v["run"]) and v["needs"] == "local" for v in f.FAMILIES.values()), str(len(names)))
    return t

if __name__ == "__main__": sys.exit(0 if run().summary() else 1)
