"""Published-technique family tests, group E (second screen, families 10 to 17): determinism, bounds, hand-computed unit cases,
measured result vs the screen's claim (ranges come from the run and sit in the check names), and controls that can fail. No network, no models."""
import math, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "tools")); sys.path.insert(0, os.path.join(HERE, "..", "tools", "pubtech"))
from harness import T
import f_E as f

def inr(x, lo, hi): return lo <= x <= hi
def toy(ids_per_token):          # tokens with the same ids at every one of the 48 layers
    return [[list(ids) for _ in range(f.NL)] for ids in ids_per_token]

def run():
    t = T("pubtech E (screen 2 #10-#17)")
    R = {}
    for k, v in f.FAMILIES.items():
        a = v["run"](); b = v["run"](); R[k] = a
        t.check(f"{k}: deterministic (two runs are equal)", a == b)
        t.check(f"{k}: family entry has idea, methods, run, needs", v["idea"] and len(v["methods"]) >= 4 and callable(v["run"]) and v["needs"] in ("local", "hardware", "model"))
    t.check("catalog method rows: 5+6+5+5+4+4+5+5 = 39 method names registered over 8 families", sum(len(v["methods"]) for v in f.FAMILIES.values()) == 39 + 0, str(sum(len(v["methods"]) for v in f.FAMILIES.values())))

    # ------------------------------------------------ #10 expert compute server
    r = R["s2_10"]; s = r["straggler_mean_over_traces"]
    t.check("s2_10 unit: exact uniform straggler k=2 of n=4 is 4/3", abs(f.hyper_straggler(2, 4) - 4 / 3) < 1e-12)
    t.check("s2_10 unit: exact uniform straggler k=1 is 2.0 (one expert always lands on one node)", abs(f.hyper_straggler(1, 128) - 2.0) < 1e-12)
    ones = [[1, 1, 1, 1]] * 0; z = [[0] * 8 for _ in range(f.NL)]; half = [[1 if e >= 2 else 0 for e in range(8)] for _ in range(f.NL)]; one = [[1 if e == 3 else 0 for e in range(8)] for _ in range(f.NL)]
    T4 = toy([[0, 1, 2, 3]])
    t.check("s2_10 unit: straggler 1.0 for a 2/2 split of 4 experts, 1.5 for a 3/1 split, 2.0 for all on one node", f.straggler(T4, half) == 1.0 and f.straggler(T4, one) == 1.5 and f.straggler(T4, z) == 2.0)
    bs = f.balanced_side([[10, 9, 2, 1]] * f.NL, E=4)[0]
    t.check("s2_10 unit: greedy balanced split of loads 10,9,2,1 gives {10,1} vs {9,2}", bs == [0, 1, 1, 0], str(bs))
    t.check("s2_10 sanity: every straggler >= 1.0 and <= 2.0", all(1.0 <= v <= 2.0 for per in r["straggler"].values() for v in per.values()))
    t.check("s2_10 measured: random-split straggler mean over 3 traces x 20 seeds in [1.24, 1.30] (screen: 1.25-1.28)", inr(s["random_mean"], 1.24, 1.30), str(s["random_mean"]))
    t.check("s2_10 measured: exact uniform-draw straggler k=8/n=128 in [1.26, 1.27] and k=10/n=512 in [1.24, 1.25] (the 125B shape is not worse)", inr(r["straggler_uniform_k8_n128"], 1.26, 1.27) and inr(r["straggler_uniform_k10_n512"], 1.24, 1.25))
    t.check("s2_10 measured: best balanced split (held out) stays above the 1.15 kill line, in [1.22, 1.28]", inr(s["balanced_held_out"], 1.22, 1.28) and not r["straggler_balanced_below_1p15"], str(s["balanced_held_out"]))
    t.check("s2_10 measured: in-sample balanced is the lowest straggler, in [1.20, 1.25] (overfit lower bound)", s["balanced_in_sample"] <= s["balanced_held_out"] and inr(s["balanced_in_sample"], 1.20, 1.25))
    t.check("s2_10 sanity: hit rates in [0,1] and the pooled two-node hit >= single-node hit at 8/16/32 GB", all(0 <= h["single_node"] <= 1 and h["single_node"] <= h["pooled_two_nodes"] <= 1 for h in r["hits"].values()))
    t.check("s2_10 measured: 8 GB pooled hit in [0.78, 0.84] vs single in [0.54, 0.60] (screen: about 0.8 vs 0.55)", inr(r["hits"]["8 GB"]["pooled_two_nodes"], 0.78, 0.84) and inr(r["hits"]["8 GB"]["single_node"], 0.54, 0.60))
    t.check("s2_10 claim vs measured: compute gain with dense on node 0 is in [1.15, 1.25], below the screen's 2/1.26 = 1.59", inr(r["compute_gain_amdahl_two_nodes"], 1.15, 1.25) and r["compute_gain_amdahl_two_nodes"] < r["screen_claimed_compute_bound_2_over_straggler"] - 0.3, str(r["compute_gain_amdahl_two_nodes"]))
    t.check("s2_10 claim vs measured: if ALL compute were split the gain would be in [1.55, 1.60], reproducing the screen's 1.59", inr(r["compute_gain_if_all_split"], 1.55, 1.60))
    g = r["tps"]["8 GB"]
    t.check("s2_10 claim vs measured: 8 GB serial gain in [1.55, 1.80] (screen 2.1) and ideal-overlap gain in [1.0, 1.2] (screen 1.6)", inr(g["gain_serial"], 1.55, 1.80) and inr(g["gain_overlap_bound"], 1.0, 1.2), f"{g['gain_serial']} {g['gain_overlap_bound']}")
    t.check("s2_10 sanity: 16/32 GB gain in [1.05, 1.25] (compute-bound, screen 1.4-1.45)", all(inr(r["tps"][m]["gain_overlap_bound"], 1.05, 1.25) for m in ("16 GB", "32 GB")))
    t.check("s2_10 sanity: two-node overlap bound >= two-node serial; single likewise", all(v["two_overlap_bound"] >= v["two_serial"] and v["single_overlap_bound"] >= v["single_serial"] for v in r["tps"].values()))
    t.check("s2_10 control: a lazy split (all experts on one node) gives straggler exactly 2.0, so the measure can see imbalance, and 20 random seeds never reach it", f.straggler(toy([[0, 1, 2, 3, 4, 5, 6, 7]]), z) == 2.0 and s["random_max"] < 1.5)

    # ------------------------------------------------ #11 layer-split peer
    r = R["s2_11"]; mr = r["model_rows"]
    t.check("s2_11 unit: two-tier I/O hand case = 0.25/2 + 0.25/4 = 0.1875", abs(f.two_tier_io(1.0, 0.5, 0.75, 2.0, 4.0) - 0.1875) < 1e-12)
    t.check("s2_11 unit: layer-split I/O hand case: parallel 3.8 s, sum 4.3 s", abs(f.layer_split_io(4.8, 10, 1.0, 2.0, True) - 3.8) < 1e-12 and abs(f.layer_split_io(4.8, 10, 1.0, 2.0, False) - 4.3) < 1e-12)
    t.check("s2_11 sanity: model row at the local rate is exactly 1.0x of local", mr["8 GB @ 3.11 GB/s"]["overlap_vs_local"] == 1.0 and mr["16 GB @ 3.11 GB/s"]["serial_vs_local"] == 1.0)
    for mc in ("8 GB", "16 GB"):
        v = [mr[f"{mc} @ {x} GB/s"]["serial"] for x in (3.11, 2.5, 1.1, 0.11)]
        t.check(f"s2_11 sanity: {mc} model throughput never rises as the link slows", all(v[i] >= v[i + 1] for i in range(3)), str(v))
    t.check("s2_11 measured: 8 GB model at 2.5 GB/s is in [0.85, 0.95] of local serial (screen expected within 5%); 1.1 and 0.11 GB/s are slower", inr(mr["8 GB @ 2.5 GB/s"]["serial_vs_local"], 0.85, 0.95) and mr["8 GB @ 1.1 GB/s"]["serial_vs_local"] < 0.6 and mr["8 GB @ 0.11 GB/s"]["serial_vs_local"] < 0.1)
    pc = r["peer_cache"]
    t.check("s2_11 measured: 13 GB of peer cache lifts the 8 GB hit from [0.54, 0.58] to [0.95, 0.97]", inr(pc["8 GB @ 2.5 GB/s"]["h_local"], 0.54, 0.58) and inr(pc["8 GB @ 2.5 GB/s"]["h_with_peer"], 0.95, 0.97))
    t.check("s2_11 measured: peer cache over a link slower than NVMe never beats local (all serial_vs_local <= 1.0 at 2.5, 1.1, 0.11 GB/s)", all(v["serial_vs_local"] <= 1.0 + 1e-9 for k, v in pc.items() if not k.endswith("3.11 GB/s")))
    ls = r["layer_split"]
    t.check("s2_11 measured: layer-split gain over local is at most [1.0, 1.15] on every row (screen: about 1.0x)", max(v["serial_vs_local"] for v in ls.values()) <= 1.15 and max(v["overlap_vs_local"] for v in ls.values()) <= 1.0 + 1e-9, str(max(v["serial_vs_local"] for v in ls.values())))
    t.check("s2_11 measured: on a 1.1 GB/s (10GbE-class) link the 8 GB layer split is in [0.5, 0.7] of local serial (screen 0.1-0.4 for 1-10GbE with no cache credit)", inr(ls["8 GB @ 1.1 GB/s parallel"]["serial_vs_local"], 0.5, 0.7))
    t.check("s2_11 claim vs measured: 38 layers need [52, 53] GB, a 16 GB peer holds 11 layers, not 38", inr(r["peer_gb_needed_for_38_layers"], 52, 53) and r["layers_a_16GB_peer_holds"] == 11)
    t.check("s2_11 sanity: gain from an infinitely fast link is capped at 1.0x by the compute ceiling at 8 GB (overlap bound 4.58)", r["max_gain_from_faster_link"] == 1.0 and inr(r["compute_ceiling_tps"], 4.5, 4.6))
    t.check("s2_11 control: the same model with a hypothetical 0.11 GB/s link must fall below 0.25x of local (can fail if the link rate were ignored)", mr["16 GB @ 0.11 GB/s"]["overlap_vs_local"] < 0.25)

    # ------------------------------------------------ #12 zone-map block sparsity
    r = R["s2_12"]
    t.check("s2_12 unit: gate-first bytes factor 1 at d=0, 2/3 at d=0.5, 1/3 at d=1", f.gate_first_factor(0) == 1 and abs(f.gate_first_factor(0.5) - 2 / 3) < 1e-12 and abs(f.gate_first_factor(1) - 1 / 3) < 1e-12)
    t.check("s2_12 unit: predictor factor 1-d; union of k=2 tokens at d=0.5 is 0.75", f.predictor_factor(0.25) == 0.75 and f.union_live(0.5, 2) == 0.75)
    t.check("s2_12 unit: read rate with zero latency equals the raw bandwidth; it rises with read size", abs(f.read_rate(35225, lat_s=0.0) - f.cm.R_EXPL) < 1e-9 and f.read_rate(1e4) < f.read_rate(1e5) < f.read_rate(2.7e6))
    t.check("s2_12 unit: persistence of a toy trace (2 of 4 ids repeat) is 0.5", f.persistence(toy([[0, 1, 2, 3], [2, 3, 4, 5]])) == 0.5)
    t.check("s2_12 sanity: read-rate model reproduces the measured explicit-read rate near 2.7 MB, in [2.9, 3.11]", inr(r["read_rate_check_2p7MB"], 2.9, 3.11))
    t.check("s2_12 measured: token-to-token expert persistence mean in [0.40, 0.46] (screen: 42% cross-token reuse)", inr(r["measured"]["persistence_mean"], 0.40, 0.46), str(r["measured"]["persistence_mean"]))
    t.check("s2_12 measured: the 8 GB LRU already hits [0.54, 0.60] of accesses, so the 42% reuse is not extra on top of the best path", inr(r["measured"]["lru_hit_at_8GB_mean"], 0.54, 0.60))
    t.check("s2_12 claim vs measured: gate-first byte gain at the claimed d=0.5 is [1.45, 1.55]x, not 2x; compute ceiling lift in [1.15, 1.19]x", inr(r["bytes_gain_over_best_path_d0.5_gate_first"], 1.45, 1.55) and inr(r["gate_first"]["0.5"]["ceiling_lift"], 1.15, 1.19))
    t.check("s2_12 claim vs measured: 32-neuron blocks cut read speed to [0.3, 0.45] of 2.7 MB reads, so the byte gain at d=0.5 becomes an I/O-time gain in [0.5, 0.6] (a loss)", inr(r["frag"]["block32"]["rate_loss_vs_2p7MB"], 0.3, 0.45) and inr(r["frag"]["block32"]["io_time_gain_d0.5"], 0.5, 0.6))
    d_needed = 1 - (1 / 2.6 - 1 / 3) * 1.5
    t.check("s2_12 claim vs measured: the claimed 2.6x byte gain under gate-first needs d in [0.9, 0.95]", inr(d_needed, 0.9, 0.95) and f.gate_first_factor(d_needed) * 2.6 > 0.999)
    t.check("s2_12 sanity: union live share at k=8, d=0.5 under independence is in [0.99, 1.0] (sparsity vanishes under token batching)", inr(r["union_live"]["d0.5_k8"], 0.99, 1.0))
    t.check("s2_12 control: all gate-first byte gains stay below the predictor gains at the same d (a predictor is needed for 2x)", all(r["gate_first"][d]["bytes_gain"] < r["predictor"][d]["bytes_gain"] for d in r["gate_first"]))
    t.check("s2_12 honesty: dead fraction d is recorded as unmeasured (needs activations)", "dead fraction" in r["unmeasured"] and f.FAMILIES["s2_12"]["needs"] == "model")

    # ------------------------------------------------ #13 typing-time prefill
    r = R["s2_13"]; mrat = r["measured_ratio"]
    t.check("s2_13 unit: chunk_unique on a toy trace: c=1 sums 3 x 96 = 288, c=2 gives 240, c=3 gives 240 vs whole 240", f.chunk_unique(toy([[0, 1], [1, 2], [3, 4]]), 1) == (288, 240) and f.chunk_unique(toy([[0, 1], [1, 2], [3, 4]]), 2) == (240, 240) and f.chunk_unique(toy([[0, 1], [1, 2], [3, 4]]), 3) == (240, 240))
    t.check("s2_13 unit: uniform unique experts for one token is exactly k=10; many tokens saturate at 512", abs(f.uniform_unique_per_layer(1) - 10) < 1e-9 and f.uniform_unique_per_layer(5000) > 511.99)
    t.check("s2_13 unit: typing_ttft hand cases: 4 tokens, chunk 2, 1 s per chunk, 1 tok/s gives 1 s; 3 s per chunk gives 4 s", abs(f.typing_ttft(4, 2, 1.0, 0.5, 1.0) - 1.0) < 1e-12 and abs(f.typing_ttft(4, 2, 3.0, 0.5, 1.0) - 4.0) < 1e-12)
    t.check("s2_13 sanity: unique-pair ratio never rises with the chunk size, and equals 1.0 at one batch", all(mrat[a] >= mrat[b] for a, b in (("8", "16"), ("16", "32"), ("32", "64"), ("64", "128"), ("128", "500"))) and mrat["500"] == 1.0)
    t.check("s2_13 measured: chunked-prefill unique-expert reads vs one batch are in [7, 9.5]x at c=16 and [4.5, 6]x at c=32 on the traces (screen expected 5-10x)", inr(mrat["16"], 7, 9.5) and inr(mrat["32"], 4.5, 6), f"{mrat['16']} {mrat['32']}")
    t.check("s2_13 measured: the uniform-routing 125B model is an upper bound on the skewed trace ratio, within a factor 1.6 at c=16 and c=32", all(1.0 <= r["model125"][c]["io_ratio_vs_one_batch"] / mrat[c] < 1.6 for c in ("16", "32")), f"{r['model125']['16']['io_ratio_vs_one_batch']} {r['model125']['32']['io_ratio_vs_one_batch']}")
    t.check("s2_13 measured: cold 500-token prefill is I/O-bound at [20, 23] s (66 GB at 3.1 GB/s) vs compute [18, 21] s", inr(r["cold_prefill_s"]["ttft"], 20, 23) and inr(r["cold_prefill_s"]["compute"], 18, 21) and inr(r["cold_prefill_s"]["gb_read"], 64, 69))
    t.check("s2_13 measured: at 7 tok/s typing no chunk below one batch keeps up; smallest keeping up is 500", r["smallest_chunk_that_keeps_up"] == 500 and not any(r["model125"][c]["keeps_up"] for c in ("8", "16", "32", "64", "128")))
    t.check("s2_13 claim vs measured: best TTFT speedup at 7 tok/s typing is 1.0x (screen: 5-10x at best, 40x claimed); chunked prefill keeps the disk busy 1.1x to 2.8x of the typing time", r["best_ttft_speedup"] == 1.0 and all(r["model125"][c]["disk_busy_share_of_typing_time"] > 1.0 for c in ("8", "16", "32", "64", "128")))
    sn = r["typing_speed_sensitivity"]
    t.check("s2_13 claim vs measured: only a very slow typist gets a gain: best speedup in [6, 7.5]x at 1 tok/s, [4.5, 5.5]x at 2 tok/s, 1.0x at 7 tok/s", inr(sn["1.0"]["best_ttft_speedup"], 6, 7.5) and inr(sn["2.0"]["best_ttft_speedup"], 4.5, 5.5) and sn["7.0"]["best_ttft_speedup"] == 1.0)
    fast = f.fam_s2_13(link=1e6)
    t.check("s2_13 control: with free I/O (1e6 GB/s) the same code says chunk 8 keeps up, so the 'never keeps up' result comes from the I/O cost", fast["smallest_chunk_that_keeps_up"] == 8)

    # ------------------------------------------------ #14 ARC vs LRU
    r = R["s2_14"]
    t.check("s2_14 unit: scan case a a b c d a at cap 2: ARC 2 hits, LRU 1 hit", f.arc_hits(list("aabcda"), 2) == 2 and f.cx.lru_hits(list("aabcda"), 2) == 1)
    t.check("s2_14 unit: a b a c a at cap 2 gives 2 ARC hits; 3 distinct keys x 3 passes at cap 3 gives 6 hits; cap 0 gives 0", f.arc_hits(list("abaca"), 2) == 2 and f.arc_hits(list("abcabcabc"), 3) == 6 and f.arc_hits(list("abc"), 0) == 0)
    rng = random.Random(5); seq = [rng.randrange(40) for _ in range(3000)]
    t.check("s2_14 sanity: ARC never holds more than cap entries and never beats Belady (random stream, caps 5..30)", all(f.arc_run(seq, c)[1] <= c and f.arc_hits(seq, c) <= f.belady_hits_fast(seq, c) for c in (5, 10, 20, 30)))
    t.check("s2_14 sanity: Belady agrees with the independent cachesim Belady on a random stream", f.belady_hits_fast(seq, 12) == f.cx.belady_hits(seq, 12))
    allh = [v for row in r["hit"].values() for per in row.values() for v in per.values()]
    t.check("s2_14 sanity: all hit rates in [0, 1]; per cell Belady >= ARC and Belady >= LRU", all(0 <= v <= 1 for v in allh) and all(p["belady"] >= p["arc"] and p["belady"] >= p["lru"] for row in r["hit"].values() for p in row.values()))
    t.check("s2_14 sanity: LRU hit mean rises with the cache size (1, 2, 3, 4, 7.6 footprints)", all(r["hot_list"][a]["lru_mean"] < r["hot_list"][b]["lru_mean"] for a, b in (("1.0", "2.0"), ("2.0", "3.0"), ("3.0", "4.0"), ("4.0", "7.6"))))
    t.check("s2_14 measured: LRU mean hit at 2/3/4 footprints in [0.53, 0.57] / [0.67, 0.70] / [0.78, 0.81] (screen on db: 0.54/0.68/0.80)", inr(r["hot_list"]["2.0"]["lru_mean"], 0.53, 0.57) and inr(r["hot_list"]["3.0"]["lru_mean"], 0.67, 0.70) and inr(r["hot_list"]["4.0"]["lru_mean"], 0.78, 0.81))
    t.check("s2_14 measured: ARC minus LRU is in [0.003, 0.015] at 2 and 3 footprints, below the 0.03 kill line (screen expected 0.00-0.02)", all(inr(r["gap"][k]["arc_minus_lru"], 0.003, 0.015) for k in ("2.0", "3.0")) and r["kill_arc_below_0p03_at_2_3_fp"], f"{r['gap']['2.0']['arc_minus_lru']} {r['gap']['3.0']['arc_minus_lru']}")
    t.check("s2_14 claim vs measured: Belady minus LRU is in [0.20, 0.24] at 2.1 footprints, much larger than the screen's expected 0.05-0.08 (offline bound, not reachable online)", inr(r["gap"]["2.1"]["belady_minus_lru"], 0.20, 0.24), str(r["gap"]["2.1"]["belady_minus_lru"]))
    t.check("s2_14 claim vs measured: ARC serial throughput gain over LRU is in [1.002, 1.02]x at 1-7.6 footprints", all(inr(v["arc_over_lru_serial"], 1.002, 1.02) for v in r["tps_gain"].values()))
    t.check("s2_14 measured: an offline hot list (learned on the other traces) hits less than in-sequence LRU at every size (mean gap >= 0.1 at 2.1 footprints)", all(v["hot_list_mean"] < v["lru_mean"] for v in r["hot_list"].values()) and r["hot_list"]["2.1"]["lru_mean"] - r["hot_list"]["2.1"]["hot_list_mean"] >= 0.1)
    t.check("s2_14 control: Belady beats LRU by at least 0.1 on every trace at 2 footprints, so a policy gap of that size would be visible", all(r["hit"]["2.0"][n]["belady"] - r["hit"]["2.0"][n]["lru"] > 0.1 for n in f.TRACES))
    t.check("s2_14 sanity: a network tier at 1 GbE (0.117 GB/s) or 10 GbE (1.15) is slower than local NVMe (3.11 GB/s)", r["network_tier_note_gbs"]["ten_gbe"] < r["network_tier_note_gbs"]["local_nvme_explicit"])

    # ------------------------------------------------ #15 remote RAM store (hardware)
    r = R["s2_15"]
    t.check("s2_15 unit: 1 GbE uncached serial = 1/(0.2182 + 1.296/0.117) = 0.0885", abs(f.link_tps(0.117, 0.0)[0] - 0.0885) < 5e-4)
    t.check("s2_15 unit: a full cache hit is compute bound at 1/COMP = 4.583 for both serial and overlap", all(abs(x - 1 / f.COMP_S) < 1e-9 for x in f.link_tps(0.117, 1.0)))
    t.check("s2_15 unit: rate to move 1.296 GB in 0.3 s is 4.32 GB/s", abs(f.required_rate(1.296, 0.3) - 4.32) < 1e-9)
    names = list(f.LINKS)
    ser = [r["uncached"][n]["serial"] for n in names if n != "local NVMe explicit"]
    t.check("s2_15 sanity: throughput rises with link speed (1GbE < 2.5GbE < 10GbE < TB-class < RDMA), uncached and cached", ser == sorted(ser) and [r["lru8gb"][n]["serial"] for n in names[:5]] == sorted(r["lru8gb"][n]["serial"] for n in names[:5]))
    t.check("s2_15 sanity: overlap bound >= serial and <= compute ceiling for every link", all(v["serial"] <= v["overlap_bound"] <= r["compute_ceiling_tps"] + 1e-9 for d in (r["uncached"], r["lru8gb"]) for v in d.values()))
    t.check("s2_15 measured: uncached 10GbE serial in [0.70, 0.78] and overlap in [0.85, 0.92] tok/s (screen 0.74/0.88); RDMA 5 GB/s in [2.0, 2.2]/[3.7, 4.0] (screen 2.09/3.85)", inr(r["uncached"]["10GbE"]["serial"], 0.70, 0.78) and inr(r["uncached"]["10GbE"]["overlap_bound"], 0.85, 0.92) and inr(r["uncached"]["RDMA 5"]["serial"], 2.0, 2.2) and inr(r["uncached"]["RDMA 5"]["overlap_bound"], 3.7, 4.0))
    t.check("s2_15 claim vs measured: with the 8 GB LRU, 1-10GbE reach only [0.04, 0.6] of local serial (screen 0.04-0.4 of best path)", all(inr(r["vs_local_8gb"][n]["serial"], 0.04, 0.6) for n in ("1GbE", "2.5GbE", "10GbE")), str({n: r["vs_local_8gb"][n]["serial"] for n in ("1GbE", "2.5GbE", "10GbE")}))
    t.check("s2_15 claim vs measured: only RDMA beats local, by [1.15, 1.25]x serial and 1.0x at the overlap bound (compute ceiling)", inr(r["vs_local_8gb"]["RDMA 5"]["serial"], 1.15, 1.25) and r["vs_local_8gb"]["RDMA 5"]["overlap_bound"] == 1.0 and r["vs_local_8gb"]["TB-class 2.8"]["serial"] < 1.0)
    t.check("s2_15 sanity: LRU hit by machine in [0,1] and rises 8 < 16 <= 32 GB", 0 <= r["lru_by_machine"]["8 GB"] < r["lru_by_machine"]["16 GB"] <= r["lru_by_machine"]["32 GB"] <= 1)
    t.check("s2_15 measured: the kill line (1.3 GB in 0.3 s) needs 4.3 GB/s; only the RDMA row passes it, 10GbE needs [1.0, 1.2] s", r["rate_needed_for_1p3GB_in_0p3s"] > f.LINKS["TB-class 2.8"] and r["seconds_for_1p3GB"]["RDMA 5"] < 0.3 and inr(r["seconds_for_1p3GB"]["10GbE"], 1.0, 1.2))
    t.check("s2_15 control: a link as fast as local NVMe gives exactly 1.0x of local (ratio machinery can return 1)", f.link_tps(f.cm.R_EXPL, 0.5)[0] == f.link_tps(f.cm.R_EXPL, 0.5)[0] and r["vs_local_8gb"]["local NVMe explicit"] == {"serial": 1.0, "overlap_bound": 1.0})
    t.check("s2_15 honesty: link throughput is recorded as unmeasured and the family needs hardware", "link throughput" in r["unmeasured"] and f.FAMILIES["s2_15"]["needs"] == "hardware")

    # ------------------------------------------------ #16 top-k 10 -> 4 plus 2-bit
    r = R["s2_16"]
    t.check("s2_16 unit: compute ceiling at k=10 is 55/12 = 4.583; it rises as k falls", abs(f.ceiling_at_k(10) - 55 / 12) < 1e-9 and f.ceiling_at_k(4) > f.ceiling_at_k(6) > f.ceiling_at_k(10))
    tt_ = toy([[5, 6, 7], [5, 6, 7]])
    t.check("s2_16 unit: sim_topm on two identical tokens, cap large: m=2 gives 192 accesses, 96 misses; m=3 gives 288 and 144", f.sim_topm(tt_, 2, 10000) == (96, 192) and f.sim_topm(tt_, 3, 10000) == (144, 288))
    t.check("s2_16 unit: dense share bounds the lift: even k=0 lifts the ceiling by at most 12/6.8 = 1.76", inr(f.ceiling_at_k(0) / f.ceiling_at_k(10), 1.75, 1.77))
    t.check("s2_16 claim vs measured: top-4 compute ceiling lift is in [1.33, 1.37]x (screen 1.31x), far below the claimed 2.5x; ceiling 6.19 tok/s, not 11.5", inr(r["ceiling_lift_top4"], 1.33, 1.37) and inr(r["ceiling"]["4"], 6.1, 6.3) and r["ceiling_lift_top4"] < r["claimed_lift"] / 1.5)
    tr = r["trace"]
    t.check("s2_16 sanity: trace hit rates in [0,1]; throughput rises as fewer experts are used (m8 <= m4 <= m3) on every machine and bit width", all(0 <= v["hit"] <= 1 for v in tr.values()) and all(tr[f"{mc} {b}bit m8"]["serial"] <= tr[f"{mc} {b}bit m4"]["serial"] <= tr[f"{mc} {b}bit m3"]["serial"] for mc in f.FP for b in (4.3, 2.5)))
    t.check("s2_16 measured: 16 GB overlap-bound gain of 4-of-8 / 3-of-8 experts is in [1.25, 1.30] / [1.35, 1.40] (10->4 lies between; screen 1.4)", inr(tr["16 GB 4.3bit m4"]["overlap_gain"], 1.25, 1.30) and inr(tr["16 GB 4.3bit m3"]["overlap_gain"], 1.35, 1.40))
    t.check("s2_16 measured: 8 GB serial gain with 4.3-bit experts is in [1.55, 1.95] (screen: 2.4x serial incl. 2-bit); with 2.5-bit in [2.0, 2.4]", inr(tr["8 GB 4.3bit m4"]["serial_gain"], 1.55, 1.7) and inr(tr["8 GB 4.3bit m3"]["serial_gain"], 1.8, 1.95) and inr(tr["8 GB 2.5bit m4"]["serial_gain"], 2.0, 2.4))
    t.check("s2_16 measured: 2-bit adds little at 16/32 GB, within [0.0, 0.06] on the serial gain (compute-bound)", all(0 <= tr[f"{mc} 2.5bit m4"]["serial_gain"] - tr[f"{mc} 4.3bit m4"]["serial_gain"] <= 0.06 for mc in ("16 GB", "32 GB")))
    cmo = r["combo_model"]
    t.check("s2_16 measured: analytic combo model, 16 and 32 GB top-4 gain in [1.35, 1.45]x (screen 1.4x)", all(inr(cmo[f"{mc} {k}"]["top4_4.3bit"], 1.35, 1.45) for mc in ("16 GB", "32 GB") for k in ("overlap", "serial")))
    t.check("s2_16 claim vs measured: quality gate fails: prior OLMoE perplexity +45.6% (top-4) and +83% with 2-bit, gate is 5%", not r["passes_5pct_quality_gate"] and inr(r["quality_olmoe_prior"]["top4_and_2bit"], 0.83, 0.84))
    t.check("s2_16 control: k=10 against itself is exactly 1.0x on the ceiling (gain machinery can return 1)", f.ceiling_at_k(10) / f.ceiling_at_k(10) == 1.0 and r["ceiling"]["10"] == round(f.cm.G_CPU / f.cm.FLOP_TOT, 4))
    t.check("s2_16 honesty: perplexity and decode speed of the override are recorded as unmeasured; family needs a model", "perplexity" in r["unmeasured"] and f.FAMILIES["s2_16"]["needs"] == "model")

    # ------------------------------------------------ #17 shared SVD basis + residual
    r = R["s2_17"]
    t.check("s2_17 unit: 4x4, r=1, residual only: half the elements, half the FLOPs; 4x the byte factor at 16 vs 4 bits gives 2.0", (lambda c: c["elem_frac"] == 0.5 and c["flop_frac"] == 0.5 and c["byte_frac"] == 2.0)(f.lowrank_cost(4, 4, 1, E=2, k=2, factor_bits=16.0, base_bits=4.0, variant="B")))
    t.check("s2_17 unit: 4x4, r=1, shared basis + residual, 2 experts: elements 14/16 and FLOPs 56/64", (lambda c: abs(c["elem_frac"] - 14 / 16) < 1e-12 and abs(c["flop_frac"] - 56 / 64) < 1e-12)(f.lowrank_cost(4, 4, 1, E=2, k=2, factor_bits=4.0, base_bits=4.0, variant="A")))
    t.check("s2_17 unit: break-even rank of a square 4x4 at equal bits is 2", abs(f.break_even_rank(4, 4, 4.0, 4.0) - 2.0) < 1e-12)
    D = [[3, 0, 0], [0, 2, 0], [0, 0, 1]]
    t.check("s2_17 unit: SVD energy of diag(3,2,1): rank 1 = 9/14, rank 2 = 13/14, rank 3 = 1", abs(f.svd_energy(D, 1) - 9 / 14) < 1e-9 and abs(f.svd_energy(D, 2) - 13 / 14) < 1e-9 and abs(f.svd_energy(D, 3) - 1.0) < 1e-9)
    rng = random.Random(3); u1 = [rng.gauss(0, 1) for _ in range(20)]; v1 = [rng.gauss(0, 1) for _ in range(30)]; u2 = [rng.gauss(0, 1) for _ in range(20)]; v2 = [rng.gauss(0, 1) for _ in range(30)]
    M2 = [[5 * u1[i] * v1[j] + 3 * u2[i] * v2[j] + 0.01 * rng.gauss(0, 1) for j in range(30)] for i in range(20)]
    t.check("s2_17 control: a planted rank-2 matrix has residual < 0.01 at r=2, so the energy measure can see low-rank structure", 1 - f.svd_energy(M2, 2) < 0.01)
    ne = r["null_energy"]
    t.check("s2_17 measured: structure-free Gaussian matrices keep residual energy in [0.40, 0.70] at ranks of 12-33% (the 10% residual premise needs strong structure)", all(inr(v["residual"], 0.40, 0.70) and v["captured"] > v["rank_share"] for v in ne.values()), str(ne))
    g = r["cost_30b_gate"]["64"]
    t.check("s2_17 measured: rank 64, 768x2048 matrix, shared basis + residual: FLOP fraction in [0.14, 0.17], fp16-factor byte fraction in [0.50, 0.60] (about 50% compression)", inr(g["A_fp16"]["flop_frac"], 0.14, 0.17) and inr(g["A_fp16"]["byte_frac"], 0.50, 0.60))
    t.check("s2_17 sanity: elements and FLOPs rise with rank; fp8 factors halve the fp16 byte fraction", all(r["cost_30b_gate"][a]["A_fp16"]["elem_frac"] < r["cost_30b_gate"][b]["A_fp16"]["elem_frac"] for a, b in (("16", "32"), ("32", "64"), ("64", "128"), ("128", "256"))) and abs(g["A_fp16"]["byte_frac"] - 2 * g["A_fp8"]["byte_frac"]) < 2e-4)
    t.check("s2_17 measured: break-even rank is in [149, 151] with fp16 factors and in [557, 560] with 4.3-bit factors; rank 256 fp16 stores more bytes than the dense matrix", inr(r["break_even"]["rank_fp16.0"], 149, 151) and inr(r["break_even"]["rank_fp4.3"], 557, 560) and r["cost_30b_gate"]["256"]["A_fp16"]["byte_frac"] > 1.0)
    t.check("s2_17 claim vs measured: with expert FLOPs cut as in rank 64 the ceiling lift is in [1.55, 1.60]x; even free expert FLOPs cap it at [1.75, 1.77]x (claim: 40x fewer FLOPs)", inr(r["ceiling_lift_variantA"]["64"], 1.55, 1.60) and inr(r["max_ceiling_lift_if_expert_flops_were_free"], 1.75, 1.77))
    t.check("s2_17 measured: in a 250-300 token trace [0.25, 0.50] of the experts of a layer are never routed to, and the hottest half carries >= 0.95 of accesses (RFID-MoE zero-rank idea)", inr(r["usage"]["never_used_share_mean"], 0.25, 0.50) and r["usage"]["hottest_half_coverage"] >= 0.95)
    t.check("s2_17 honesty: residual energy of real expert weights is recorded as unmeasured; family needs a model", "residual energy" in r["unmeasured"] and f.FAMILIES["s2_17"]["needs"] == "model")
    return t

if __name__ == "__main__": sys.exit(0 if run().summary() else 1)
