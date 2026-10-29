"""Published-technique family tests, group B (screen 1 ideas 8..15), run on the recorded router traces plus closed-form arithmetic.
Each family: determinism, sanity bounds, hand-computed helper cases, the measured result against the screen's claim (range stated in the check name; where the
recomputation disagrees with the screen the check passes on the measured value and the disagreement is noted in the name), and a control that can fail.
The optional I/O micro-benchmark (PUBTECH_IO=1) is not part of the default run; its checks only apply when it ran."""
import collections, os, random, struct, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from harness import T
from pubtech import f_B as B

def rg(t, name, v, lo, hi):
    return t.check(f"{name} in [{lo}, {hi}]", v is not None and lo <= v <= hi, str(v))
def twice(t, fam, fn):
    a = fn(); b = fn(); t.check(f"{fam}: deterministic (two runs identical)", a == b); return a
def walk(o):
    if isinstance(o, dict):
        for v in o.values(): yield from walk(v)
    elif isinstance(o, (list, tuple)):
        for v in o: yield from walk(v)
    else: yield o
def rate_like(r, key_has):
    out = []
    def go(o, path):
        if isinstance(o, dict):
            for k, v in o.items(): go(v, path + [k])
        elif isinstance(o, (int, float)) and not isinstance(o, bool) and any(h in path[-1] for h in key_has): out.append(o)
    go(r, []); return out
def synth(name, tokens): B._CACHE[name] = tokens; return name
def const_tokens(ids_per_token):
    return [{l: tuple(ids) for l in range(B.NL)} for ids in ids_per_token]

def run():
    t = T("pubtech B (screen 1 #8-#15)")
    # ---------------------------------------------------------------- shared helpers: hand-computed cases
    t.check("lru_scored unit: [a,b,a,c,b] cap 2 has exactly 1 hit of 5 (hand-computed)", abs(B.lru_scored(list("abacb"), 2) - 0.2) < 1e-12)
    t.check("lru_scored unit: scoring from index 2 counts only the later hit (1/3)", abs(B.lru_scored(list("abacb"), 2, 2) - 1 / 3) < 1e-12)
    t.check("lru_scored unit: cap 0 never hits, huge cap hits every repeat", B.lru_scored(list("aaaa"), 0) == 0.0 and abs(B.lru_scored(list("aaaa"), 9) - 0.75) < 1e-12)
    s, o = B.thr(0.56, 2.4)
    t.check("thr unit: hit 0.56 at 2.4 GB/s gives serial 2.19, overlap 4.20 (hand-computed)", abs(s - 2.187) < 0.005 and abs(o - 4.196) < 0.005, f"{s} {o}")
    t.check("thr unit: hit 1.0 reaches the compute ceiling in both modes", B.thr(1.0, 0.34) == (B.COMPUTE, B.COMPUTE))
    t.check("thr unit: serial <= overlap <= compute for any hit and rate", all(B.thr(h / 10, r)[0] <= B.thr(h / 10, r)[1] <= B.COMPUTE + 1e-9 for h in range(0, 11) for r in (0.34, 1.56, 2.4)))
    t.check("top_pairs unit: ties broken by key, n<=0 gives empty", B.top_pairs({"b": 2, "a": 2, "c": 1}, 2) == ["a", "b"] and B.top_pairs({"a": 1}, 0) == [])
    t.check("align_up / page_amplification unit", B.align_up(5000, 4096) == 8192 and B.align_up(4096, 4096) == 4096 and B.page_amplification(0, 16384) == 1.0 and B.page_amplification(8192, 16384) == 2.0)
    t.check("tensor_bytes unit: Q4_K (256,4,3) = 1728 bytes", B.tensor_bytes((256, 4, 3), 12) == 1728 and B.tensor_bytes((64,), 8) == 68)
    tk = const_tokens([(0, 1, 2, 3, 4, 5, 6, 7), (2, 3, 4, 5, 8, 9, 10, 11)]); synth("syn1", tk)
    t.check("window_union unit: first-4 union of two tokens is {0,1,2,3} + {2,3,4,5} = 6 (hand-computed)", B.window_union(tk, 2, 4, 1) == 6.0 and B.prefix_union(tk, 2, 4) == 6.0 and B.prefix_union(tk, 1, 8) == 8.0)
    t.check("marginal_coverage unit: identical tokens cover 8/128 for any n; disjoint pair covers 16*0.75/128 at n=2",
            abs(B.marginal_coverage(const_tokens([range(8)] * 2), 5) - 8 / 128) < 1e-12 and abs(B.marginal_coverage(const_tokens([range(8), range(8, 16)]), 2) - 16 * 0.75 / 128) < 1e-12)
    synth("syn_swap", const_tokens([range(8)]))
    pins = frozenset((l, e) for l in range(B.NL) for e in (0, 1, 100))
    t.check("swap_hit unit: 2 natural pinned of 8; m=2 optimistic -> 4/8; capped by the one spare pin per layer -> 3/8; target 4 capped -> 3/8",
            B.swap_hit("syn_swap", pins, m=2, capped=False)[0] == 0.5 and B.swap_hit("syn_swap", pins, m=2)[0] == 0.375 and B.swap_hit("syn_swap", pins, target=4)[0] == 0.375)
    synth("syn_lru", const_tokens([range(8)] * 2))
    t.check("lru_bias unit: a repeated token hits on its second pass (8 of 16) with or without bias", B.lru_bias("syn_lru", 10000, 0)[0] == 0.5 and B.lru_bias("syn_lru", 10000, 2)[0] == 0.5)
    synth("syn_bias", const_tokens([range(8), range(8), (0, 1, 2, 3, 4, 5, 6, 20)]))
    h, sw = B.lru_bias("syn_bias", 10000, 1)
    t.check("lru_bias unit: the one uncached id of token 3 is swapped for a cached id, so token 3 is all hits (hit 16/24, one swap in 24 selections)", abs(h - 16 / 24) < 1e-12 and abs(sw - 1 / 24) < 1e-12, f"{h} {sw}")
    g = B.hash_geometry(const_tokens([range(8)] * 3), 4)
    t.check("hash_geometry unit: ids in [0, 4E), distinct per token-layer, s=1 is the identity", all(0 <= e < 512 and len(set(tk[l])) == 8 for tk in g for l in tk for e in tk[l]) and B.hash_geometry(tk, 1) == tk)
    t.check("hash_geometry unit: deterministic and not constant across tokens", B.hash_geometry(tk, 4) == B.hash_geometry(tk, 4) and len({tuple(x[0]) for x in g}) > 1)
    t.check("ttft unit: 69 GB at 0.34 GB/s + 500x12 GFLOP at 311 GFLOP/s = 222.2 s serial; overlapped at 2.4 GB/s = max(28.75, 19.29) = 28.75 s",
            abs(B.ttft(1.0, 0.34, False) - 222.23) < 0.05 and abs(B.ttft(1.0, 2.4, True) - 28.75) < 1e-9)
    # GGUF parser on a hand-built header
    def s_(x): b = x.encode(); return struct.pack("<Q", len(b)) + b
    kv = s_("general.alignment") + struct.pack("<II", 4, 64) + s_("x.arr") + struct.pack("<I", 9) + struct.pack("<IQ", 8, 2) + s_("ab") + s_("c")
    t1 = s_("blk.0.ffn_gate_exps.weight") + struct.pack("<I", 3) + struct.pack("<3Q", 256, 4, 3) + struct.pack("<IQ", 12, 0)
    t2 = s_("blk.0.ffn_up_exps.weight") + struct.pack("<I", 3) + struct.pack("<3Q", 256, 4, 3) + struct.pack("<IQ", 12, 1728)
    hdr = b"GGUF" + struct.pack("<IQQ", 3, 2, 2) + kv + t1 + t2
    infos, ds, al = B.gguf_tensor_infos(hdr)
    t.check("gguf parser unit: tensors, data start and alignment from a hand-built header", infos == [("blk.0.ffn_gate_exps.weight", (256, 4, 3), 12, 0), ("blk.0.ffn_up_exps.weight", (256, 4, 3), 12, 1728)] and ds == B.align_up(len(hdr), 64) and al == 64)
    lay, ne = B.expert_layout(infos, ds, 1)
    t.check("expert_layout unit: 3 experts of 576 bytes at the right absolute offsets", ne == 3 and lay == {0: {"gate": (ds, 576), "up": (ds + 1728, 576)}})
    try: B.gguf_tensor_infos(b"NOPE" + b"\0" * 40); bad = False
    except ValueError: bad = True
    t.check("gguf parser control: a non-GGUF buffer is rejected", bad)

    # ---------------------------------------------------------------- data sanity shared by all families
    for nm in B.NAMES:
        T_ = B.load(nm)
        t.check(f"{nm}: whole tokens, 48 layers, 8 distinct ids in 0..127 per token-layer", len(T_) >= 249 and all(len(tk) == 48 and all(len(set(tk[l])) == 8 and all(0 <= e < 128 for e in tk[l]) for l in tk) for tk in T_))
    R = {k: twice(t, k, B.FAMILIES[k]["run"]) for k in B.FAMILIES}
    for k, v in B.FAMILIES.items():
        t.check(f"{k}: catalog entry has idea, method rows, needs in the allowed set and a callable", bool(v["idea"]) and len(v["methods"]) >= 3 and v["needs"] in ("local", "hardware", "model") and callable(v["run"]))
        t.check(f"{k}: result names its unmeasured parts and has no NaN/negative rate or hit", bool(R[k]["unmeasured"]) and all(not (isinstance(x, float) and x != x) for x in walk(R[k])) and all(x >= 0 for x in rate_like(R[k], ("hit", "serial", "overlap", "gb_s"))))
    shuf = {}
    for nm in B.NAMES:
        tk = list(B.load(nm)); random.Random(1).shuffle(tk); shuf[nm] = tk

    # ---------------------------------------------------------------- s1 #8 pinned set + routing bias
    r = R["s1_8"]; c = r["cross_prompt_pin_hit"]; fr = sorted(c, key=float)
    t.check("s1_8: pin hit rises with the pinned share and stays in [0,1]", all(c[a]["hit"] < c[b]["hit"] for a, b in zip(fr, fr[1:])) and all(0 <= c[f]["hit"] <= 1 for f in fr))
    rg(t, "s1_8: cross-prompt pin hit at 24.5% of pairs (screen 0.528)", c["0.245"]["hit"], 0.51, 0.55)
    rg(t, "s1_8: cross-prompt pin hit at 7% (screen 0.159)", c["0.07"]["hit"], 0.14, 0.18)
    rg(t, "s1_8: cross-prompt pin hit at 40% (screen 0.725)", c["0.4"]["hit"], 0.70, 0.75)
    b = r["pin_plus_bias"]
    rg(t, "s1_8: optimistic 2-swap bias at 24.5% pin (screen 0.76)", b["0.245"]["opt_m2"]["hit"], 0.74, 0.78)
    rg(t, "s1_8: optimistic 4-swap bias at 7% pin (screen 0.66)", b["0.07"]["opt_m4"]["hit"], 0.63, 0.68)
    t.check("s1_8: capping swaps by the pins available in a layer never raises the hit above the optimistic model", all(b[f][f"cap_{m}"]["hit"] <= b[f][f"opt_{m}"]["hit"] + 1e-12 for f in b for m in ("m2", "m4")))
    t.check("s1_8: more allowed swaps give a higher hit and a larger share of selections changed (quality-risk proxy)", all(b[f]["cap_m4"]["hit"] >= b[f]["cap_m2"]["hit"] and b[f]["cap_m4"]["swapped"] >= b[f]["cap_m2"]["swapped"] for f in b))
    rg(t, "s1_8: forcing half the experts to be pinned changes this share of selections at a 24.5% pin", b["0.245"]["cap_half"]["swapped"], 0.05, 0.25)
    rg(t, "s1_8: pin chosen from the first 16 tokens, 24.5% share, scored on the rest", r["first16_pin_hit_24p5"], 0.65, 0.72)
    rg(t, "s1_8: LRU hit at the 8 GB cache (2.1 footprints; screen 0.56)", r["lru_8gb"]["hit"], 0.54, 0.58)
    rg(t, "s1_8: LRU plus a 2-swap bias at 8 GB (screen 0.78)", r["lru_bias_m2_8gb"]["hit"], 0.76, 0.82)
    t.check("s1_8: LRU bias hit is monotone in the swap budget m=0,1,2,4", r["lru_8gb"]["hit"] < r["lru_bias_m1_8gb"]["hit"] < r["lru_bias_m2_8gb"]["hit"] < r["lru_bias_m4_8gb"]["hit"])
    g = r["gain_vs_lru_8gb"]
    rg(t, "s1_8: serial gain of the LRU bias over plain LRU at 8 GB (screen 1.35x)", g["serial_bias_m2"], 1.30, 1.45)
    rg(t, "s1_8: overlap gain of the LRU bias at 8 GB (screen 1.09x)", g["overlap_bias_m2"], 1.05, 1.12)
    rg(t, "s1_8: a static pin at the same 8 GB capacity vs LRU, serial (loses; screen: little worth)", g["serial_static_pin"], 0.70, 0.82)
    rg(t, "s1_8: bias plus LRU vs plain mmap at 8 GB, serial (read size is most of it)", r["gain_vs_mmap_8gb"]["serial_bias_m2"], 5.3, 6.1)
    t.check("s1_8: control - 'a static pin beats LRU at equal 8 GB' is false on these traces (check would fail if pin >= LRU)", r["static_pin_8gb"]["hit"] < r["lru_8gb"]["hit"])
    t.check("s1_8: control - an empty pin set scores 0 and a full pin set scores 1", B.pin_hit(B.NAMES[0], frozenset()) == 0.0 and B.pin_hit(B.NAMES[0], frozenset((l, e) for l in range(48) for e in range(128))) == 1.0)
    t.check("s1_8: control - shuffling token order lowers the LRU hit (temporal locality is what the cache and bias use)", all(B.lru_scored(B.flat(shuf[n]), 795) < B.lru_scored(B.flat(B.load(n)), 795) for n in B.NAMES))

    # ---------------------------------------------------------------- s1 #9 expert-major prefill
    r = R["s1_9"]; u = r["unique_experts_first_N_tokens_of_128"]
    rg(t, "s1_9: unique experts per layer, db, first 16 tokens (screen 43)", u["0b_db"]["16"], 42.7, 43.7)
    rg(t, "s1_9: unique experts per layer, db, first 64 tokens (screen 60)", u["0b_db"]["64"], 60.0, 61.0)
    rg(t, "s1_9: unique experts per layer, db, all 299 tokens (screen 79)", u["0b_db"]["299"], 78.3, 79.3)
    t.check("s1_9: unique experts grow with N and stay <= 128", all(list(v.values()) == sorted(v.values()) and max(v.values()) <= 128 for v in u.values()))
    cv = r["coverage_all_tokens"]
    rg(t, "s1_9: coverage after all tokens, db (screen 0.62)", cv["0b_db"], 0.60, 0.63); rg(t, "s1_9: coverage, story (screen 0.55)", cv["story"], 0.54, 0.57); rg(t, "s1_9: coverage, code (screen 0.70)", cv["_code"], 0.68, 0.71)
    rg(t, "s1_9: independent-draw model extrapolated to 500 tokens (mean of 3 traces)", r["coverage_proxy_500"]["model_500"], 0.59, 0.63)
    t.check("s1_9: the independent-draw model at 64 tokens is within 0.08 of the measured window mean on every trace", all(abs(m["model_at_64"] - m["empirical_at_64"]) < 0.08 for m in r["marginal_model"].values()))
    t.check("s1_9: uniform-routing coverage at 500 tokens is ~1 (the screen's upper end) while the trace proxy is far lower", r["coverage_proxy_500"]["uniform_125b_top10_of_512"] > 0.9999 and r["coverage_proxy_500"]["model_500"] < 0.75)
    f = r["ttft_500_tokens_125b"]["full_1.0"]
    rg(t, "s1_9: TTFT of plain mmap at 0.34 GB/s, serial, full coverage (screen 222 s)", f["mmap_0.34"]["serial_s"], 221.0, 223.5)
    rg(t, "s1_9: TTFT of explicit 4 MiB reads, overlapped (screen 29 s)", f["explicit_2.4"]["overlap_s"], 28.5, 29.0)
    rg(t, "s1_9: TTFT of explicit reads, serial (screen 48 s)", f["explicit_2.4"]["serial_s"], 47.5, 48.5)
    rg(t, "s1_9: gain over mmap 0.34, serial (screen 4.6x)", f["gain_vs_mmap_0.34"]["serial"], 4.5, 4.7); rg(t, "s1_9: gain over mmap 0.34, overlapped (screen 7.6x)", f["gain_vs_mmap_0.34"]["overlap"], 7.6, 7.8)
    rg(t, "s1_9: gain over mmap at 1 GB/s, serial (measured 1.84x; screen said 2.4-3x, disagrees on the serial end)", f["gain_vs_mmap_1.0"]["serial"], 1.8, 1.9)
    rg(t, "s1_9: gain over mmap at 1 GB/s, overlapped (screen 3x)", f["gain_vs_mmap_1.0"]["overlap"], 3.0, 3.15)
    rg(t, "s1_9: prefill compute floor at 311 GFLOP/s (screen 19 s, not 109 s)", r["compute_floor_s"], 19.2, 19.4); rg(t, "s1_9: the 55 GFLOP/s decode rate would give", r["compute_floor_if_55gflops_s"], 109.0, 109.2)
    t.check("s1_9: per-token loading straw man is 650 GB / 271 s", r["per_token_loading_strawman"]["gb"] == 650.0 and abs(r["per_token_loading_strawman"]["seconds_at_2p4"] - 270.83) < 0.01)
    rg(t, "s1_9: prompt length above which prefill is compute-bound (screen ~750)", r["breakeven_prompt_tokens_compute_bound"], 740.0, 750.0)
    rg(t, "s1_9: with measured partial coverage the overlapped gain over mmap 0.34 is lower than the full-coverage 7.7x", r["ttft_500_tokens_125b"]["proxy_measured"]["gain_vs_mmap_0.34"]["overlap"], 7.4, 7.7)
    t.check("s1_9: control - coverage is invariant to relabelling experts, and identical tokens cover only 8/128",
            abs(B.prefix_union(B.load(B.NAMES[0]), 64, 8) - B.prefix_union([{l: tuple((e * 37 + 5) % 128 for e in tk[l]) for l in tk} for tk in B.load(B.NAMES[0])], 64, 8)) < 1e-9 and B.prefix_union(const_tokens([range(8)] * 50), 50, 8) == 8.0)
    io = B._IO.get("res")
    if io:
        t.check("s1_9: I/O run - all measured rates positive and below 20 GB/s", all(0 < x["gb_s"] < 20 for x in io["micro"]))
        t.check("s1_9: I/O run - 8-thread explicit 4 MiB reads beat the single-thread mmap copy of the same extents (timing-robust comparison)", B.io_rate(io, "pread nocache", 4.0, 8) > B.io_rate(io, "mmap copy", 4.0, 1))
    else: t.check("s1_9: I/O micro-benchmark not requested (PUBTECH_IO unset), model/GGUF-dependent parts reported as unmeasured", r["measured_io_rates_gb_s"] is None)

    # ---------------------------------------------------------------- s1 #10 pin after prefill
    r = R["s1_10"]; w = r["W64"]; w16 = r["W16"]
    rg(t, "s1_10: W=64 pin hit at 10% share (screen: 0.37-0.50)", w["0.1"]["mean_pin"], 0.41, 0.45); rg(t, "s1_10: W=64 LRU hit at 10% share (screen 0.47-0.52)", w["0.1"]["mean_lru"], 0.47, 0.51)
    rg(t, "s1_10: W=64 pin hit at 36% (screen 0.81-0.92)", w["0.36"]["mean_pin"], 0.85, 0.89); rg(t, "s1_10: W=64 LRU hit at 36% (screen 0.89-0.94)", w["0.36"]["mean_lru"], 0.92, 0.95)
    rg(t, "s1_10: W=64 hybrid hit at 36% (screen 0.89-0.96)", w["0.36"]["mean_hybrid"], 0.92, 0.95)
    t.check("s1_10: pin is below LRU at every share >= 10% for both learning windows (the screen's finding)", all(x[s]["mean_pin"] < x[s]["mean_lru"] for x in (w, w16) for s in ("0.1", "0.2", "0.36")))
    t.check("s1_10: below one footprint (6% share) LRU is exactly 0 (cliff) while a pin still hits", w["0.06"]["mean_lru"] == 0.0 and w["0.06"]["mean_pin"] > 0.2)
    rg(t, "s1_10: hybrid minus LRU at 36% (kill rule needs +0.05; screen: at most +0.01-0.04)", w["0.36"]["mean_hybrid"] - w["0.36"]["mean_lru"], -0.02, 0.01)
    t.check("s1_10: learning window 64 predicts better than 16 at 36% share, and in-sample oracle pin >= learned pin everywhere", w["0.36"]["mean_pin"] > w16["0.36"]["mean_pin"] and all(x[s]["mean_oracle"] >= x[s]["mean_pin"] for x in (w, w16) for s in x))
    rg(t, "s1_10: the claimed '40% reuse' is what a 10% pin delivers (W=64)", r["claim_check_40pct"]["hit_of_10pct_pin_W64"], 0.40, 0.46)
    rg(t, "s1_10: a 36% pin learned on OTHER prompts only", r["cross_prompt_pin_36"], 0.66, 0.70)
    rg(t, "s1_10: pin-by-share (hit 0.26) vs LRU at 8 GB, serial (screen 0.6-0.75x)", r["gain_pin_vs_lru_8gb"]["serial"], 0.70, 0.78); rg(t, "s1_10: same, overlapped", r["gain_pin_vs_lru_8gb"]["overlap"], 0.57, 0.62)
    rg(t, "s1_10: claimed hit 0.40 gives serial tok/s (screen 1.84)", r["throughput_explicit_4mib"]["claimed_h0.40"]["serial"], 1.83, 1.85)
    t.check("s1_10: pin and LRU rise with share, all hits in [0,1]", all(w[a]["mean_pin"] < w[b]["mean_pin"] and w[a]["mean_lru"] <= w[b]["mean_lru"] for a, b in (("0.06", "0.1"), ("0.1", "0.2"), ("0.2", "0.36"))))
    t.check("s1_10: control - hybrid_hit with all keys pinned is 1.0 and with nothing pinned equals plain LRU", B.hybrid_hit(list("abab"), 0, set("ab"), 1) == 1.0 and abs(B.hybrid_hit(list("abacb"), 0, set(), 2) - 0.2) < 1e-12)
    t.check("s1_10: control - a 10% pin learned on OTHER prompts is worse than one learned on this prompt's first 64 tokens (prompt-specific learning is what the idea relies on)",
            all(B.pin_hit(n, B.cross_pins(n, 614)) + 0.05 < w["0.1"]["per_trace"][n[-5:]]["pin"] for n in B.NAMES))

    # ---------------------------------------------------------------- s1 #11 prefill-only top-4
    r = R["s1_11"]; fw = r["first_window_union"]
    rg(t, "s1_11: first 16 tokens, db, k=8 unique experts (screen 43.2)", fw["0b_db"]["16"]["k8"], 43.0, 43.4); rg(t, "s1_11: first 16 tokens, db, k=4 (screen 28.2)", fw["0b_db"]["16"]["k4"], 28.0, 28.4)
    rg(t, "s1_11: first 64 tokens, db, k=8 (screen 60.5)", fw["0b_db"]["64"]["k8"], 60.3, 60.7); rg(t, "s1_11: first 64 tokens, db, k=4 (screen 42.0)", fw["0b_db"]["64"]["k4"], 41.8, 42.2)
    rg(t, "s1_11: first 64 tokens, code, k=8 (screen 69.4)", fw["_code"]["64"]["k8"], 69.2, 69.6)
    wu = r["window_union_unique_experts_per_layer"]
    t.check("s1_11: k=8/k=4 union ratio falls as the window grows (the saving vanishes toward a full prompt)", all([x["ratio"] for x in v.values()] == sorted([x["ratio"] for x in v.values()], reverse=True) for v in wu.values()))
    rg(t, "s1_11: union ratio k8/k4 over the whole trace, db (1.0 = no saving)", wu["0b_db"]["all(299)"]["ratio"], 1.35, 1.40)
    t.check("s1_11: k=4 never touches more experts than k=8 in any window", all(x["k4"] <= x["k8"] for v in wu.values() for x in v.values()))
    io_r = r["io_ratio_k4_over_k10"]
    rg(t, "s1_11: prefill bytes ratio k4/k10, 500-token model on traces", io_r["model_500_on_traces"], 0.73, 0.78); rg(t, "s1_11: prefill bytes ratio, uniform 125B (all experts touched)", io_r["uniform_125b"], 0.975, 0.985)
    tt = r["ttft_500_tokens_125b"]
    rg(t, "s1_11: TTFT gain, uniform-routing 125B, overlapped (screen 1.1-1.2x; measured 1.02x, below the screen)", tt["uniform_125b"]["gain_overlap"], 1.01, 1.04)
    rg(t, "s1_11: TTFT gain, uniform-routing 125B, serial (screen 1.1-1.2x)", tt["uniform_125b"]["gain_serial"], 1.11, 1.15)
    rg(t, "s1_11: TTFT gain if the 30B skew carried over, overlapped (upper bound, 1.33x; screen 1.1-1.2x)", tt["trace_model_500"]["gain_overlap"], 1.30, 1.36)
    rg(t, "s1_11: prefill compute at k=4 (14.3 s; FLOP factor 0.74)", tt["trace_model_500"]["comp4_s"], 14.2, 14.4)
    dm = r["decode_misses_per_token_fixed_cache_2p1fp"]
    rg(t, "s1_11: decode misses/token, db, k=8 at 2.1 footprints (screen 170.1)", dm["8"]["0b_db"], 169.5, 170.7); rg(t, "s1_11: decode misses/token, db, k=4 (screen 70.9)", dm["4"]["0b_db"], 70.3, 71.5)
    rg(t, "s1_11: decode misses/token, code, k=3 (screen 53.1)", dm["3"]["_code"], 52.8, 53.5); rg(t, "s1_11: decode miss ratio k4/k8 (2.2-2.4x fewer bytes)", r["decode_miss_ratio_k4"], 0.42, 0.43)
    rg(t, "s1_11: decode compute ceiling at k=4 (screen ~6 tok/s)", r["decode_compute_ceiling_k4"], 6.1, 6.3)
    h = r["lru_hit_equal_footprints"]
    t.check("s1_11: at equal footprints the k=4 hit curve lies above the k=8 curve and both rise with cache size", all(h["4"][s] > h["8"][s] for s in h["8"]) and h["8"]["1.0"] < h["8"]["2.1"] < h["8"]["4.0"])
    t.check("s1_11: misses fall monotonically with k (3 < 4 < 8) on every trace", all(dm["3"][n] < dm["4"][n] < dm["8"][n] for n in dm["8"]))
    t.check("s1_11: control - using k=8 or k=100 as the prefix gives the same union, and a repeated token has union 4 at k=4", B.window_union(B.load(B.NAMES[1]), 16, 8) == B.window_union(B.load(B.NAMES[1]), 16, 100) and B.window_union(const_tokens([range(8)] * 5), 3, 4) == 4.0)

    # ---------------------------------------------------------------- s1 #12 GPU tier
    r = R["s1_12"]; sm = r["lru_hit_same_prompt"]
    rg(t, "s1_12: LRU hit at 29% of experts, db (screen 0.855)", sm["0.29"]["0b_db"], 0.85, 0.86); rg(t, "s1_12: story (screen 0.884)", sm["0.29"]["story"], 0.88, 0.89); rg(t, "s1_12: code (screen 0.809)", sm["0.29"]["_code"], 0.80, 0.82)
    t.check("s1_12: below one footprint (6% of experts) the LRU hit is 0 (the cliff); the hit rises monotonically with the cache fraction", all(v == 0.0 for v in sm["0.06"].values()) and all(sm[a][n] < sm[b][n] for a, b in (("0.06", "0.12"), ("0.12", "0.17"), ("0.17", "0.29"), ("0.29", "0.5")) for n in sm["0.29"]))
    rg(t, "s1_12: warm LRU primed on another prompt, db at 29% (screen ~0.81-0.89)", r["lru_hit_warm_from_other_prompt"]["0.29"]["0b_db"], 0.85, 0.87)
    t.check("s1_12: a pin learned on other prompts (29%) is well below LRU (screen: 0.44-0.70 vs 0.81-0.88)", all(r["pin_hit_cross_prompt_0.29"][n] < sm["0.29"][n] for n in sm["0.29"]))
    sp = r["lru_hit_by_sparsity_fraction_of_experts"]
    t.check("s1_12: sparser geometry never raises the hit at a given cache fraction (s=1 >= 2 >= 4)", all(sp["1"][f] >= sp["2"][f] >= sp["4"][f] for f in sp["1"]))
    rg(t, "s1_12: hit at 29% of experts with 4x sparser experts (screen kill threshold 0.5, keep threshold 0.75; measured passes)", sp["4"]["0.29"], 0.77, 0.81)
    t.check("s1_12: control - the s=1 hash geometry reproduces the trace exactly", r["control_hash_s1_equals_original"] is True)
    ue = r["unit_error_check"]
    rg(t, "s1_12: the claimed 6.2 tok/s used 0.9 MB for a 2.7 MB expert", ue["claimed_toks_at_1p56"], 6.2, 6.3); rg(t, "s1_12: corrected I/O-only rate (screen 2.1)", ue["corrected_toks_at_1p56_io_only"], 2.07, 2.09)
    v = r["vram_tiers"]
    rg(t, "s1_12: 24 GB card, dense 30B hit gives an I/O bound of (screen ~12 tok/s)", v["vram_24gb"]["dense30b"]["io_bound_toks"], 11.9, 12.6); rg(t, "s1_12: 24 GB card, 4x sparser geometry I/O bound", v["vram_24gb"]["sparse4x"]["io_bound_toks"], 8.5, 9.0)
    rg(t, "s1_12: 6 GB card (cache 5 GB), dense hit (screen: ~0-0.2, cliff)", v["vram_6gb"]["dense30b"]["hit"], 0.40, 0.46)
    t.check("s1_12: bigger tiers give higher hit and a higher I/O bound, sparse never above dense", v["vram_6gb"]["dense30b"]["hit"] < v["vram_12gb"]["dense30b"]["hit"] < v["vram_24gb"]["dense30b"]["hit"] and all(v[k]["sparse4x"]["hit"] <= v[k]["dense30b"]["hit"] for k in v))
    t.check("s1_12: CPU-only best path stays at the 4.57 ceiling at 16/32 GB; the GPU ceiling is tagged as an unmeasured assumption", r["cpu_only_best_path_tok_s"]["32"]["overlap"] == 4.57 and "NOT measured" in r["gpu_ceiling_assumed_toks"]["tag"])
    t.check("s1_12: control - shuffling token order lowers the 29% LRU hit", all(B.lru_scored(B.flat(shuf[n]), int(0.29 * 6144)) < sm["0.29"][n[-5:]] for n in B.NAMES))

    # ---------------------------------------------------------------- s1 #13 top-4 for TTFT and decode
    r = R["s1_13"]
    rg(t, "s1_13: decode miss ratio k6/k8", r["decode_miss_ratio"]["6"], 0.69, 0.71); rg(t, "s1_13: decode miss ratio k4/k8 (screen 2.2-2.4x fewer)", r["decode_miss_ratio"]["4"], 0.42, 0.43); rg(t, "s1_13: decode miss ratio k3/k8", r["decode_miss_ratio"]["3"], 0.28, 0.30)
    rg(t, "s1_13: db misses/token at k=4 (screen 70.9)", r["decode_misses_per_token_fixed_cache_2p1fp"]["4"]["0b_db"], 70.3, 71.5)
    t.check("s1_13: control - k=8 against itself is exactly 1.0 and misses fall monotonically 8 > 6 > 4 > 3", r["decode_miss_ratio"]["8"] == 1.0 and r["decode_miss_ratio"]["6"] > r["decode_miss_ratio"]["4"] > r["decode_miss_ratio"]["3"])
    p = r["prefill_touch_probability_125b"]
    rg(t, "s1_13: P(expert touched in 500 tokens), top-4 of 512 (screen 0.98)", p["top4"], 0.979, 0.981); rg(t, "s1_13: top-10 of 512", p["top10"], 0.9999, 1.0)
    fl = r["flop_split"]
    rg(t, "s1_13: compute gain of top-4 with the dense part fixed (screen 1.3x, claim 1.8-2.2x)", fl["compute_gain_k4"], 1.34, 1.36)
    pf = r["prefill_ttft_s"]
    rg(t, "s1_13: prefill TTFT gain, overlapped (screen 1.0-1.1x)", pf["gain_overlap"], 1.01, 1.04); rg(t, "s1_13: prefill TTFT gain, serial", pf["gain_serial"], 1.11, 1.15)
    rg(t, "s1_13: I/O floor of prefill at either k (s)", pf["io_floor_k4"], 28.0, 28.5)
    t.check("s1_13: prefill is I/O-bound at both settings (I/O floor above compute floor), so the compute cut does not show up", pf["io_floor_k10"] > pf["compute_floor_k10"] and pf["io_floor_k4"] > pf["compute_floor_k4"])
    d = r["decode_top4_tok_s"]
    rg(t, "s1_13: 8 GB decode serial gain of top-4 (measured 1.74x; screen said 1.3x, disagrees)", d["8"]["gain_serial"], 1.70, 1.78); rg(t, "s1_13: 8 GB overlap gain (screen 1.4x)", d["8"]["gain_overlap"], 1.44, 1.50)
    rg(t, "s1_13: 16 GB serial gain (screen 1.2-1.3x)", d["16"]["gain_serial"], 1.33, 1.42); rg(t, "s1_13: 32 GB overlap gain = ceiling 6.19/4.57 (screen 1.3x)", d["32"]["gain_overlap"], 1.34, 1.37)
    t.check("s1_13: top-4 decode is never slower than top-8 and the miss ratio rises with cache size (bigger caches save less)", all(d[k]["gain_serial"] >= 1 and d[k]["gain_overlap"] >= 1 for k in d) and d["8"]["miss_ratio_k4"] < d["16"]["miss_ratio_k4"] < d["32"]["miss_ratio_k4"])
    t.check("s1_13: quality is cited from the earlier OLMoE run, not re-measured here, and is listed as unmeasured", r["quality_cited_olmoe_relative_ppl_loss"]["top4"] == 0.456 and any("perplexity" in x for x in r["unmeasured"]))

    # ---------------------------------------------------------------- s1 #14 repacked extents
    r = R["s1_14"]; br = r["by_ram"]
    rg(t, "s1_14: 8 GB serial gain, 4 MiB reads vs mmap (screen 4x)", br["8"]["gain_4mib_vs_mmap"]["serial"], 4.0, 4.3); rg(t, "s1_14: 8 GB overlap gain (screen up to 7x)", br["8"]["gain_4mib_vs_mmap"]["overlap"], 6.9, 7.2)
    rg(t, "s1_14: 16 GB serial gain (screen ~1.6x)", br["16"]["gain_4mib_vs_mmap"]["serial"], 1.6, 1.7); rg(t, "s1_14: 32 GB serial gain (screen 1.1-1.2x; measured higher)", br["32"]["gain_4mib_vs_mmap"]["serial"], 1.4, 1.6)
    rg(t, "s1_14: 8 GB, one 2.7 MB extent vs three 0.9 MB reads, serial", br["8"]["gain_repacked_2.7mb_vs_stock_split_0.9mb"], 1.3, 1.45)
    t.check("s1_14: repacked is never slower than the split layout and never exceeds the ceiling; gain over mmap falls with RAM", all(br[k]["read_2.7mb"]["serial"] >= br[k]["read_0.9mb"]["serial"] for k in br) and br["8"]["gain_4mib_vs_mmap"]["serial"] > br["16"]["gain_4mib_vs_mmap"]["serial"] > 1.0)
    rg(t, "s1_14: 8 GB LRU hit at 2.1 footprints (screen 0.56)", br["8"]["hit"], 0.54, 0.58)
    t.check("s1_14: reads per token 1152 stock vs 384 repacked; 4 KiB padding of a 2.7 MB expert is 0.12%", r["reads_per_token_30b"] == {"stock": 1152, "repacked": 384} and abs(r["padding_4k_overhead_2p7mb_expert"] - 0.001244) < 1e-5)
    t.check("s1_14: 69 GB load is 203 s through mmap vs 28.75 s explicit (7.06x)", abs(r["prefill_load_s_69gb"]["mmap"] - 202.94) < 0.01 and abs(r["prefill_load_s_69gb"]["gain"] - 7.06) < 0.01)
    lo = r["stock_layout_of_local_gguf"]
    if lo:
        t.check("s1_14: local Qwen3-30B file has 48 layers x 128 experts; gate and up slices are 884,736 bytes, down 1,290,240", lo["layers"] == 48 and lo["experts"] == 128 and lo["slice_bytes_gate"] == lo["slice_bytes_up"] == 884736 and lo["slice_bytes_down"] == 1290240, str(lo))
        rg(t, "s1_14: mean expert size in the stock file (MB; the screen assumed 2.7)", lo["expert_bytes_mean"] / 1e6, 2.8, 2.9)
        t.check("s1_14: in the stock file no expert slice starts on a 4 KiB boundary (tensor data is only 32-byte aligned), so direct I/O needs the repack or head/tail trimming", lo["share_slices_4k_aligned"] < 0.01 and lo["alignment"] == 32)
        rg(t, "s1_14: 16 KiB page amplification of the unaligned stock slices (read-size overhead)", lo["mean_16k_page_amplification"], 1.0, 1.05)
        t.check("s1_14: an expert's gate and up extents are ~114 MB apart in the stock layout, so the three reads cannot be merged", lo["gate_to_up_distance_bytes"] > 100e6 and lo["reads_per_expert_stock"] == 3)
    else: t.check("s1_14: local GGUF absent, layout analysis reported as unmeasured", lo is None)
    io = r["io_microbench"]
    if io:
        t.check("s1_14: I/O run - rates positive and the 8-thread pread of 2.7 MB beats the single-thread mmap copy", all(0 < x["gb_s"] < 20 for x in io["micro"]) and B.io_rate(io, "pread nocache", 2.7, 8) > B.io_rate(io, "mmap copy", 2.7, 1))
    else: t.check("s1_14: I/O micro-benchmark not requested, rates reported as unmeasured", io is None)
    t.check("s1_14: control - a 4 KiB-aligned 2.7 MB extent adds no more than one page of padding while a stock slice start shifts by <4 KiB", 0 <= B.align_up(2700000, 4096) - 2700000 < 4096 and B.page_amplification(100, 884736) > 1.0)

    # ---------------------------------------------------------------- s1 #15 no-cache slab pread
    r = R["s1_15"]; pr = r["prev_token_reuse"]
    rg(t, "s1_15: previous-token reuse, db (screen 0.425)", pr["0b_db"], 0.42, 0.43); rg(t, "s1_15: story (screen 0.455)", pr["story"], 0.45, 0.46); rg(t, "s1_15: code (screen 0.417)", pr["_code"], 0.41, 0.42)
    lh = r["lru_hit_at_footprints"]
    rg(t, "s1_15: LRU hit at 1 footprint, db (screen 0.301)", lh["1"]["0b_db"], 0.30, 0.31); rg(t, "s1_15: at 2 footprints, story (screen 0.585)", lh["2"]["story"], 0.58, 0.59); rg(t, "s1_15: at 3 footprints, code (screen 0.640)", lh["3"]["_code"], 0.63, 0.65)
    t.check("s1_15: LRU hit rises with footprints on every trace", all(lh["1"][n] < lh["2"][n] < lh["3"][n] for n in lh["1"]))
    tp = r["throughput_by_read_size"]
    rg(t, "s1_15: no-cache serial at 4 MiB reads (screen 1.31)", tp["4mib_2.4"]["none"]["serial"], 1.30, 1.33); rg(t, "s1_15: no-cache overlap at 4 MiB (screen 1.85)", tp["4mib_2.4"]["none"]["overlap"], 1.84, 1.86)
    rg(t, "s1_15: no-cache serial at 1 MiB (screen 0.95)", tp["1mib_1.56"]["none"]["serial"], 0.94, 0.96); rg(t, "s1_15: no-cache overlap at 0.9 MB reads (screen 1.0)", tp["0.9mb_1.3"]["none"]["overlap"], 0.99, 1.01)
    rg(t, "s1_15: no cache vs plain mmap at 8 GB, serial (screen 2.5x)", r["none_vs_mmap_8gb"]["serial"], 2.45, 2.55); rg(t, "s1_15: same, overlap (screen 3.1x)", r["none_vs_mmap_8gb"]["overlap"], 3.05, 3.15)
    rg(t, "s1_15: no cache vs our best path (explicit reads + LRU) at 8 GB, serial (screen 0.60x)", r["none_vs_best_path_8gb"]["serial"], 0.59, 0.61); rg(t, "s1_15: same, overlap (screen 0.44x)", r["none_vs_best_path_8gb"]["overlap"], 0.43, 0.45)
    ks = r["kill_rule_policy_beats_none_by_over_1.7x_io"]
    t.check("s1_15: the screen's kill rule fires: previous-token reuse and LRU-2 each cut I/O time by more than 1.7x (1.76x, 2.22x), so the cache must be kept", ks == {"none": False, "prev": True, "lru2": True} and r["io_time_speedup_vs_none"]["prev"] > 1.7)
    t.check("s1_15: policy ordering by hit none < prev < lru2, and previous-token reuse is mean persistence", 0 == r["throughput_by_read_size"]["4mib_2.4"]["none"]["hit"] < tp["4mib_2.4"]["prev"]["hit"] < tp["4mib_2.4"]["lru2"]["hit"] and abs(sum(pr.values()) / 3 - tp["4mib_2.4"]["prev"]["hit"]) < 1e-12)
    t.check("s1_15: control - persistence of identical consecutive tokens is 1.0 and of disjoint tokens 0.0", B.tt.persistence(const_tokens([range(8)] * 3)) == 1.0 and B.tt.persistence(const_tokens([range(8), range(8, 16)])) == 0.0)
    t.check("s1_15: control - shuffling token order lowers previous-token reuse below the real traces", all(B.tt.persistence(shuf[n]) < pr[n[-5:]] for n in B.NAMES))
    rp = r["io_replay"]
    if rp:
        d = {(x["policy"], x["variant"]): x for x in rp["replay"]}
        t.check("s1_15: I/O run - replays read the expected bytes (coalesced = split3, prev and lru2 fewer), no thread errors", abs(d[("none", "coalesced")]["gb"] - d[("none", "split3")]["gb"]) < 1e-6 and d[("prev", "coalesced")]["gb"] < d[("none", "coalesced")]["gb"] and d[("lru2", "coalesced")]["gb"] < d[("none", "coalesced")]["gb"] and all(x["errors"] == 0 for x in rp["replay"]))
    else: t.check("s1_15: I/O replay not requested, measured read path reported as unmeasured", rp is None)
    return t

if __name__ == "__main__": sys.exit(0 if run().summary() else 1)
