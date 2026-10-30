"""v41_model: calibration against the measured Qwen3-30B Mac runs, monotonic properties, a control that must fail. No model is run."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "tools")); from harness import T
import v41_model as m
def run():
    t = T("T21 v41_model projection")
    t.check("calibration: worst error against the 7 measured Qwen3 configurations is under 15%", m_validate_quiet() < 0.15)
    a, _ = m.tok_s("qwen3", hit_override=0.59); b, _ = m.tok_s("qwen3", hit_override=0.80)
    t.check("higher hit rate is never slower", b > a)
    r1, _ = m.tok_s("v41", ram=16, rate=1.4); r2, _ = m.tok_s("v41", ram=16, rate=3.0)
    t.check("faster reads are never slower", r2 >= r1)
    t.check("fewer experts per token is never slower", m.tok_s("v41", K=4, ram=16)[0] >= m.tok_s("v41", K=6, ram=16)[0])
    t.check("more RAM is never slower", m.tok_s("v41", ram=32)[0] >= m.tok_s("v41", ram=16)[0])
    ceiling = 1.0 / (16e9 * 2 / 1e9 / m.G_CPU)
    t.check("no setting at 6 experts exceeds the compute ceiling %.2f tok/s" % ceiling, all(m.tok_s("v41", ram=64, rate=50.0, hit_override=1.0, route=b)[0] <= ceiling + 1e-9 for b in (0, 1, 3)))
    t.check("plain mmap is slower than the pool at the same RAM", m.tok_s("v41", plain=True, ram=16)[0] < m.tok_s("v41", ram=16)[0])
    t.check("bias 3.0 removes more misses than bias 1.0", m.tok_s("v41", ram=16, route=3)[1] > m.tok_s("v41", ram=16, route=1)[1] > m.tok_s("v41", ram=16)[1])
    t.check("cache size is never negative", m.cache_gb(m.GEOM["v41"], 4) == 0.0)
    t.check("risk score orders 4 of 6 experts above 5 of 6 above 6", m.risk(dict(K=4, bits=4.3, route=0, spec=0)) > m.risk(dict(K=5, bits=4.3, route=0, spec=0)) > m.risk(dict(K=6, bits=4.3, route=0, spec=0)))
    saved = m.R_EFF; m.R_EFF = 20.0
    t.check("control: a wrong read rate (20 GB/s) breaks the calibration (must exceed 15% error)", m_validate_quiet() > 0.15); m.R_EFF = saved
    t.check("after restoring the rate the calibration passes again", m_validate_quiet() < 0.15)
    desktop_checks(t)
    return t
def desktop_checks(t):
    D = m.desk_run
    t.check("desktop: results are deterministic", D("mid", 7.0, route=1) == D("mid", 7.0, route=1))
    t.check("desktop: low <= mid <= high scenario for the same levers", all(D("low", 7.0, **kw)["tok_s"] <= D("mid", 7.0, **kw)["tok_s"] <= D("high", 7.0, **kw)["tok_s"] for _, kw in m.LADDER))
    t.check("desktop: GPU for dense, then the VRAM tier, then bias 1.0 each never slow it down", D("mid", 7.0, gpu=False)["tok_s"] <= D("mid", 7.0, tier=False)["tok_s"] <= D("mid", 7.0)["tok_s"] <= D("mid", 7.0, route=1)["tok_s"])
    t.check("desktop: faster SSD, more RAM, more overlap, PCIe stream, fewer experts are never slower", D("mid", 12.0, route=1)["tok_s"] >= D("mid", 7.0, route=1)["tok_s"] and all(D("mid", 7.0, route=1, **kw)["tok_s"] >= D("mid", 7.0, route=1)["tok_s"] for kw in (dict(ram=128), dict(o=0.75), dict(stream=20.0), dict(K=5))))
    ok = True
    for name in ("low", "mid", "high"):
        for _, kw in m.LADDER:
            r = D(name, 7.0, **kw); ok &= 0.0 <= r["hv"] <= r["h"] <= 1.0 and r["tok_s"] > 0
    t.check("desktop: hit rates in [0,1] with VRAM hits <= total hits in every ladder row and scenario", ok)
    ceiling = 1.0 / (m.HW["desk"]["t_dense"] + m.T_ENGRAM)
    t.check("desktop: no setting exceeds the GPU-dense + Engram ceiling %.0f tok/s" % ceiling, D("high", 12.0, route=3, K=4, bits=2.0, o=1.0, stream=50.0, ram=512)["tok_s"] <= ceiling)
    t.check("desktop: PCIe stream cannot use more than 90%% of the paper DRAM bandwidth", D("mid", 7.0, stream=1e6)["tc"] >= D("mid", 7.0, stream=m.HW["desk"]["peak"] * 0.9 - m.HW["desk"]["bw"])["tc"] - 1e-12)
    h = 1 - m.MMAP_MISS * (1 - m.cm.lru(1.37)); m.cm.LRU = m.CURVES["lru"]; h = 1 - m.MMAP_MISS * (1 - m.cm.lru(1.37))
    t.check("calibration: plain-mmap miss factor reproduces the measured Mac plain hit 0.29 within 0.02 (got %.3f)" % h, abs(h - 0.29) < 0.02)
    t.check("control: a miss factor of 1.0 (plain mmap as good as ideal LRU) misses that point by more than 0.1", abs(1 - 1.0 * (1 - m.cm.lru(1.37)) - 0.29) > 0.1)
    low_b, mid_b = D("low", 7.0, route=1), D("low", 7.0, route=1, sc="mid")
    t.check("audit fix: the bias spread of the low scenario is applied (low bias 1.0 hit %.3f is below the mid-spread hit %.3f)" % (low_b["h"], mid_b["h"]), low_b["h"] < mid_b["h"] - 0.01 and D("high", 7.0, route=1)["h"] > D("high", 7.0, route=1, sc="mid")["h"] + 0.005)
    before = m.cm.lru(3.0); D("low", 7.0, route=1); D("high", 7.0, K=5); m.hw_tok_s(plain=True); m.tok_s("v41", curve="real", ram=16)
    t.check("audit fix: calling the projections leaves combo_model's own hit curve unchanged (lru(3.0) = %.3f before and after)" % before, m.cm.lru(3.0) == before)
    t.check("audit fix: the 30B curve keeps its own name, so COMBO_HIT=real cannot turn the low scenario into the public curve", m.CURVES["lru"] == m.cm.LRU_30B and m.CURVES["lru"] != m.CURVES["real"])
    r0 = D("mid", 7.0, ram=0, route=1); t.check("audit fix: 0 GB of RAM is a valid input (cache clamped at 0, VRAM tier still hits: h %.3f >= hv %.3f > 0)" % (r0["h"], r0["hv"]), r0["h"] >= r0["hv"] > 0 and r0["tok_s"] > 0)
    t.check("audit fix: fewer-experts hit credit is monotone (kc 0 <= 0.5 <= 1 at 5 of 6)", D("mid", 7.0, K=5, kc=0.0)["tok_s"] <= D("mid", 7.0, K=5, kc=0.5)["tok_s"] <= D("mid", 7.0, K=5, kc=1.0)["tok_s"] and D("mid", 7.0, K=6, kc=0.0)["tok_s"] == D("mid", 7.0, K=6, kc=1.0)["tok_s"])
    t.check("kc=0 means a fewer-experts cut earns no extra hits: 5 of 6 has the same hit as 6 of 6 (%.4f vs %.4f); kc=1 earns more" % (D("mid", 7.0, K=5, kc=0.0)["h"], D("mid", 7.0)["h"]), abs(D("mid", 7.0, K=5, kc=0.0)["h"] - D("mid", 7.0)["h"]) < 1e-9 and D("mid", 7.0, K=5, kc=1.0)["h"] > D("mid", 7.0)["h"] + 0.005)
    t.check("audit fix: plain mmap DRAM term counts the DMA write of faulted pages (more misses, more DRAM time)", m.hw_tok_s(plain=True, curve="real", r_mmap=1e9)["tc"] > (m.HW["desk"]["t_dense"] + m.GEOM["v41"]["foot0"] / m.HW["desk"]["bw"] + m.T_ENGRAM))
    t.check("desktop: stock hybrid (dense on GPU, experts paged) beats the same stack without the GPU", m.desk_base("mid", gpu=True) > m.desk_base("mid", gpu=False))
    t.check("tier_hit: hand-computed points (0.03 mid = 0.189, 0.015 low = 0.030, 0 = 0, above the grid = last value)", abs(m.tier_hit(0.03, "mid") - 0.189) < 1e-9 and abs(m.tier_hit(0.015, "low") - 0.030) < 1e-9 and m.tier_hit(0.0, "high") == 0.0 and m.tier_hit(0.9, "high") == 0.892)
    t.check("tier_hit: increasing in size and low <= mid <= high everywhere", all(m.tier_hit(x, "mid") <= m.tier_hit(x + 0.01, "mid") for x in (0.0, 0.02, 0.1, 0.25)) and all(m.tier_hit(x, "low") <= m.tier_hit(x, "mid") <= m.tier_hit(x, "high") for x in (0.01, 0.03, 0.1, 0.3)))
    meas = tier_from_traces()
    t.check("tier curve: recomputed from data/traces (leave-one-out) matches the table at 3%% and 10%% within 0.002 (got %s)" % ", ".join("%s %.3f" % (k, v) for k, v in sorted(meas.items())), all(abs(meas[k] - v) < 0.002 for k, v in {"cross3": 0.064, "adapt3": 0.189, "oracle3": 0.246, "cross10": 0.232, "adapt10": 0.407, "oracle10": 0.518}.items()))
    G = m.GEOM["v41"]; file_gb = G["foot0"] * 2.6 / 4.3 * 64 + (G["act"] - G["routed"]) * m.DENSE_BPW / 8 / 1e9 + 86.5
    t.check("geometry: routed + dense bytes at 2.6 bpw plus the 3.5-bit Engram table (86.5 GB) are within 5%% of the published 264.5 GB Q2_K file (got %.0f GB)" % file_gb, abs(file_gb / 264.5 - 1) < 0.05)
    rows = m.desk_sweep(7.0, 14)
    t.check("desk_sweep: ordered by quality risk, and no row with risk 0 reaches 14x in the mid scenario", all(rows[i][0] <= rows[i + 1][0] for i in range(len(rows) - 1)) and all(r[2]["mid"] < 14 for r in rows if r[0] == 0))
    t.check("control: a DRAM bandwidth of 5 GB/s (10x too low) cuts the mid-scenario speed by more than half", D("mid", 7.0, route=1, bw=5.0)["tok_s"] < 0.5 * D("mid", 7.0, route=1)["tok_s"])
def tier_from_traces():
    import collections, trace_tools as tt
    T = {n: tt.parse(open(os.path.join(HERE, "..", "data", "traces", "qwen3-30b_%s.txt" % n)))[0] for n in ("db", "story", "code")}
    cnt = lambda toks: collections.Counter((l, e) for tk in toks for l in tk for e in tk[l]); out = {}
    for x, tag in ((0.03, "3"), (0.10, "10")):
        n = int(x * 48 * 128); cr = []; ad = []; orc = []
        for test in T:
            ct = cnt(T[test]); tot = sum(ct.values()); orc.append(sum(c for _, c in ct.most_common(n)) / tot)
            top = {k for k, _ in cnt([y for q in T if q != test for y in T[q]]).most_common(n)}; cr.append(sum(c for k, c in ct.items() if k in top) / tot)
            head = {k for k, _ in cnt(T[test][:32]).most_common(n)}; rest = cnt(T[test][32:]); ad.append(sum(c for k, c in rest.items() if k in head) / sum(rest.values()))
        out["cross" + tag] = sum(cr) / 3; out["adapt" + tag] = sum(ad) / 3; out["oracle" + tag] = sum(orc) / 3
    return out
def m_validate_quiet():
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()): return m.validate()
if __name__ == "__main__": sys.exit(0 if run().summary() else 1)
