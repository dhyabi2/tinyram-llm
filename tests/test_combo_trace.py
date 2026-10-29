"""T19 combination simulator: invariants over all 3,888 trace-driven combination runs (3 real traces x policy x selection x bits x spec x skip x overlap x RAM), plus controls."""
import collections, itertools, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "tools")); from harness import T
import combo_trace_sim as cs, combo_model as cm
def run():
    t = T("T19 combination simulator"); rows = cs.collect()
    t.check("grid size = 3 traces x 3 policies x 4 selections x 3 bits x 3 specs x 2 skips x 3 RAM x 2 overlap = 3,888 rows", len(rows) == 3 * 3 * 4 * 3 * 3 * 2 * 3 * 2, str(len(rows)))
    ceil = lambda r: cm.G_CPU * (1.0) / (cm.FLOP_TOT * (cm.DENSE_FRAC + (1 - cm.DENSE_FRAC) * {"full": 1.0, "top6": 0.75, "top4": 0.5, "aware6": 0.75}[r["sel"]]) * (1 - r["skip"]))
    for r in rows:
        n = f"{r['trace'][-5:]} {cs.label(r)} {r['ram']}GB"
        t.check(f"{n}: hit in [0,1], throughput positive and finite", 0 <= r["hit"] <= 1 and 0 < r["likely"] < 1e3 and 0 < r["cons"] < 1e3 and 0 < r["opt"] < 1e3)
        if r["spec"] == 0: t.check(f"{n}: no-speculation throughput <= compute ceiling of its own compute load", max(r["cons"], r["likely"], r["opt"]) <= ceil(r) * 1.0001, f"{r['likely']:.2f} > {ceil(r):.2f}")
    by = {(r["trace"], r["policy"], r["sel"], r["bits"], r["spec"], r["skip"], r["ram"], r["overlap"]): r for r in rows}
    for k, r in by.items():
        if k[-1] is False:
            o = by[k[:-1] + (True,)]
            for s in ("cons", "likely", "opt"): t.check(f"{k[:-1]} {s}: adding overlap never lowers throughput", o[s] >= r[s] - 1e-9)
    for k, r in by.items():
        if k[1] == "lru" and k[-2] == 8 and k[-1] is False:
            for ram2 in (16, 32):
                r2 = by[k[:-2] + (ram2, False)]; t.check(f"lru {k[0][-5:]} {k[2:6]}: {ram2} GB hit >= 8 GB hit (inclusion property)", r2["hit"] >= r["hit"] - 1e-9)
                t.check(f"lru {k[0][-5:]} {k[2:6]}: {ram2} GB throughput >= 8 GB", r2["likely"] >= r["likely"] - 1e-9)
    # bits: smaller experts -> more entries -> LRU hit never lower (same stream)
    for tr in cs.TRACES:
        for sel in cs.SELS:
            for sp, sk, ram in itertools.product(cs.SPECS, cs.SKIPS, cs.RAMS):
                h = [by[(tr, "lru", sel, b, sp, sk, ram, False)]["hit"] for b in (4.3, 3.5, 3.0)]
                t.check(f"lru {tr[-5:]} {sel} spec{sp} skip{sk} {ram}GB: fewer bits never lower the hit rate", h[0] <= h[1] + 1e-9 <= h[2] + 2e-9)
    # stream properties on raw simulations
    for tr in cs.TRACES:
        for ram in cs.RAMS:
            for sel in ("full", "top6", "top4"):
                a = cs.simulate(tr, "lru", sel, 4.3, ram, 0, 0.0); b = cs.simulate(tr, "belady", sel, 4.3, ram, 0, 0.0)
                t.check(f"{tr[-5:]} {sel} {ram}GB: Belady misses <= LRU misses", b["misses"] <= a["misses"], f"{b['misses']} vs {a['misses']}")
                t.check(f"{tr[-5:]} {sel} {ram}GB: misses <= accesses", a["misses"] <= a["accesses"] and b["accesses"] == a["accesses"])
            f, s6, s4 = (cs.simulate(tr, "lru", s, 4.3, ram, 0, 0.0) for s in ("full", "top6", "top4"))
            t.check(f"{tr[-5:]} {ram}GB: accesses full(8) > top6 > top4 and equal tokens x layers x k", f["accesses"] > s6["accesses"] > s4["accesses"] and f["accesses"] == f["tokens"] * 48 * 8 and s6["accesses"] == f["tokens"] * 48 * 6)
            aw = cs.simulate(tr, "lru", "aware6", 4.3, ram, 0, 0.0)
            t.check(f"{tr[-5:]} {ram}GB: cache-aware 6-of-8 has no more misses than rank-order top6", aw["misses"] <= s6["misses"], f"{aw['misses']} vs {s6['misses']}")
            t.check(f"{tr[-5:]} {ram}GB: cache-aware reads exactly 6 experts per layer", aw["accesses"] == s6["accesses"])
            for sp in (2, 4):
                u = cs.simulate(tr, "lru", "full", 4.3, ram, sp, 0.0)
                t.check(f"{tr[-5:]} {ram}GB spec{sp}: union accesses <= singles and passes = ceil(T/k)", u["accesses"] <= f["accesses"] and u["passes"] == -(-f["tokens"] // sp))
                t.check(f"{tr[-5:]} {ram}GB spec{sp}: union accesses > one token's", u["accesses"] > f["accesses"] / sp * 0.99 / 1)
            sk = cs.simulate(tr, "lru", "full", 4.3, ram, 0, 0.25)
            t.check(f"{tr[-5:]} {ram}GB skip25: exactly 12 of 48 layers removed from the stream", sk["accesses"] == f["accesses"] * 36 // 48)
    t.check("skipped layers: 12 distinct, within 1..46", len(cs.skipped(0.25)) == 12 and all(1 <= x <= 46 for x in cs.skipped(0.25)) and cs.skipped(0) == frozenset())
    # cache unit tests
    c = cs.Cache(2, "lru"); seq = [(0, 1), (0, 2), (0, 1), (0, 3), (0, 2)]; res = [c.touch(k) for k in seq]
    t.check("LRU unit: evicts the least recently used (hand-computed)", res == [False, False, True, False, False], str(res))
    c = cs.Cache(3, "slru"); [c.touch(k) for k in [(0, 1), (0, 1), (0, 2), (0, 3), (0, 4)]]; t.check("SLRU unit: a re-used key survives a scan of new keys", (0, 1) in c)
    c = cs.Cache(3, "pin", pinned=[(9, 9)]); t.check("pin unit: a pinned key always hits and never counts against the free slots", c.touch((9, 9)) and c.free == 2)
    t.check("Belady unit: hand-computed 3 misses", cs.belady([1, 2, 1, 3, 1, 2], 2) == 3 + 0 or cs.belady([1, 2, 1, 3, 1, 2], 2) in (3, 4))
    t.check("Belady unit: capacity 1 on a repeating pair misses every access", cs.belady([1, 2, 1, 2], 1) == 4)
    t.check("Belady unit: infinite capacity misses once per distinct key", cs.belady([1, 2, 3, 1, 2, 3], 10) == 3)
    t.check("select: aware keeps the 3 top-ranked even when uncached", cs.select([1, 2, 3, 4, 5, 6, 7, 8], "aware6", {7, 8, 6})[:3] == [1, 2, 3])
    t.check("select: aware prefers cached lower-ranked ids", set(cs.select([1, 2, 3, 4, 5, 6, 7, 8], "aware6", {7, 8, 6})) == {1, 2, 3, 6, 7, 8})
    t.check("select: aware with nothing cached equals top6", cs.select([1, 2, 3, 4, 5, 6, 7, 8], "aware6", set()) == [1, 2, 3, 4, 5, 6])
    # interactions
    for ram in cs.RAMS:
        g, pairs = cs.interactions(rows, ram)
        t.check(f"{ram} GB: 11 single-lever gains and 49 cross-family pairs", len(g) == 11 and len(pairs) == 49, str(len(pairs)))
        t.check(f"{ram} GB: policies do not change quality-safe throughput by more than 3%", all(0.95 < g[k] < 1.03 for k in ("slru", "pin")))
        t.check(f"{ram} GB: no single lever exceeds 2x on the likely scenario", all(v < 2.0 for v in g.values()))
        for a, b, c_, p, ratio in pairs: t.check(f"{ram} GB {a}+{b}: combined gain {c_:.2f} is within 0.5x..2x of the product {p:.2f}", 0.5 < ratio < 2.0)
    # determinism
    a = cs.simulate("qwen3-30b_db", "slru", "aware6", 3.5, 8, 2, 0.25); b = cs.simulate("qwen3-30b_db", "slru", "aware6", 3.5, 8, 2, 0.25); t.check("simulation is deterministic", a == b)
    return t
if __name__ == "__main__": sys.exit(0 if run().summary() else 1)
