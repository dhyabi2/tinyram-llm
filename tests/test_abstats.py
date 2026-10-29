"""T06: A/B statistics and design rules."""
import itertools, math, os, random, statistics, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "tools")); sys.path.insert(0, HERE)
import abstats as ab; from harness import T

def brute_p(a, b):
    """Independent exact two-sided p by enumerating every split of the pooled values (own mid-rank code)."""
    pooled = list(a) + list(b); N = len(pooled); n1 = len(a)
    def midranks(v):
        r = []
        for x in v: r.append(sum(1 for y in v if y < x) + (sum(1 for y in v if y == x) + 1) / 2)
        return r
    rk = midranks(pooled); mean = n1 * (N + 1) / 2; obs = abs(sum(rk[:n1]) - mean); c = tot = 0
    for idx in itertools.combinations(range(N), n1):
        tot += 1; c += abs(sum(rk[i] for i in idx) - mean) >= obs - 1e-9
    return c / tot

def mc_p(a, b, rng, B=20000):
    """Monte-Carlo permutation estimate of the SAME statistic the library uses (rank sum with mid-ranks)."""
    pooled = list(a) + list(b); n1 = len(a); N = len(pooled)
    rk = [sum(1 for y in pooled if y < x) + (sum(1 for y in pooled if y == x) + 1) / 2 for x in pooled]
    mean = n1 * (N + 1) / 2; obs = abs(sum(rk[:n1]) - mean); c = 0
    for _ in range(B):
        rng.shuffle(rk); c += abs(sum(rk[:n1]) - mean) >= obs - 1e-9
    return (c + 1) / (B + 1)

def table(n, eff, noise, seed, alt=True):
    rng = random.Random(seed); rows = []
    for i in range(2 * n):
        arm = "AB"[(i % 2)] if alt else ("A" if i < n else "B")
        rows.append({"arm": arm, "value": 100 * (1 + (eff if arm == "B" else 0)) * (1 + rng.gauss(0, noise)), "valid": True, "order": i, "host": "vps", "patch": "p1", "hash": "h0", "cold": 100.0, "unit": "tok/s"})
    return rows

def run():
    t = T("T06 A/B statistics"); rng = random.Random(7)
    for i in range(60):                                                   # exact DP / enumeration vs independent brute force, with and without ties
        n1, n2 = rng.randrange(2, 7), rng.randrange(2, 7)
        pool = [rng.randrange(1, 9) if i % 2 else rng.random() for _ in range(n1 + n2)]
        t.check(f"exact p equals brute-force enumeration, case {i} ({'ties' if i % 2 else 'no ties'}, {n1} vs {n2})", abs(ab.mwu_exact_p(pool[:n1], pool[n1:]) - brute_p(pool[:n1], pool[n1:])) < 1e-12)
    for n in (3, 4, 5): t.check(f"minimum reachable two-sided p for {n} vs {n} is {2/math.comb(2*n,n):.5f} and complete separation attains it", abs(ab.min_reachable_p(n, n) - 2 / math.comb(2 * n, n)) < 1e-15 and abs(ab.mwu_exact_p(list(range(n)), list(range(10, 10 + n))) - 2 / math.comb(2 * n, n)) < 1e-12)
    for k in range(3, 9):
        sq = ab.latin_square(k)
        t.check(f"Latin square {k}: every row is a permutation (each treatment once per round)", all(sorted(r) == list(range(k)) for r in sq))
        t.check(f"Latin square {k}: every column is a permutation (each treatment once per position)", all(sorted(sq[i][j] for i in range(k)) == list(range(k)) for j in range(k)))
    for n, eff, noise in itertools.product((3, 4, 5, 6, 8, 10, 12), (0.0, 0.10, 0.40), (0.02, 0.10)):
        seed = n * 1000 + int(eff * 100) * 10 + int(noise * 100); rows = table(n, eff, noise, seed)
        a = [r["value"] for r in rows if r["arm"] == "A"]; b = [r["value"] for r in rows if r["arm"] == "B"]
        p = ab.mwu_exact_p(a, b); pm = mc_p(a, b, random.Random(seed)); name = f"n={n}/arm effect {int(eff*100)}% noise {int(noise*100)}%"
        t.check(f"{name}: exact p {p:.4f} agrees with a 20000-shuffle Monte-Carlo estimate of the same statistic {pm:.4f} (+-0.015)", abs(p - pm) <= 0.015, f"seed {seed}")
        res = ab.evaluate(rows)
        separated = max(a) < min(b) or max(b) < min(a)
        want = n >= 5 and separated                       # complete separation reaches the minimum reachable p, which is below 0.05 from n=4; the n>=5 rule is the design rule
        t.check(f"{name}: speedup is printed exactly when n>=5 and the arms are separated -> {want}", res["print_speedup"] == want, f"p={res['p']} separated={separated} reasons={res['reasons']}")
        if eff == 0.40 and noise == 0.02: t.check(f"{name}: strong planted effect reaches the minimum reachable p", abs(p - ab.min_reachable_p(n, n)) < 1e-12)
    h0 = sum(ab.mwu_exact_p([100 * (1 + random.Random(s * 2).gauss(0, .05)) for _ in range(0)] or [100 + random.Random(s).gauss(0, 5) for _ in range(6)], [100 + random.Random(s + 5000).gauss(0, 5) for _ in range(6)]) < 0.05 for s in range(0)) if False else 0
    fp = 0
    for s in range(200):
        r = random.Random(s); fp += ab.mwu_exact_p([r.gauss(100, 5) for _ in range(6)], [r.gauss(100, 5) for _ in range(6)]) < 0.05
    t.check(f"null pair: same-distribution arms are 'significant' in {fp}/200 simulated sets, at most the nominal 5% (+ margin)", fp <= 0.09 * 200)
    det = 0
    for s in range(50):
        r = random.Random(900 + s); det += ab.mwu_exact_p([r.gauss(100, 5) for _ in range(6)], [r.gauss(130, 5) for _ in range(6)]) < 0.05
    t.check(f"positive control: a planted 30% effect at n=6 is detected in {det}/50 sets", det >= 48)
    pl = sum(0.95 <= statistics.median([random.Random(s * 3 + i).gauss(100, 3) for i in range(5)]) / statistics.median([random.Random(s * 3 + 7000 + i).gauss(100, 3) for i in range(5)]) <= 1.05 for s in range(40))
    t.check(f"placebo arm: median ratio stays within 1.05 in {pl}/40 simulated sets", pl >= 38)
    # refusal rules, one case each
    base = table(6, 0.40, 0.02, 1)
    t.check("baseline table (n=6, balanced, same host/patch/hash, separated, cold) may print a speedup", ab.evaluate(base)["print_speedup"], str(ab.evaluate(base)["reasons"]))
    def mut(f): rows = [dict(r) for r in base]; f(rows); return ab.evaluate(rows)
    cases = {
        "an invalid row blocks printing": (lambda R: R[0].update(valid=False), "invalid row"),
        "overlapping ranges block printing": (lambda R: R[1].update(value=90.0), "ranges overlap"),
        "n=4 per arm blocks printing": (lambda R: R.__delitem__(slice(8, 12)), "fewer than 5"),
        "a missing output hash blocks printing": (lambda R: R[2].update(hash=""), "no output hash"),
        "different output hashes block printing": (lambda R: R[3].update(hash="h1"), "outputs differ"),
        "cold-gate failure blocks printing": (lambda R: R[4].update(cold=0.1), "cold-state gate"),
        "mixed hosts block printing": (lambda R: R[5].update(host="mac"), "mixed hosts"),
        "mixed patch hashes block printing": (lambda R: R[6].update(patch="p2"), "mixed patch"),
        "mixed units block printing": (lambda R: R[7].update(unit="ms"), "mixed units"),
        "an empty arm blocks printing": (lambda R: [r.update(arm="A") for r in R], "arm is empty"),
    }
    for name, (f, frag) in cases.items():
        res = mut(f); t.check(f"refusal rule: {name}", (not res["print_speedup"]) and any(frag in x for x in res["reasons"]), str(res["reasons"]))
    unb = table(6, 0.40, 0.02, 1, alt=False); res = ab.evaluate(unb)
    t.check("refusal rule: one arm always first blocks printing", not res["print_speedup"] and any("unbalanced" in x for x in res["reasons"]), str(res["reasons"]))
    res = ab.evaluate(base); t.check("a printed result carries n, p and the ratio", res["print_speedup"] and res["n"] == {"A": 6, "B": 6} and res["p"] is not None and res["ratio"] > 1.3)
    return t

if __name__ == "__main__":
    sys.exit(0 if run().summary() else 1)
