import sys, os, random, math, itertools, statistics as st, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import combo_model as cm
cs = [c for c in cm.combos() if c["gpu"] == 0]                       # CPU-only: nothing beyond the target machine
sc = lambda i: cm.sc_at(i)
def mc_params(rng): return {k: rng.triangular(v[0] if k != "lowrank" else v[2], v[2] if k != "lowrank" else v[0], v[1]) for k, v in cm.SCEN.items()}
print(f"CPU-only combinations: {len(cs)}")
for m in cm.CACHE_GIB:
    bp = {i: cm.best_path(m, sc(i)) for i in (0, 1, 2)}; mm = cm.mmap_path(m, sc(1))
    print(f"\n=== {m}: mmap {mm:.2f} | best path {bp[1]:.2f} tok/s | compute ceiling {cm.G_CPU/cm.FLOP_TOT:.2f}")
    for cap in (0, 2, 4):
        rows = sorted(({"c": c, "n": cm.name(c), "r": cm.risk(c), "t": [cm.tok_s(c, m, sc(i)) for i in (0, 1, 2)]} for c in cs if cm.risk(c) <= cap), key=lambda r: -r["t"][1])[:4]
        print(f"  -- risk <= {cap}:")
        for r in rows: print(f"     {r['n']:<44} risk {r['r']} | cons {r['t'][0]:4.2f} likely {r['t'][1]:4.2f} opt {r['t'][2]:5.2f} | x best path {r['t'][1]/bp[1]:4.2f} (cons {r['t'][0]/bp[0]:4.2f}) | x mmap {r['t'][1]/mm:4.1f}")
# Monte Carlo: probability of >= 3x (and >= 2x) over the best path with every uncertain parameter drawn at random
print("\n== Monte Carlo (400 draws of every [E] parameter between conservative and optimistic, mode = likely): probability of beating the best path")
rng = random.Random(1)
for m in cm.CACHE_GIB:
    sel = sorted([c for c in cs if cm.risk(c) <= 2], key=lambda c: -cm.tok_s(c, m, sc(1)))[:5]
    print(f" {m} (risk <= 2):")
    for c in sel:
        ratios = []
        for _ in range(400):
            p = mc_params(rng); ratios.append(cm.tok_s(c, m, p) / cm.best_path(m, p))
        print(f"    {cm.name(c):<40} median x{st.median(ratios):4.2f} | P(>=2x) {sum(r>=2 for r in ratios)/400:4.0%} | P(>=3x) {sum(r>=3 for r in ratios)/400:4.0%} | P(<1x) {sum(r<1 for r in ratios)/400:4.0%}")
# marginal effect of each lever: mean log-ratio over all combinations, with vs without, on the 8 GB and 32 GB machines (likely scenario)
print("\n== Marginal effect of each lever (geometric mean multiplier when switched on, over all other combinations, likely scenario)")
levers = [("top6", lambda c: c["K"] == 6), ("top4", lambda c: c["K"] == 4), ("3.0bit", lambda c: c["bits"] == 3.0 and not c["lowrank"]), ("2.5bit", lambda c: c["bits"] == 2.5), ("lowrank", lambda c: c["lowrank"]), ("cache routing", lambda c: c["route"]), ("overlap", lambda c: c["overlap"]), ("spec k=4", lambda c: c["spec"] == 4), ("spec k=8", lambda c: c["spec"] == 8), ("lut kernels", lambda c: c["lut"]), ("early exit", lambda c: c["exit"] != 1.0)]
for m in cm.CACHE_GIB:
    T = {id(c): cm.tok_s(c, m, sc(1)) for c in cs}
    out = []
    for nm, f in levers:
        on = [math.log(T[id(c)]) for c in cs if f(c)]; off = [math.log(T[id(c)]) for c in cs if not f(c)]
        out.append(f"{nm} x{math.exp(st.mean(on)-st.mean(off)):.2f}")
    print(f" {m}: " + " | ".join(out))
# pairwise interaction on the 8 GB machine: does the pair beat the product of the two single effects?
print("\n== Interactions at 8 GB (likely): pair multiplier vs product of the two single multipliers, each measured from the best path")
base = dict(K=10, bits=4.3, lowrank=False, route=False, overlap=False, spec=0, gpu=0, lut=False, exit=1.0); m = "8 GB"; b0 = cm.tok_s(base, m, sc(1))
sets = {"top6": dict(K=6), "3.0bit": dict(bits=3.0), "cache routing": dict(route=True), "overlap": dict(overlap=True), "spec4": dict(spec=4), "lut": dict(bits=3.0, lut=True), "exit75": dict(exit=0.75)}
single = {k: cm.tok_s({**base, **v}, m, sc(1)) / b0 for k, v in sets.items()}
rows = []
for (a, va), (b, vb) in itertools.combinations(sets.items(), 2):
    both = {**base, **va, **vb}; pair = cm.tok_s(both, m, sc(1)) / b0
    rows.append((a, b, single[a], single[b], pair, pair / (single[a] * single[b])))
for a, b, sa, sb, p, rel in sorted(rows, key=lambda r: r[5]):
    print(f"   {a:>13} + {b:<13} singles x{sa:.2f} x{sb:.2f} -> pair x{p:.2f} (product x{sa*sb:.2f}; {'SUPER-additive' if rel>1.03 else 'SUB-additive' if rel<0.97 else 'about additive'} {rel:.2f})")
