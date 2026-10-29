"""T02 (synthetic trace generator) and T03 (trace analysis). The brute-force functions here are independent of tools/trace_tools.py on purpose."""
import itertools, os, random, sys, collections
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "tools")); sys.path.insert(0, HERE)
from tracegen import gen_trace; import trace_tools as tt; from harness import T

# ---- independent brute-force metrics (plain loops over sets, no shared code) ----
def bf_persistence(tokens):
    s = 0.0; n = 0
    for t in range(1, len(tokens)):
        for l in tokens[t]:
            a = set(tokens[t][l]); b = set(tokens[t - 1][l]); c = 0
            for x in a:
                if x in b: c += 1
            s += c / len(a); n += 1
    return s / n
def bf_window(tokens, W):
    hit = tot = 0
    for t in range(W, len(tokens)):
        for l in tokens[t]:
            for x in tokens[t][l]:
                tot += 1
                if any(x in tokens[j][l] for j in range(t - W, t)): hit += 1
    return hit / tot
def bf_ws(tokens):
    per = []
    for l in tokens[0]:
        u = set()
        for tk in tokens: u.update(tk[l])
        per.append(len(u))
    return sum(per) / len(per)
def bf_cov(tokens, frac, E):
    d = {}
    for tk in tokens:
        for l in tk:
            for x in tk[l]: d[(l, x)] = d.get((l, x), 0) + 1
    order = sorted(d.values(), reverse=True); n = int(frac * len(tokens[0]) * E)
    return sum(order[:n]) / sum(order)

def run():
    t = T("T02/T03 traces")
    # ---------------- T02 generator: one case per grid point ----------------
    cov = {}
    grid = list(itertools.product((12, 24, 48), (64, 128, 512), (8, 10), (0.5, 0.8, 1.0, 1.2, 1.5), (0.0, 0.3, 0.6, 0.9)))
    if os.environ.get('QUICK'): grid = [g for g in grid if g[0] == 12 and g[1] == 64]          # fast mode for mutation checks
    for L, E, k, s, rho in grid:
        seed = (L * 7919 + E * 104729 + k * 31 + int(s * 100) * 17 + int(rho * 100)) % 10**6
        lines = gen_trace(L, E, k, s, rho, 60, seed); tokens, st = tt.parse(lines)
        p = tt.persistence(tokens)
        t.check(f"gen L{L} E{E} top{k} zipf{s} rho{rho}: persistence {p:.3f} within +-0.05 of {rho}", abs(p - rho) <= 0.05, f"seed {seed}")
        ok = len(tokens) == 60 and all(len(set(ids)) == k and all(0 <= e < E for e in ids) for tk in tokens for ids in tk.values())
        t.check(f"gen L{L} E{E} top{k} zipf{s} rho{rho}: 60 whole tokens, {k} distinct in-range ids per record", ok)
        cov[(L, E, k, rho, s)] = tt.coverage(tokens, 0.10, E)
    for L, E, k, rho in itertools.product((12, 24, 48), (64, 128, 512), (8, 10), (0.0, 0.3, 0.6, 0.9)):
        if os.environ.get('QUICK') and not (L == 12 and E == 64): continue
        for a, b in zip((0.5, 0.8, 1.0, 1.2), (0.8, 1.0, 1.2, 1.5)):
            t.check(f"popularity skew: top-10% coverage rises with Zipf exponent (L{L} E{E} top{k} rho{rho}: s={a} -> s={b})", cov[(L, E, k, rho, b)] >= cov[(L, E, k, rho, a)] - 0.01, f"{cov[(L,E,k,rho,a)]:.3f} -> {cov[(L,E,k,rho,b)]:.3f}")
    for sd in range(10):
        a = gen_trace(6, 64, 8, 1.0, 0.5, 30, sd); b = gen_trace(6, 64, 8, 1.0, 0.5, 30, sd); c = gen_trace(6, 64, 8, 1.0, 0.5, 30, sd + 1000)
        t.check(f"same seed {sd} gives identical text, different seed gives different text", a == b and a != c)
    # ---------------- T03 analysis: malformed inputs, one case each ----------------
    clean = gen_trace(6, 64, 8, 1.0, 0.5, 20, 99); want, _ = tt.parse(clean)
    L = 6; per_tok = L
    bad = [("empty line", ""), ("line with only the tensor name", "blk.3.ffn_down_exps.weight"), ("non-integer expert id", "blk.2.ffn_down_exps.weight 1 2 x 4 5 6 7 8"),
        ("negative expert id", "blk.2.ffn_down_exps.weight 1 2 -3 4 5 6 7 8"), ("expert id beyond E (accepted as data, not a parse error)", "blk.2.ffn_down_exps.weight 1 2 3 4 5 6 7 9999"),
        ("tab separated ids", "blk.2.ffn_down_exps.weight\t1\t2\t3\t4\t5\t6\t7\t8"), ("CRLF line ending", "blk.2.ffn_down_exps.weight 1 2 3 4 5 6 7 8\r"),
        ("truncated last line (no newline)", "blk.2.ffn_down_exps.weight 1 2 3"), ("layer index out of order", "blk.0.ffn_down_exps.weight 1 2 3 4 5 6 7 8"),
        ("same expert twice in one line", "blk.2.ffn_down_exps.weight 1 1 2 3 4 5 6 7"), ("fewer ids than top-k", "blk.2.ffn_down_exps.weight 1 2"),
        ("more ids than top-k", "blk.2.ffn_down_exps.weight 1 2 3 4 5 6 7 8 9 10 11 12"), ("very large integer id", "blk.2.ffn_down_exps.weight 1 2 3 4 5 6 7 99999999999999999999"),
        ("unicode in tensor name", "blk.2.ffn_down_exps.weighté 1 2 3 4 5 6 7 8"), ("tensor name of the wrong family", "blk.2.ffn_up_exps.weight 1 2 3 4 5 6 7 8"),
        ("leading spaces", "   blk.2.ffn_down_exps.weight 1 2 3 4 5 6 7 8"), ("trailing spaces", "blk.2.ffn_down_exps.weight 1 2 3 4 5 6 7 8   "),
        ("comment line starting with #", "# a comment"), ("line from a prefill batch (longer than top-k)", "blk.2.ffn_down_exps.weight " + " ".join(str(i) for i in range(40))),
        ("non-ASCII digits", "blk.2.ffn_down_exps.weight ١ ٢ 3 4 5 6 7 8"), ("hex ids", "blk.2.ffn_down_exps.weight 0x1 0x2 3 4 5 6 7 8"), ("floating point ids", "blk.2.ffn_down_exps.weight 1.5 2 3 4 5 6 7 8"),
        ("embedded NUL byte", "blk.2.ffn_down_exps.weight 1 2 3\x00 4 5 6 7 8"), ("1 MB single line", "x" * 1_000_000), ("blank line made of spaces", "     "), ("layer number with sign", "blk.+2.ffn_down_exps.weight 1 2 3 4 5 6 7 8"),
        ("layer number with leading zeros", "blk.002.ffn_down_exps.weight 1 2 3 4 5 6 7 8"), ("two spaces between ids", "blk.2.ffn_down_exps.weight  1  2  3  4  5  6  7  8"), ("missing space after name", "blk.2.ffn_down_exps.weight1 2 3 4 5 6 7 8")]
    for name, line in bad:
        mixed = clean[:per_tok] + [line] + clean[per_tok:]            # bad record sits between two whole tokens
        try: got, st = tt.parse(mixed); crashed = False
        except Exception as e: crashed = True; got = None
        t.check(f"parser handles: {name} (no exception)", not crashed)
        if not crashed:
            # a record that parses as a valid stray record forms a partial block that is dropped; a malformed one is skipped: either way the whole tokens are unchanged
            t.check(f"parser handles: {name} (whole tokens unchanged)", got == want, f"{len(got)} vs {len(want)} tokens")
    t.check("empty file gives no tokens and no error", tt.parse([])[0] == [])
    t.check("file of only garbage gives no tokens", tt.parse(["a", "b", "c"])[0] == [])
    # ---------------- T03 metrics vs brute force: real traces, then random traces ----------------
    root = os.path.join(HERE, "..", "data", "traces")
    for name in ("qwen3-30b_db", "qwen3-30b_story", "qwen3-30b_code"):
        toks, st = tt.parse(open(os.path.join(root, name + ".txt")))
        E = 128
        t.check(f"{name}: {len(toks)} whole tokens of 48 layers", st["layers"] == 48 and len(toks) >= 240)
        t.check(f"{name}: persistence equals the brute-force value", abs(tt.persistence(toks) - bf_persistence(toks)) < 1e-9)
        t.check(f"{name}: window-2 reuse equals the brute-force value", abs(tt.window_reuse(toks, 2) - bf_window(toks, 2)) < 1e-9)
        t.check(f"{name}: window-8 reuse equals the brute-force value", abs(tt.window_reuse(toks, 8) - bf_window(toks, 8)) < 1e-9)
        t.check(f"{name}: working set equals the brute-force value", abs(tt.working_set(toks) - bf_ws(toks)) < 1e-9)
        t.check(f"{name}: hottest-10% coverage equals the brute-force value", abs(tt.coverage(toks, 0.10, E) - bf_cov(toks, 0.10, E)) < 1e-9)
    db, _ = tt.parse(open(os.path.join(root, "qwen3-30b_db.txt")))
    t.check("known value: db persistence is 0.425", abs(tt.persistence(db) - 0.425) < 0.001, f"{tt.persistence(db):.4f}")
    t.check("known value: db window-8 reuse is 0.784", abs(tt.window_reuse(db, 8) - 0.784) < 0.001)
    t.check("known value: db working set is 78.8 of 128", abs(tt.working_set(db) - 78.8) < 0.1)
    t.check("known value: db hottest-10% coverage is 0.513", abs(tt.coverage(db, 0.10, 128) - 0.513) < 0.001)
    for seed in range(8):
        rng = random.Random(seed); Lr, Er, kr = rng.choice((3, 4, 6)), rng.choice((16, 32, 64)), rng.choice((2, 4, 8))
        toks, _ = tt.parse(gen_trace(Lr, Er, kr, rng.choice((0.5, 1.0, 1.5)), rng.choice((0.0, 0.5, 0.9)), 40, seed))
        t.check(f"random trace seed {seed}: persistence equals the brute-force value", abs(tt.persistence(toks) - bf_persistence(toks)) < 1e-9)
        t.check(f"random trace seed {seed}: working set equals the brute-force value", abs(tt.working_set(toks) - bf_ws(toks)) < 1e-9)
        t.check(f"random trace seed {seed}: hottest-10% coverage equals the brute-force value", abs(tt.coverage(toks, 0.10, Er) - bf_cov(toks, 0.10, Er)) < 1e-9)
    return t

if __name__ == "__main__":
    sys.exit(0 if run().summary() else 1)
