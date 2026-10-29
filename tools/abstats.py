"""Statistics and design helpers for A/B runs. Pure Python.
 - exact two-sided Mann-Whitney p-value (dynamic programming over rank sums; exact when there are no ties, enumeration when there are ties and N <= 20)
 - Latin squares for balanced run order
 - evaluate(): the rules that decide whether a speedup may be printed at all
"""
import itertools, math, statistics

def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i]); ranks = [0.0] * len(values); i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]: j += 1
        for q in range(i, j + 1): ranks[order[q]] = (i + j) / 2 + 1
        i = j + 1
    return ranks

def mwu_exact_p(a, b):
    """Two-sided exact p for H0: same distribution. Statistic: rank sum of `a` (mid-ranks); p = P(|R - mean| >= |R_obs - mean|)."""
    n1, n2 = len(a), len(b); N = n1 + n2; pooled = list(a) + list(b); ranks = _ranks(pooled)
    r_obs = sum(ranks[:n1]); mean = n1 * (N + 1) / 2
    if len(set(pooled)) == N:                                   # no ties: integer ranks, exact DP over subset sums
        maxs = N * (N + 1) // 2; ways = [[0] * (maxs + 1) for _ in range(n1 + 1)]; ways[0][0] = 1
        for r in range(1, N + 1):
            for c in range(min(r, n1), 0, -1):
                prev = ways[c - 1]; cur = ways[c]
                for s in range(maxs, r - 1, -1):
                    if prev[s - r]: cur[s] += prev[s - r]
        total = math.comb(N, n1); dev = abs(r_obs - mean)
        return sum(ways[n1][s] for s in range(maxs + 1) if abs(s - mean) >= dev - 1e-9) / total
    if N > 20: raise ValueError("ties with more than 20 observations: exact enumeration refused")
    dev = abs(r_obs - mean); cnt = tot = 0
    for idx in itertools.combinations(range(N), n1):
        tot += 1
        if abs(sum(ranks[i] for i in idx) - mean) >= dev - 1e-9: cnt += 1
    return cnt / tot

def min_reachable_p(n1, n2): return 2 / math.comb(n1 + n2, n1)

def latin_square(k):
    """Cyclic Latin square: row i is the run order of round i; every treatment appears once in every position and once in every round."""
    return [[(i + j) % k for j in range(k)] for i in range(k)]

def evaluate(rows, cfg=None):
    """rows: dicts with keys arm ('A'|'B'), value (higher = faster), valid (bool), order (int), host, patch, hash, cold (read per token), unit.
    Returns {'print_speedup': bool, 'reasons': [...], 'ratio': float|None, 'p': float|None, 'n': {arm: n}}. Any reason blocks printing a speedup."""
    c = {"min_n": 5, "min_cold": 20.0, "alpha": 0.05, **(cfg or {})}; why = []
    A = [r for r in rows if r["arm"] == "A"]; B = [r for r in rows if r["arm"] == "B"]
    if not A or not B: return {"print_speedup": False, "reasons": ["an arm is empty"], "ratio": None, "p": None, "n": {"A": len(A), "B": len(B)}}
    if any(not r.get("valid", True) for r in rows): why.append("an invalid row is present")
    if any(r.get("cold") is not None and r["cold"] < c["min_cold"] for r in rows): why.append("a row failed the cold-state gate (read per token below threshold)")
    if min(len(A), len(B)) < c["min_n"]: why.append(f"fewer than {c['min_n']} runs in an arm")
    if len({r.get("host") for r in rows}) > 1: why.append("mixed hosts")
    if len({r.get("patch") for r in rows}) > 1: why.append("mixed patch hashes")
    if any(not r.get("hash") for r in rows): why.append("a row has no output hash")
    if len({r["hash"] for r in rows if r.get("hash")}) > 1: why.append("outputs differ between runs (hashes not identical)")
    if len({r.get("unit") for r in rows}) > 1: why.append("mixed units")
    byorder = sorted(rows, key=lambda r: r["order"])
    first = [r["arm"] for r in byorder[: max(2, len(rows) // 4)]]
    if len(rows) >= 4 and len(set(first)) == 1: why.append("run order is unbalanced (one arm always goes first)")
    va, vb = [r["value"] for r in A], [r["value"] for r in B]
    if not (max(va) < min(vb) or max(vb) < min(va)): why.append("ranges overlap")
    p = mwu_exact_p(va, vb) if len(set(va + vb)) == len(va + vb) or len(va + vb) <= 20 else None
    if p is not None and p >= c["alpha"]: why.append(f"exact p = {p:.4f} is not below {c['alpha']}")
    return {"print_speedup": not why, "reasons": why, "ratio": statistics.median(vb) / statistics.median(va), "p": p, "n": {"A": len(A), "B": len(B)}}
