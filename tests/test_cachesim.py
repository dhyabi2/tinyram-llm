"""T01: cache simulator correctness - properties, differential testing and exact closed-form cases."""
import random, itertools, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools")); sys.path.insert(0, os.path.dirname(__file__))
from cachesim import *; from harness import T

def rand_trace(rng, n, k): return [rng.randrange(k) for _ in range(n)]

def run():
    t = T("T01 cachesim"); rng = random.Random(20260101)
    # 1. differential: fast LRU == naive LRU on random traces
    for i in range(120):
        tr = rand_trace(rng, rng.randrange(1, 300), rng.randrange(1, 40)); cap = rng.randrange(0, 45)
        t.check(f"lru==reference seed-case {i}", lru_hits(tr, cap) == lru_hits_reference(tr, cap), f"n={len(tr)} cap={cap}")
    # 2. stack (inclusion) property: LRU hits are non-decreasing in capacity
    for i in range(60):
        tr = rand_trace(rng, 400, rng.randrange(5, 60)); prev = -1
        ok = all((h := lru_hits(tr, c)) >= prev and not (prev := h) is None for c in range(0, 65, 4))
        t.check(f"lru monotone in capacity case {i}", ok)
    # 3. Belady is at least LRU for every capacity, and equals brute-force optimum on tiny traces
    for i in range(60):
        tr = rand_trace(rng, rng.randrange(5, 200), rng.randrange(2, 30)); cap = rng.randrange(1, 20)
        t.check(f"belady>=lru case {i}", belady_hits(tr, cap) >= lru_hits(tr, cap))
    def brute(tr, cap):
        from functools import lru_cache
        @lru_cache(None)
        def go(i, cache):
            if i == len(tr): return 0
            a = tr[i]
            if a in cache: return 1 + go(i + 1, cache)
            if len(cache) < cap: return go(i + 1, tuple(sorted(cache + (a,))))
            return max(go(i + 1, tuple(sorted(tuple(x for x in cache if x != v) + (a,)))) for v in cache)
        return go(0, ())
    for i in range(40):
        tr = tuple(rng.randrange(5) for _ in range(rng.randrange(4, 13))); cap = rng.randrange(1, 4)
        t.check(f"belady == brute-force optimum case {i}", belady_hits(list(tr), cap) == brute(tr, cap), f"{tr} cap={cap}")
    # 4. bounds and exact cases
    for i in range(40):
        tr = rand_trace(rng, rng.randrange(1, 200), rng.randrange(1, 30)); u = len(set(tr))
        t.check(f"cap 0 gives 0 hits {i}", lru_hits(tr, 0) == 0)
        t.check(f"cap >= unique gives n - unique hits {i}", lru_hits(tr, u) == len(tr) - u and lru_hits(tr, u + 7) == len(tr) - u)
        t.check(f"hits never exceed n - unique {i}", all(lru_hits(tr, c) <= len(tr) - u for c in (1, 2, 5, u)))
    # 5. cyclic access: LRU gets exactly 0 hits when cap < period (the footprint cliff), n - period hits when cap >= period
    for period, rounds in itertools.product((2, 3, 5, 8, 13, 29, 64), (3, 6)):
        tr = list(range(period)) * rounds
        t.check(f"cyclic period {period} x{rounds}: cap=period-1 -> 0 hits", lru_hits(tr, period - 1) == 0)
        t.check(f"cyclic period {period} x{rounds}: cap=period -> n-period hits", lru_hits(tr, period) == len(tr) - period)
        t.check(f"cyclic period {period} x{rounds}: Belady beats LRU below the cliff", period < 3 or belady_hits(tr, period - 1) > 0)
    # 6. pinned set
    for i in range(40):
        tr = [rng.choice([0, 0, 0, 1, 1, 2, 3, 4, 5, 6]) for _ in range(300)]
        cnt = collections.Counter(tr)
        for n_pin in (0, 1, 2, 7):
            exp = sum(c for _, c in cnt.most_common(n_pin))
            t.check(f"pinned top-{n_pin} equals counted hits {i}", pinned_hits(tr, n_pin) == exp)
    t.check("pinned all keys hits everything", pinned_hits([1, 2, 3, 1], 9) == 4)
    t.check("pinned empty set learns nothing", pinned_hits([1, 2, 3], 2, learn=[]) == 0)
    # 7. per-layer LRU equals LRU of each layer on its own accesses
    for i in range(30):
        tr = [(rng.randrange(4), rng.randrange(20)) for _ in range(300)]
        want = sum(lru_hits([a for a in tr if a[0] == L], 5) for L in range(4))
        t.check(f"per-layer LRU decomposes by layer {i}", per_layer_lru_hits(tr, 5) == want)
    # 8. footprint
    t.check("footprint of identical tokens", footprint([[1, 2, 3]] * 5) == 3.0)
    t.check("footprint counts distinct only", footprint([[1, 1, 2], [3]]) == 1.5)
    return t

if __name__ == "__main__":
    ok = run().summary(); sys.exit(0 if ok else 1)
