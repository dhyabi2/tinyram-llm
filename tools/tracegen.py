"""Seeded synthetic router traces in the GGML_MOE_TRACE text format: one line per layer per token, "blk.<L>.ffn_down_exps.weight id id ...".
Model: per layer a fixed popularity order (Zipf weights over a random permutation of the experts). Each token keeps each of the previous token's experts with
probability `rho`, then fills the rest by sampling WITHOUT replacement from the experts that were not in the previous set. Because new experts are drawn
outside the previous set, the expected token-to-token persistence equals `rho` exactly (it is not inflated by chance overlap).
"""
import random

def zipf_weights(E, s): return [1.0 / (r + 1) ** s for r in range(E)]

def _sample_without_replacement(rng, items, weights, m):
    """Weighted sampling without replacement in one pass (exponential race: smallest -log(u)/w wins). Equivalent to drawing one at a time and removing."""
    import heapq, math
    keys = [(-math.log(1.0 - rng.random()) / w, e) for e, w in zip(items, weights)]
    return [e for _, e in heapq.nsmallest(m, keys)]

def gen_trace(L, E, k, s, rho, T, seed):
    """Returns the trace as a list of text lines (T tokens x L layers)."""
    assert E >= 2 * k, "need E >= 2k so new experts can be drawn outside the previous set"
    rng = random.Random(seed); w = zipf_weights(E, s)
    perm = [rng.sample(range(E), E) for _ in range(L)]              # per-layer popularity order
    weight_of = [{e: w[r] for r, e in enumerate(p)} for p in perm]
    prev = [None] * L; lines = []
    for _ in range(T):
        for l in range(L):
            if prev[l] is None:
                cur = _sample_without_replacement(rng, range(E), [weight_of[l][e] for e in range(E)], k)
            else:
                keep = [e for e in prev[l] if rng.random() < rho]
                pool = [e for e in range(E) if e not in set(prev[l])]
                fill = _sample_without_replacement(rng, pool, [weight_of[l][e] for e in pool], k - len(keep))
                cur = keep + fill
            prev[l] = cur
            lines.append(f"blk.{l}.ffn_down_exps.weight " + " ".join(str(e) for e in cur))
    return lines
