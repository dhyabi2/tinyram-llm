"""Cache simulators over expert-access traces. A trace is a list of hashable keys, usually (layer, expert), in access order.
Pure Python, no dependencies. Used by the tests and by the model scripts.
"""
import collections

def lru_hits(trace, cap):
    """Hits of a fully associative LRU cache of `cap` entries (cold misses included)."""
    if cap <= 0: return 0
    c = collections.OrderedDict(); h = 0
    for a in trace:
        if a in c: c.move_to_end(a); h += 1
        else:
            c[a] = None
            if len(c) > cap: c.popitem(last=False)
    return h

def lru_hits_reference(trace, cap):
    """Independent, deliberately naive LRU (list based) for differential testing."""
    if cap <= 0: return 0
    stack = []; h = 0                       # stack[0] is the most recently used
    for a in trace:
        if a in stack: stack.remove(a); h += 1
        stack.insert(0, a)
        del stack[cap:]
    return h

def belady_hits(trace, cap):
    """Belady's MIN (evict the entry whose next use is farthest): the optimum for any online or offline policy."""
    if cap <= 0: return 0
    n = len(trace); nxt = [0] * n; last = {}
    for i in range(n - 1, -1, -1):
        nxt[i] = last.get(trace[i], n + 1); last[trace[i]] = i
    cache = {}; h = 0                        # key -> index of its next use
    for i, a in enumerate(trace):
        if a in cache: h += 1
        elif len(cache) >= cap:
            del cache[max(cache, key=cache.get)]
        cache[a] = nxt[i]
    return h

def pinned_hits(trace, n_pin, learn=None):
    """Static pinned set of the `n_pin` most frequent keys in `learn` (default: the trace itself = in-sample). Hits = accesses inside the set."""
    cnt = collections.Counter(learn if learn is not None else trace)
    hot = {k for k, _ in cnt.most_common(max(0, n_pin))}
    return sum(1 for a in trace if a in hot)

def per_layer_lru_hits(trace, cap_per_layer):
    """Independent LRU per layer; keys must be (layer, expert)."""
    by = collections.defaultdict(list)
    for a in trace: by[a[0]].append(a)
    return sum(lru_hits(v, cap_per_layer) for v in by.values())

def hit_rate(hits, n): return hits / n if n else 0.0

def footprint(trace_by_token):
    """Mean number of distinct (layer, expert) pairs one token touches. `trace_by_token` is a list of per-token access lists."""
    return sum(len(set(t)) for t in trace_by_token) / len(trace_by_token)
