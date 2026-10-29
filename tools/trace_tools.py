"""Parser and metrics for router traces (GGML_MOE_TRACE format). The parser never raises on bad input: it skips malformed records and counts them."""
import re, collections
_REC = re.compile(r"^blk\.(\d+)\.ffn_down_exps\.weight((?:[ \t]+\d+)+)[ \t]*$")

def parse(lines):
    """Returns (tokens, stats). tokens: list of dicts layer -> tuple(ids), only complete tokens (every layer 0..NL-1 present, once).
    A token starts at a record whose layer is <= the previous record's layer. A leading partial token and a trailing partial token are dropped."""
    recs = []; stats = collections.Counter()
    for raw in lines:
        line = raw.rstrip("\r\n")
        if line.strip() == "" or line.lstrip().startswith("#"): stats["ignored"] += 1; continue
        m = _REC.match(line.strip())
        if not m: stats["malformed"] += 1; continue
        ids = tuple(int(x) for x in m.group(2).split())
        if len(set(ids)) != len(ids): stats["duplicate_ids"] += 1; continue
        recs.append((int(m.group(1)), ids))
    if not recs: return [], stats
    k = collections.Counter(len(i) for _, i in recs).most_common(1)[0][0]
    kept = []
    for l, ids in recs:
        if len(ids) != k: stats["wrong_k"] += 1; continue
        kept.append((l, ids))
    blocks = []; cur = {}; last = None
    for l, ids in kept:
        if last is not None and l <= last: blocks.append(cur); cur = {}
        if l in cur: stats["duplicate_layer"] += 1
        cur[l] = ids; last = l
    blocks.append(cur)
    NL = 1 + max(l for l, _ in kept)
    tokens = [b for b in blocks if sorted(b) == list(range(NL))]
    stats["tokens"] = len(tokens); stats["dropped_partial_blocks"] = len(blocks) - len(tokens); stats["k"] = k; stats["layers"] = NL
    return tokens, stats

def persistence(tokens):
    """Mean over layers and consecutive tokens of |S_t & S_(t-1)| / k."""
    if len(tokens) < 2: return 0.0
    k = len(next(iter(tokens[0].values()))); tot = n = 0
    for t in range(1, len(tokens)):
        for l in tokens[t]: tot += len(set(tokens[t][l]) & set(tokens[t - 1][l])) / k; n += 1
    return tot / n

def window_reuse(tokens, W):
    """Share of this token's experts that appeared in the previous W tokens of the same layer."""
    hit = tot = 0
    for t in range(W, len(tokens)):
        for l in tokens[t]:
            seen = set()
            for j in range(t - W, t): seen |= set(tokens[j][l])
            hit += len(set(tokens[t][l]) & seen); tot += len(tokens[t][l])
    return hit / tot if tot else 0.0

def working_set(tokens):
    """Mean over layers of the number of distinct experts used in the whole trace."""
    layers = sorted(tokens[0]); return sum(len({e for tk in tokens for e in tk[l]}) for l in layers) / len(layers)

def coverage(tokens, frac, E=None):
    """Share of all accesses covered by the hottest `frac` of the (layer, expert) pairs; E = experts per layer (default: largest id + 1)."""
    cnt = collections.Counter((l, e) for tk in tokens for l in tk for e in tk[l])
    E = E or 1 + max(e for _, e in cnt); NL = len(tokens[0]); n = int(frac * NL * E)
    return sum(c for _, c in cnt.most_common(n)) / sum(cnt.values())

def accesses(tokens):
    """The flat (layer, expert) access sequence in order, for the cache simulators."""
    return [(l, e) for tk in tokens for l in sorted(tk) for e in tk[l]]
