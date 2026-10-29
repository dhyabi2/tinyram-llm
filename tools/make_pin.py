#!/usr/bin/env python3
"""Build a GGML_MOE_PIN list ("layer expert" lines) from a router trace, using the shared loader trace_tools.parse (layers come from the tensor name, a leading partial token is dropped).
usage: make_pin.py TRACE FRACTION OUT [--experts-per-layer E] [--compare EXISTING_PIN_FILE]
The hottest FRACTION of all (layer, expert) pairs is written. With --compare, prints how many pairs it shares with an existing file (an independent check of the loader)."""
import argparse, collections, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import trace_tools as tt
def build(trace_path, frac, E=None):
    tokens, st = tt.parse(open(trace_path)); cnt = collections.Counter((l, e) for tk in tokens for l in tk for e in tk[l])
    E = E or 1 + max(e for _, e in cnt); n = int(frac * len(tokens[0]) * E)
    order = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))          # ties broken by (layer, expert) so the result is deterministic
    return [pair for pair, _ in order[:n]], st
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("trace"); ap.add_argument("fraction", type=float); ap.add_argument("out"); ap.add_argument("--experts-per-layer", type=int); ap.add_argument("--compare"); a = ap.parse_args()
    pairs, st = build(a.trace, a.fraction, a.experts_per_layer)
    open(a.out, "w").write("".join(f"{l} {e}\n" for l, e in pairs)); print(f"{len(pairs)} pairs from {st['tokens']} tokens x {st['layers']} layers -> {a.out}")
    if a.compare:
        old = {tuple(map(int, l.split())) for l in open(a.compare) if l.strip()}; print(f"shared with {a.compare}: {len(old & set(pairs))} of {len(old)}")
