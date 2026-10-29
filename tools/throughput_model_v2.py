#!/usr/bin/env python3
"""Throughput model v2 (replaces the percent-of-experts / dense-charged-to-disk version).

Every input is printed with its source tag: [M] measured here, [E] estimate, [A] assumption.
Corrections over v1:
  - cache size is expressed in token-FOOTPRINTS (cache entries / expert accesses per token), because LRU behaviour is set by that ratio
    (30B: 48 layers x 8 = 384 per token; 125B: 48 x 10 = 480), not by percent of all experts;
  - only EXPERT bytes are charged to the disk: the always-on dense part is resident (about 1.9 GB for the 125B at 4.3 bpw) and is not read per token;
  - two bounds are printed: serial 1/(io+compute) and overlap min(compute, rate/bytes). No code overlaps I/O with compute today, so serial is the honest one;
  - read size sets the disk rate: 0.34 GB/s is what mmap paging reached on the Mac, 1.56 / 2.38 GB/s are 1 MiB / 4 MiB cache-bypassing reads.
usage: throughput_model_v2.py [trace.txt ...]   (router traces from GGML_MOE_TRACE; without them a measured LRU table is used)
"""
import sys, collections, statistics as st

EXPERT_MB, SLICES_PER_TOKEN = 2.7, 48 * 10                  # [E] 125B target: gate+up+down ~2.7 MB per expert, 480 expert slices per token
EXP_GB = SLICES_PER_TOKEN * EXPERT_MB / 1000                # disk bytes per token at hit 0
COMPUTE = 55e9 / (2 * 6e9)                                  # [M] 55 GFLOP/s (Mac, cached decode) / 12 GFLOP per token = 4.58 tok/s
RATES = {"mmap paging": 0.34, "explicit 1 MiB QD1": 1.56, "explicit 4 MiB QD1": 2.38, "explicit 2.7 MB x4 threads": 3.11}   # [M] GB/s on the Mac internal NVMe (tools/read_bench.py: an expert read as one 2.7 MB extent with 4 reads in flight)
PCS = {"8 GB": 2.5, "16 GB": 9.15, "32 GB": 25.15}          # [E] expert cache GiB left after OS, KV, dense (plan F)

def load(f):
    L = [l.split() for l in open(f) if l.startswith("blk.")]
    while L and not L[0][0].startswith("blk.0."): L = L[1:]
    n = len(L) // 48; L = L[: n * 48]
    return [(int(l[0].split(".")[1]), int(x)) for l in L for x in l[1:]]

def lru(tr, cap):
    c = collections.OrderedDict(); h = 0
    for a in tr:
        if a in c: c.move_to_end(a); h += 1
        else:
            c[a] = 1
            if len(c) > cap: c.popitem(last=False)
    return h / len(tr)

TABLE = {0.9: 0.000, 1.0: 0.306, 1.5: 0.454, 2.0: 0.549, 3.0: 0.686, 4.0: 0.793, 7.5: 0.950, 10: 0.962, 20.6: 0.962}   # [M] mean of 3 prompts, cold start included
if len(sys.argv) > 1:
    traces = [load(f) for f in sys.argv[1:]]
    TABLE = {k: st.mean(lru(t, int(k * 384)) for t in traces) for k in TABLE}
    print(f"LRU table recomputed from {len(traces)} trace(s)")
ks = sorted(TABLE)
def hit(k):
    if k <= ks[0]: return TABLE[ks[0]]
    for a, b in zip(ks, ks[1:]):
        if a <= k <= b: return TABLE[a] + (TABLE[b] - TABLE[a]) * (k - a) / (b - a)
    return TABLE[ks[-1]]

print(f"[E] expert bytes per token at hit 0 = {EXP_GB:.2f} GB ({SLICES_PER_TOKEN} slices x {EXPERT_MB} MB); [M] compute ceiling {COMPUTE:.2f} tok/s; dense part resident, not charged")
print(f"{'PC':<6}{'cache GiB':>10}{'footprints':>11}{'LRU hit':>9} | {'read path':<16}{'serial':>8}{'overlap':>9}")
for pc, gib in PCS.items():
    k = gib * 1.074e9 / (EXPERT_MB * 1e6) / SLICES_PER_TOKEN
    h = hit(k)
    for name, rate in RATES.items():
        b = (1 - h) * EXP_GB
        ser = 1 / (b / rate + 1 / COMPUTE); ov = min(COMPUTE, rate / b) if b > 0 else COMPUTE
        print(f"{pc:<6}{gib:>10.2f}{k:>11.1f}{h:>9.2f} | {name:<16}{ser:>8.2f}{ov:>9.2f}")
    print()
print("Notes: hit rates come from a 30B model (128 experts, top-8) mapped by FOOTPRINT; the 125B has never run, so every row is an extrapolation [A].")
print("No prefill/time-to-first-token term is included: all traces are decode-only (prefill touches most experts). The pinned-hot-set path is not modelled here because a share-based pin loses to LRU at equal bytes in these traces.")
