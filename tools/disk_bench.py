#!/usr/bin/env python3
"""Cache-bypassing read benchmark for a large file; works on macOS (F_NOCACHE) and Linux (O_DIRECT). No fio needed.

usage: disk_bench.py FILE [--seconds 10] [--block-kib 1024]
Prints one JSON line: {"seq_mib_s": .., "rand_mib_s": .., "block_kib": .., "method": ..}
Sequential: reads blocks from a random start. Random: reads block-aligned blocks at random offsets. The file should be larger than RAM,
and the numbers are only as cold as the bypass is: if the filesystem ignores the hint, a second pass over the same blocks would be cached,
so each pass here starts at a fresh random region and never rereads a block.
"""
import argparse, json, mmap, os, random, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("file"); ap.add_argument("--seconds", type=float, default=10); ap.add_argument("--block-kib", type=int, default=1024, help="read size; 4096-byte multiples (O_DIRECT needs alignment)")
a = ap.parse_args()
BLK = a.block_kib * 1024
size = os.path.getsize(a.file)
if size < 8 * BLK:
    sys.exit("file too small for a read benchmark")

if sys.platform == "darwin":
    import fcntl
    fd = os.open(a.file, os.O_RDONLY)
    fcntl.fcntl(fd, 48, 1)                      # F_NOCACHE
    method = "F_NOCACHE"
else:
    fd = os.open(a.file, os.O_RDONLY | getattr(os, "O_DIRECT", 0))
    method = "O_DIRECT"
buf = mmap.mmap(-1, BLK)                          # page-aligned, as O_DIRECT needs
nblk = size // BLK


def run(offsets):
    n, t0 = 0, time.time()
    for off in offsets:
        if time.time() - t0 >= a.seconds:
            break
        got = os.preadv(fd, [buf], off)
        if got <= 0:
            break
        n += got
    return n / 2**20 / max(time.time() - t0, 1e-9)


rng = random.Random(1234)
start = rng.randrange(0, max(1, nblk - 1))
seq = run(((start + i) % nblk) * BLK for i in range(nblk))
order = list(range(nblk)); rng.shuffle(order)
rnd = run(b * BLK for b in order)
print(json.dumps({"seq_mib_s": round(seq), "rand_mib_s": round(rnd), "block_kib": a.block_kib, "method": method}))
