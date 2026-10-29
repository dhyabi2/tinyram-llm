#!/usr/bin/env python3
"""Explicit-read throughput for expert-sized extents: random aligned extents of a big file, read with os.pread (cache bypass), at several read sizes and thread counts,
and the same extents touched through mmap for comparison. usage: read_bench.py FILE [--seconds 6]
Prints one JSON line per configuration. The mmap pass cannot be made cold without sudo purge on macOS, so it is a LOWER-bound comparison only if pages are not cached; the pread numbers are cache-bypassing."""
import argparse, json, mmap, os, random, sys, time, threading
ap = argparse.ArgumentParser(); ap.add_argument("file"); ap.add_argument("--seconds", type=float, default=6); a = ap.parse_args()
size = os.path.getsize(a.file); PAGE = 4096
def nocache(fd):
    if sys.platform == "darwin":
        import fcntl; fcntl.fcntl(fd, 48, 1)
def pread_run(ext, threads, secs):
    fd = os.open(a.file, os.O_RDONLY | (getattr(os, "O_DIRECT", 0) if sys.platform != "darwin" else 0)); nocache(fd); total = [0]; stop = time.time() + secs
    def work(seed):
        rng = random.Random(seed); buf = mmap.mmap(-1, ext); n = 0
        while time.time() < stop:
            off = rng.randrange(0, (size - ext) // PAGE) * PAGE; n += os.preadv(fd, [buf], off)
        total[0] += n
    ts = [threading.Thread(target=work, args=(i,)) for i in range(threads)]; t0 = time.time(); [t.start() for t in ts]; [t.join() for t in ts]
    return total[0] / (time.time() - t0) / 1e9
def mmap_run(ext, secs):
    fd = os.open(a.file, os.O_RDONLY); m = mmap.mmap(fd, 0, access=mmap.ACCESS_READ); rng = random.Random(99); n = 0; stop = time.time() + secs; t0 = time.time()
    while time.time() < stop:
        off = rng.randrange(0, (size - ext) // PAGE) * PAGE
        for p in range(off, off + ext, PAGE): m[p]
        n += ext
    return n / (time.time() - t0) / 1e9
for ext_mb in (0.9, 2.7, 4.0):
    ext = int(ext_mb * 1048576) // PAGE * PAGE
    for th in (1, 4, 8):
        print(json.dumps({"extent_mb": ext_mb, "threads": th, "method": "pread", "gb_s": round(pread_run(ext, th, a.seconds), 3)}), flush=True)
    print(json.dumps({"extent_mb": ext_mb, "threads": 1, "method": "mmap page touch", "gb_s": round(mmap_run(ext, a.seconds), 3)}), flush=True)
