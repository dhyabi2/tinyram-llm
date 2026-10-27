#!/usr/bin/env python3
"""build_ngram.py — turn the N-gram embedding table into the three files the runtime reads (design: `ngram-disk-lookup`).

  ngram_cold.q4k.bin   EVERY row, Q4_K, fixed stride (d=2048 -> 8 super-blocks x 144 B = 1152 B/row)  -> opened with pread + POSIX_FADV_RANDOM
  ngram_hot.q8.bin     the most-hit rows, Q8_0 (d=2048 -> 64 blocks x 34 B = 2176 B/row)               -> mmap'd
  ngram_hot.rows.u32   sorted row ids of the hot shard (position in the hot file = index into this list)
  ngram_owner.bitmap   1 bit per row, 1 = the row is in the hot shard
  ngram_manifest.json  tensors, strides, counts, sha256 of every output

Reads safetensors directly (header + offsets, no library), handles F32 / F16 / BF16, processes in chunks so memory stays flat.
The two quantizers follow llama.cpp's block layouts and are checked against the reference `gguf` library's dequantizer (test_build_ngram.py).
Q4_K here is a plain min/max quantizer (valid blocks, slightly less accurate than llama.cpp's iterative search).

usage: build_ngram.py --src <dir|file.safetensors> [--pattern ngram] --out-dir DIR [--hot-rows 65536] [--hits hits.npy] [--chunk-rows 16384]
"""
import argparse, glob, hashlib, json, os, re, struct, sys
import numpy as np

QK_K, QK8_0 = 256, 32

# ------------------------------------------------------------------------------------------ safetensors
_DT = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16}

def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n

def find_tensors(src, pattern):
    files = sorted(glob.glob(os.path.join(src, "*.safetensors"))) if os.path.isdir(src) else [src]
    found = []
    for fp in files:
        hdr, base = read_header(fp)
        for name, meta in hdr.items():
            if name != "__metadata__" and re.search(pattern, name):
                found.append((name, fp, base, meta))
    return sorted(found, key=lambda t: t[0])

def rows_f32(fp, base, meta, start, stop):
    """Rows [start, stop) of a 2-D tensor as float32."""
    rows, d = meta["shape"]
    off = base + meta["data_offsets"][0]
    dt = _DT[meta["dtype"]]
    mm = np.memmap(fp, dtype=dt, mode="r", offset=off, shape=(rows, d))
    a = np.asarray(mm[start:stop])
    if meta["dtype"] == "BF16":
        a = (a.astype(np.uint32) << 16).view(np.float32)
    return a.astype(np.float32, copy=False)

# ------------------------------------------------------------------------------------------ quantizers
def quantize_q4_k(x):
    """x: float32 [n, d] with d % 256 == 0  ->  uint8 [n, d/256*144]   (block_q4_K: d, dmin, scales[12], qs[128])"""
    n, d = x.shape
    b = x.reshape(-1, 8, 32)                                   # [blocks, 8 sub-blocks, 32 values]
    mn = np.minimum(b.min(-1), 0.0)                            # ggml: a positive minimum is clamped to 0
    mx = b.max(-1)
    scale = (mx - mn) / 15.0                                   # per sub-block step
    mins = -mn                                                 # >= 0 : the stored offset
    max_scale = scale.max(-1, keepdims=True); max_min = mins.max(-1, keepdims=True)
    inv_s = np.divide(63.0, max_scale, out=np.zeros_like(max_scale), where=max_scale > 0)
    inv_m = np.divide(63.0, max_min, out=np.zeros_like(max_min), where=max_min > 0)
    ls = np.minimum(63, np.rint(inv_s * scale)).astype(np.uint8)
    lm = np.minimum(63, np.rint(inv_m * mins)).astype(np.uint8)
    d16 = (max_scale / 63.0).astype(np.float16); dm16 = (max_min / 63.0).astype(np.float16)
    sub_d = d16.astype(np.float32) * ls; sub_m = dm16.astype(np.float32) * lm
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(sub_d[..., None] > 0, np.rint((b + sub_m[..., None]) / sub_d[..., None]), 0)
    q = np.clip(q, 0, 15).astype(np.uint8)                     # [blocks, 8, 32]
    sc = np.zeros((b.shape[0], 12), np.uint8)
    sc[:, 0:4] = ls[:, 0:4]; sc[:, 4:8] = lm[:, 0:4]
    for j in range(4, 8):
        sc[:, j + 4] = (ls[:, j] & 0xF) | ((lm[:, j] & 0xF) << 4)
        sc[:, j - 4] |= (ls[:, j] >> 4) << 6
        sc[:, j] |= (lm[:, j] >> 4) << 6
    qs = np.empty((b.shape[0], 128), np.uint8)
    for c in range(4):                                         # sub-blocks (2c, 2c+1) share 32 bytes: low nibble / high nibble
        qs[:, 32 * c:32 * c + 32] = q[:, 2 * c, :] | (q[:, 2 * c + 1, :] << 4)
    out = np.concatenate([d16.view(np.uint8).reshape(-1, 2), dm16.view(np.uint8).reshape(-1, 2), sc, qs], axis=1)   # [blocks, 144]
    return out.reshape(n, -1)

def dequantize_q4_k(raw, d):
    """Inverse of quantize_q4_k (own implementation, used for the round-trip check)."""
    blocks = raw.reshape(-1, 144)
    dd = blocks[:, 0:2].copy().view(np.float16).astype(np.float32); dmin = blocks[:, 2:4].copy().view(np.float16).astype(np.float32)
    sc = blocks[:, 4:16]; qs = blocks[:, 16:]
    ls = np.empty((blocks.shape[0], 8), np.uint8); lm = np.empty_like(ls)
    ls[:, :4] = sc[:, 0:4] & 63; lm[:, :4] = sc[:, 4:8] & 63
    for j in range(4, 8):
        ls[:, j] = (sc[:, j + 4] & 0xF) | ((sc[:, j - 4] >> 6) << 4)
        lm[:, j] = (sc[:, j + 4] >> 4) | ((sc[:, j] >> 6) << 4)
    out = np.empty((blocks.shape[0], 8, 32), np.float32)
    for c in range(4):
        chunk = qs[:, 32 * c:32 * c + 32]
        out[:, 2 * c, :] = (chunk & 0xF)
        out[:, 2 * c + 1, :] = (chunk >> 4)
    out = dd[:, None, :] * ls[:, :, None] * out - dmin[:, None, :] * lm[:, :, None]
    return out.reshape(-1, d)

def quantize_q8_0(x):
    """x: float32 [n, d] with d % 32 == 0  ->  uint8 [n, d/32*34]   (block_q8_0: d(fp16), qs[32] int8)"""
    n, d = x.shape
    b = x.reshape(-1, 32)
    amax = np.abs(b).max(-1)
    scale = (amax / 127.0).astype(np.float16)
    s32 = scale.astype(np.float32)
    q = np.where(s32[:, None] > 0, np.rint(b / np.where(s32 > 0, s32, 1)[:, None]), 0).astype(np.int8)
    out = np.concatenate([scale.view(np.uint8).reshape(-1, 2), q.view(np.uint8)], axis=1)
    return out.reshape(n, -1)

def dequantize_q8_0(raw, d):
    blocks = raw.reshape(-1, 34)
    s = blocks[:, 0:2].copy().view(np.float16).astype(np.float32)
    return (blocks[:, 2:].copy().view(np.int8).astype(np.float32) * s).reshape(-1, d)

# ------------------------------------------------------------------------------------------ build
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""): h.update(chunk)
    return h.hexdigest()

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--pattern", default="ngram"); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hot-rows", type=int, default=65536); ap.add_argument("--hits", help=".npy of per-row hit counts, concatenated in tensor-name order")
    ap.add_argument("--chunk-rows", type=int, default=16384)
    a = ap.parse_args(argv)

    tensors = find_tensors(a.src, a.pattern)
    if not tensors: sys.exit(f"no 2-D tensor matching /{a.pattern}/ in {a.src}")
    dims = {m["shape"][1] for _, _, _, m in tensors if len(m["shape"]) == 2}
    if len(dims) != 1 or any(len(m["shape"]) != 2 for _, _, _, m in tensors): sys.exit(f"expected 2-D tensors of one width, got {[m['shape'] for *_, m in tensors]}")
    d = dims.pop()
    if d % QK_K: sys.exit(f"row width {d} is not a multiple of {QK_K}")
    total = sum(m["shape"][0] for *_, m in tensors)
    cold_stride, hot_stride = d // QK_K * 144, d // QK8_0 * 34
    os.makedirs(a.out_dir, exist_ok=True)

    # which rows are hot
    n_hot = min(a.hot_rows, total)
    if a.hits:
        hits = np.load(a.hits)
        if hits.shape[0] != total: sys.exit(f"--hits has {hits.shape[0]} entries for {total} rows")
        hot_ids = np.sort(np.argsort(-hits, kind="stable")[:n_hot]).astype(np.uint32)
    else:
        print("WARNING: no --hits given; the first rows are used as the hot shard. Size it from a real hit-count pass.", file=sys.stderr)
        hot_ids = np.arange(n_hot, dtype=np.uint32)
    hot_set = np.zeros(total, bool); hot_set[hot_ids] = True
    hot_pos = np.cumsum(hot_set) - 1                                   # row id -> position in the hot file

    cold_p, hot_p = (os.path.join(a.out_dir, n) for n in ("ngram_cold.q4k.bin", "ngram_hot.q8.bin"))
    first = 0; manifest_t = []
    with open(cold_p, "wb") as cold, open(hot_p, "wb") as hot:
        for name, fp, base, meta in tensors:
            rows = meta["shape"][0]
            manifest_t.append({"name": name, "rows": rows, "first_row": first, "dtype": meta["dtype"], "file": os.path.basename(fp)})
            for s in range(0, rows, a.chunk_rows):
                x = rows_f32(fp, base, meta, s, min(s + a.chunk_rows, rows))
                cold.write(quantize_q4_k(x).tobytes())
                sel = hot_set[first + s:first + s + x.shape[0]]
                if sel.any(): hot.write(quantize_q8_0(x[sel]).tobytes())
            first += rows
    np.packbits(hot_set, bitorder="little").tofile(os.path.join(a.out_dir, "ngram_owner.bitmap"))
    hot_ids.tofile(os.path.join(a.out_dir, "ngram_hot.rows.u32"))

    man = {"rows": int(total), "d": int(d), "cold_stride_bytes": int(cold_stride), "hot_stride_bytes": int(hot_stride), "hot_rows": int(n_hot),
           "bitmap_bit_order": "little", "row_address": "byte_offset = row_id * cold_stride_bytes", "tensors": manifest_t, "sha256": {}}
    for n in ("ngram_cold.q4k.bin", "ngram_hot.q8.bin", "ngram_owner.bitmap", "ngram_hot.rows.u32"):
        man["sha256"][n] = sha256_file(os.path.join(a.out_dir, n))
    json.dump(man, open(os.path.join(a.out_dir, "ngram_manifest.json"), "w"), indent=1)
    print(f"{total} rows x {d}: cold {os.path.getsize(cold_p)/1e9:.3f} GB ({cold_stride} B/row), hot {n_hot} rows {os.path.getsize(hot_p)/1e6:.1f} MB ({hot_stride} B/row)")
    return man

if __name__ == "__main__":
    main()
