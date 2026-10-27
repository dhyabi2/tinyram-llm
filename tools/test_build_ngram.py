#!/usr/bin/env python3
"""Tests for build_ngram.py. Expects `bn` (the loaded build_ngram module) to be provided by the runner (see run_ngram_test.py)."""
import json, os, struct, tempfile, hashlib
import numpy as np

rng = np.random.default_rng(11)
RES = []
def check(name, ok, detail=""):
    RES.append(bool(ok)); print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))

D = 2048
def write_safetensors(path, tensors):
    hdr, blobs, off = {}, [], 0
    for name, (dtype, arr) in tensors.items():
        raw = arr.tobytes(); hdr[name] = {"dtype": dtype, "shape": list(arr.shape), "data_offsets": [off, off + len(raw)]}; blobs.append(raw); off += len(raw)
    h = json.dumps(hdr).encode(); h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h))); f.write(h)
        for b in blobs: f.write(b)

def to_bf16(x32): return (x32.view(np.uint32) >> 16).astype(np.uint16)

tmp = tempfile.mkdtemp(); src = os.path.join(tmp, "model.safetensors"); out = os.path.join(tmp, "out")
t2 = (rng.standard_normal((3000, D)) * 0.05).astype(np.float32)
t3 = (rng.standard_normal((1500, D)) * 0.2).astype(np.float32)
write_safetensors(src, {"model.ngram.order2": ("F16", t2.astype(np.float16)), "model.ngram.order3": ("BF16", to_bf16(t3)), "model.embed_tokens": ("F16", rng.standard_normal((10, D)).astype(np.float16))})
hits = rng.integers(0, 1000, 4500).astype(np.int64); np.save(os.path.join(tmp, "hits.npy"), hits)

man = bn.main(["--src", src, "--out-dir", out, "--hot-rows", "256", "--hits", os.path.join(tmp, "hits.npy"), "--chunk-rows", "700"])
total = 4500
check("only the matching tensors are used (decoy ignored, 2 tensors, 4,500 rows)", man["rows"] == total and len(man["tensors"]) == 2 and man["tensors"][1]["first_row"] == 3000)
sz = lambda n: os.path.getsize(os.path.join(out, n))
check("cold file: every row at a fixed 1,152-byte stride", sz("ngram_cold.q4k.bin") == total * 1152, f"{sz('ngram_cold.q4k.bin')} B = {total} x 1152")
check("hot file: 256 rows at a fixed 2,176-byte stride", sz("ngram_hot.q8.bin") == 256 * 2176, f"{sz('ngram_hot.q8.bin')} B")
check("bitmap is 1 bit per row; hot id list is 4 bytes per hot row", sz("ngram_owner.bitmap") == (total + 7) // 8 and sz("ngram_hot.rows.u32") == 256 * 4)

cold = np.fromfile(os.path.join(out, "ngram_cold.q4k.bin"), np.uint8).reshape(total, 1152)
src_rows = np.concatenate([t2.astype(np.float16).astype(np.float32), (to_bf16(t3).astype(np.uint32) << 16).view(np.float32)])   # what the file actually stores
rel = lambda a, b: float(np.sqrt(((a - b) ** 2).mean()) / np.sqrt((b ** 2).mean()))
sample = rng.choice(total, 40, replace=False); sample = np.concatenate([sample, [0, 2999, 3000, 4499]])
errs = [rel(bn.dequantize_q4_k(cold[r:r + 1], D)[0], src_rows[r]) for r in sample]
check("Q4_K rows round-trip with a relative RMS error under 12% (min/max 4-bit on Gaussian data)", max(errs) < 0.12, f"worst {max(errs):.3f}, mean {np.mean(errs):.3f}")
check("rows on both sides of the F16 / BF16 tensor boundary decode correctly", all(rel(bn.dequantize_q4_k(cold[r:r + 1], D)[0], src_rows[r]) < 0.12 for r in (2999, 3000)))

hot_ids = np.fromfile(os.path.join(out, "ngram_hot.rows.u32"), np.uint32); hot = np.fromfile(os.path.join(out, "ngram_hot.q8.bin"), np.uint8).reshape(256, 2176)
expect_ids = np.sort(np.argsort(-hits, kind="stable")[:256])
check("the hot shard holds exactly the 256 most-hit rows, sorted by row id", np.array_equal(hot_ids, expect_ids))
herr = [rel(bn.dequantize_q8_0(hot[p:p + 1], D)[0], src_rows[hot_ids[p]]) for p in rng.choice(256, 40, replace=False)]
check("hot rows (Q8_0) decode to the right source row with under 1% error", max(herr) < 0.01, f"worst {max(herr):.4f}")
bits = np.unpackbits(np.fromfile(os.path.join(out, "ngram_owner.bitmap"), np.uint8), bitorder="little")[:total]
check("the ownership bitmap marks exactly the hot rows", bits.sum() == 256 and np.array_equal(np.flatnonzero(bits), hot_ids))
check("manifest checksums match the files", all(hashlib.sha256(open(os.path.join(out, n), "rb").read()).hexdigest() == h for n, h in man["sha256"].items()))

# ---- byte-format compatibility with the reference gguf library
try:
    import gguf
    from gguf.quants import dequantize
    gq = dequantize(cold[:64], gguf.GGMLQuantizationType.Q4_K)
    mine = bn.dequantize_q4_k(cold[:64], D)
    check("Q4_K bytes decode identically with the reference gguf library", np.abs(gq - mine).max() < 1e-3, f"max |diff| vs gguf {np.abs(gq - mine).max():.2e}")
    gq8 = dequantize(hot[:64], gguf.GGMLQuantizationType.Q8_0)
    check("Q8_0 bytes decode identically with the reference gguf library", np.abs(gq8 - bn.dequantize_q8_0(hot[:64], D)).max() < 1e-4, f"max |diff| vs gguf {np.abs(gq8 - bn.dequantize_q8_0(hot[:64], D)).max():.2e}")
    check("the reference library also reconstructs the source (error under 12%)", rel(gq[5], src_rows[5]) < 0.12, f"{rel(gq[5], src_rows[5]):.3f}")
except ImportError as e:
    check("reference gguf library available", False, str(e))

# ---- error handling
bad = os.path.join(tmp, "bad.safetensors"); write_safetensors(bad, {"model.ngram.x": ("F16", np.zeros((10, 1000), np.float16))})
try: bn.main(["--src", bad, "--out-dir", os.path.join(tmp, "o2")]); rejected = False
except SystemExit as e: rejected = "not a multiple of 256" in str(e)
check("a row width that is not a multiple of 256 is rejected", rejected)
try: bn.main(["--src", src, "--pattern", "nomatch", "--out-dir", os.path.join(tmp, "o3")]); none = False
except SystemExit as e: none = "no 2-D tensor" in str(e)
check("a pattern that matches nothing is an error", none)

print(f"\n{sum(RES)}/{len(RES)} checks passed")
raise SystemExit(0 if all(RES) else 1)
