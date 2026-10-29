"""mincore residency probe: consistency, a just-written file, range queries, short files, errors, and (if present) that reading a region of a real model raises its residency."""
import os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "tools")); sys.path.insert(0, HERE)
import mincore_probe as mp; from harness import T
def run():
    t = T("T07 mincore probe"); PAGE = mp.PAGE
    with tempfile.NamedTemporaryFile(delete=False) as f: f.write(os.urandom(8 * 2**20)); p = f.name
    r = mp.resident(p)
    t.check("a just-written 8 MiB file is (nearly) fully resident", r["fraction"] > 0.95, str(r))
    t.check("page count equals size / page size", r["pages"] == 8 * 2**20 // PAGE and r["bytes"] == 8 * 2**20)
    t.check("fraction equals resident pages / pages", abs(r["fraction"] - r["resident_pages"] / r["pages"]) < 1e-12 and r["resident_bytes"] == r["resident_pages"] * PAGE)
    for start, ln in ((0, PAGE), (PAGE, 3 * PAGE), (2**20, 2**20), (5 * 2**20 + 123, 100000), (8 * 2**20 - 1, 10)):
        q = mp.resident(p, start, ln); t.check(f"range query start {start} length {ln} stays within the file and 0 <= fraction <= 1", 0 <= q["fraction"] <= 1 and q["bytes"] <= ln and q["pages"] >= 1)
    t.check("a range beyond the end of the file is empty", mp.resident(p, 9 * 2**20, 1000)["pages"] == 0)
    t.check("length 0 is empty", mp.resident(p, 0, 0)["pages"] == 0)
    t.check("an empty file is handled", (lambda q: q["pages"] == 0)(mp.resident(tempfile.NamedTemporaryFile(delete=False).name)))
    try: mp.resident("/nonexistent/file"); ok = False
    except OSError: ok = True
    t.check("a missing file raises OSError", ok)
    t.check("probing does not read the file: residency does not rise from probing alone", mp.resident(p)["resident_pages"] == r["resident_pages"])
    os.unlink(p)
    big = os.path.expanduser("~/Downloads/Qwen3.8-27B-UD-IQ4_XS.gguf")
    if os.path.exists(big):
        size = os.path.getsize(big); a = mp.resident(big, size // 2, 256 * 2**20)
        with open(big, "rb") as g:                                    # read 256 MiB from the middle (this is what populates the cache)
            g.seek(size // 2)
            while g.tell() < size // 2 + 256 * 2**20: g.read(8 * 2**20)
        b = mp.resident(big, size // 2, 256 * 2**20)
        t.check(f"reading a 256 MiB region of a real model raises its residency ({a['fraction']:.2f} -> {b['fraction']:.2f})", b["fraction"] >= a["fraction"] and b["fraction"] > 0.5)
    return t
if __name__ == "__main__":
    sys.exit(0 if run().summary() else 1)
