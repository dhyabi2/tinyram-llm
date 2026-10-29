"""make_pin: the shared loader builds the right pin list (brute-force count), deterministically, and agrees with the pin files used in the VPS test when they exist."""
import collections, os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "tools")); sys.path.insert(0, HERE)
import make_pin as mp; from tracegen import gen_trace; from harness import T
def run():
    t = T("T03 pin builder"); root = os.path.join(HERE, "..", "data", "traces"); db = os.path.join(root, "qwen3-30b_db.txt")
    for name, frac, want in (("qwen3-30b_db", 0.05, 307), ("qwen3-30b_db", 0.10, 614), ("qwen3-30b_story", 0.10, 614), ("qwen3-30b_code", 0.10, 614)):
        pairs, st = mp.build(os.path.join(root, name + ".txt"), frac, 128)
        t.check(f"{name} {int(frac*100)}%: {want} pairs, all layers 0..47 and experts 0..127", len(pairs) == want and all(0 <= l < 48 and 0 <= e < 128 for l, e in pairs), f"{len(pairs)}")
        # brute-force count straight from the text, no shared code
        cnt = collections.Counter(); cur = None
        for line in open(os.path.join(root, name + ".txt")):
            f = line.split()
            if len(f) == 9 and f[0].startswith("blk."): cnt[(int(f[0].split(".")[1]), *map(int, f[1:]))] += 0
        raw = collections.Counter()
        recs = [l.split() for l in open(os.path.join(root, name + ".txt")) if l.startswith("blk.")]
        while recs and not recs[0][0].startswith("blk.0."): recs = recs[1:]
        recs = recs[: len(recs) // 48 * 48]
        for f in recs:
            for x in f[1:]: raw[(int(f[0].split(".")[1]), int(x))] += 1
        top = sorted(raw.items(), key=lambda kv: (-kv[1], kv[0]))[:want]
        t.check(f"{name} {int(frac*100)}%: equals an independent top-{want} count", set(pairs) == {k for k, _ in top})
    a, _ = mp.build(db, 0.10, 128); b, _ = mp.build(db, 0.10, 128)
    t.check("the list is deterministic", a == b)
    t.check("a bigger fraction contains the smaller one", set(mp.build(db, 0.05, 128)[0]) <= set(mp.build(db, 0.10, 128)[0]))
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f: f.write("".join(gen_trace(4, 16, 2, 1.0, 0.5, 30, 3)) if False else "\n".join(gen_trace(4, 16, 2, 1.0, 0.5, 30, 3)))
    pairs, st = mp.build(f.name, 0.25, 16); os.unlink(f.name)
    t.check("a synthetic trace builds a list of the right size", len(pairs) == int(0.25 * 4 * 16), str(len(pairs)))
    old = os.environ.get("TINYRAM_OLD_PIN_FILE", "")   # optional: a pin file from an earlier run to compare against
    if os.path.exists(old):
        mine = set(mp.build(db, 0.10, 128)[0]); theirs = {tuple(map(int, l.split())) for l in open(old) if l.strip()}
        t.check(f"overlap with the pin file used in the VPS test is {len(mine & theirs)} of {len(theirs)} (must be >= 600)", len(mine & theirs) >= 600)
    return t
if __name__ == "__main__":
    sys.exit(0 if run().summary() else 1)
