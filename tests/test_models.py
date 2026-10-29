"""T04 model outputs: throughput_model_v2, ram_sweep and hundredx_map printed tables obey their own physics (monotone, bounded by compute, serial <= overlap, units)."""
import os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); from harness import T
TOOLS = os.path.join(HERE, "..", "tools")
def sh(*a):
    return subprocess.run([sys.executable, "-I", *a], capture_output=True, text=True, cwd=TOOLS, timeout=300).stdout
def run():
    t = T("T04 model outputs")
    out = sh("throughput_model_v2.py")
    t.check("expert bytes per token = 480 slices x 2.7 MB = 1.30 GB", "1.30 GB (480 slices x 2.7 MB)" in out and abs(480 * 2.7 / 1000 - 1.296) < 1e-9)
    ceil = float(re.search(r"compute ceiling ([\d.]+) tok/s", out).group(1)); t.check("compute ceiling = 55e9/12e9 = 4.58", abs(ceil - 55 / 12) < 0.01)
    rows = re.findall(r"^(\d+) GB\s+([\d.]+)\s+([\d.]+)\s+([\d.]+) \| (.+?)\s+([\d.]+)\s+([\d.]+)$", out, re.M)
    t.check("3 RAM sizes x 4 read paths = 12 rows", len(rows) == 12, str(len(rows)))
    byram = {}
    for ram, gib, fp, hit, path, ser, ov in rows:
        ser, ov, hit = float(ser), float(ov), float(hit)
        t.check(f"{ram} GB {path}: serial <= overlap", ser <= ov + 0.006)
        t.check(f"{ram} GB {path}: overlap <= compute ceiling", ov <= ceil + 0.006)
        t.check(f"{ram} GB {path}: serial < compute ceiling", ser < ceil)
        t.check(f"{ram} GB {path}: hit rate in [0,1]", 0 <= hit <= 1)
        t.check(f"{ram} GB {path}: throughput positive", ser > 0 and ov > 0)
        byram.setdefault(ram, []).append((path, ser, ov, hit))
    for ram, v in byram.items():
        for (p1, s1, o1, _), (p2, s2, o2, _) in zip(v, v[1:]):
            t.check(f"{ram} GB: faster read path ({p2}) is not slower than ({p1})", s2 >= s1 - 0.006 and o2 >= o1 - 0.006)
    rams = sorted(byram, key=int)
    for a, b in zip(rams, rams[1:]):
        for (p, s1, o1, h1), (_, s2, o2, h2) in zip(byram[a], byram[b]):
            t.check(f"more RAM ({b} vs {a} GB) never lowers hit or speed on {p}", h2 >= h1 - 1e-9 and s2 >= s1 - 0.006)
    # ram sweep
    out = sh("ram_sweep.py"); blocks = out.split("=== CPU decode rate")[1:]
    t.check("ram_sweep prints three CPU speeds", len(blocks) == 3, str(len(blocks)))
    for blk in blocks:
        ceil = float(re.search(r"compute ceiling ([\d.]+)", blk).group(1)); prev = None
        for m in re.finditer(r"^\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)x\s+(\d+)%", blk, re.M):
            ram, cache, hit, bt, bs, ov, mm, ratio, pct = m.groups(); ram = int(ram); bs, ov, mm, ratio = map(float, (bs, ov, mm, ratio))
            t.check(f"ceiling {ceil}: {ram} GB base <= ceiling", bs <= ceil + 0.011); t.check(f"ceiling {ceil}: {ram} GB mmap <= base (+overlap)", mm <= ov + 0.011)
            t.check(f"ceiling {ceil}: {ram} GB hit in [0,1]", 0 <= float(hit) <= 1)
            t.check(f"ceiling {ceil}: {ram} GB cache GiB = 0.944*RAM-5.05 (floored at 0.3)", abs(float(cache) - max(0.3, 0.944 * ram - 5.05)) < 0.12, f"{cache}")
            t.check(f"ceiling {ceil}: {ram} GB base/mmap ratio matches the two columns", abs(ratio - bs / mm) < 0.06 * ratio + 0.06)
            if prev: t.check(f"ceiling {ceil}: base speed does not fall from {prev[0]} to {ram} GB", bs >= prev[1] - 1e-9)
            prev = (ram, bs)
        t.check(f"ceiling {ceil}: sweep reached a row", prev is not None)
    # 100x map
    out = sh("hundredx_map.py"); secs = out.split("=== ")[1:]; t.check("hundredx_map prints 3 machine sizes", len(secs) >= 3, str(len(secs)))
    for sec in secs[:3]:
        name = sec.split(" machine")[0]
        for m in re.finditer(r"^(.+?)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+) tok/s", sec, re.M):
            st, risk, c, l, o, b = m.groups(); c, l, o, b = map(float, (c, l, o, b))
            t.check(f"{name} {st.strip()[:40]}: cons <= opt", c <= o + 1e-9); t.check(f"{name} {st.strip()[:40]}: B* = F/100", abs(b - l / 100) < 0.0006 or abs(b - l / 100) < 0.01 * l / 100 + 0.0006)
            t.check(f"{name} {st.strip()[:40]}: risk 0..12", 0 <= int(risk) <= 12)
    # real-model hit curve (COMBO_HIT=real)
    sys.path.insert(0, TOOLS); import combo_model as cm
    ks = sorted(cm.LRU_REAL); hs = [cm.LRU_REAL[k] for k in ks]
    t.check("real hit curve is monotone non-decreasing in cache size", all(b >= a for a, b in zip(hs, hs[1:])))
    t.check("real hit curve stays within [0, 1]", all(0 <= h <= 1 for h in hs))
    t.check("real hit curve stays below the 30B-trace curve between 4 and 16 footprints (the 30B plateau is a cold-start artefact, so beyond that the real curve can exceed it)", all(cm.LRU_REAL[k] <= cm.lru(k) + 1e-9 for k in ks if 4 <= k <= 16))
    env = dict(os.environ, COMBO_HIT="real")
    out_real = subprocess.run([sys.executable, "-I", "ram_sweep.py"], capture_output=True, text=True, cwd=TOOLS, env=env, timeout=300).stdout
    rows_r = {int(m.group(1)): float(m.group(5)) for m in re.finditer(r"^\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", out_real, re.M)}
    rows_d = {int(m.group(1)): float(m.group(5)) for m in re.finditer(r"^\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", sh("ram_sweep.py"), re.M)}
    t.check("under the real curve the 16 GB base path is slower than under the 30B curve", rows_r.get(16, 9) < rows_d.get(16, 0))
    t.check("under the real curve the 8 GB base path is slower than under the 30B curve", rows_r.get(8, 9) < rows_d.get(8, 0))
    t.check("real-curve base speed does not fall as RAM grows", all(rows_r[b] >= rows_r[a] - 1e-9 for a, b in zip(sorted(rows_r), sorted(rows_r)[1:])))
    return t
if __name__ == "__main__": sys.exit(0 if run().summary() else 1)
