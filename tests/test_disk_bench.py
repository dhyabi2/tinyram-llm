"""T07 disk_bench: JSON shape, block sizes, short files, error paths, platform method label."""
import json, os, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); from harness import T
DB = os.path.join(HERE, "..", "tools", "disk_bench.py")
def go(path, *a):
    return subprocess.run([sys.executable, "-I", DB, path, *a], capture_output=True, text=True, timeout=120)
def run():
    t = T("T07 disk_bench"); d = tempfile.mkdtemp()
    f = os.path.join(d, "big.bin"); open(f, "wb").write(os.urandom(64 * 2**20))
    for kib in (4, 16, 64, 256, 1024, 4096):
        r = go(f, "--seconds", "1", "--block-kib", str(kib)); ok = r.returncode == 0
        t.check(f"{kib} KiB: exits 0", ok, r.stderr[-200:])
        j = json.loads(r.stdout.strip().splitlines()[-1]) if ok and r.stdout.strip() else {}
        t.check(f"{kib} KiB: JSON has the four fields", set(j) == {"seq_mib_s", "rand_mib_s", "block_kib", "method"})
        t.check(f"{kib} KiB: block_kib echoed", j.get("block_kib") == kib)
        t.check(f"{kib} KiB: rates positive integers", isinstance(j.get("seq_mib_s"), int) and j.get("seq_mib_s", 0) > 0 and j.get("rand_mib_s", 0) > 0)
        t.check(f"{kib} KiB: method matches the platform", j.get("method") == ("F_NOCACHE" if sys.platform == "darwin" else "O_DIRECT"))
    r = go(f, "--seconds", "1", "--block-kib", "1024"); r2 = go(f, "--seconds", "1", "--block-kib", "1024")
    t.check("two runs agree within 10x (no run is broken)", r.returncode == 0 and r2.returncode == 0 and 0.1 < json.loads(r.stdout)["seq_mib_s"] / max(1, json.loads(r2.stdout)["seq_mib_s"]) < 10)
    small = os.path.join(d, "small.bin"); open(small, "wb").write(os.urandom(1000))
    r = go(small); t.check("a file smaller than 8 blocks is refused", r.returncode != 0 and "too small" in r.stderr)
    t.check("a missing file fails with non-zero exit", go(os.path.join(d, "nope")).returncode != 0)
    t.check("a directory fails with non-zero exit", go(d).returncode != 0)
    t.check("a non-numeric block size is rejected", go(f, "--block-kib", "abc").returncode != 0)
    t.check("a block size larger than the file / 8 is refused", go(f, "--block-kib", "16384", "--seconds", "1").returncode != 0)
    t.check("--seconds 0 returns quickly without crashing", go(f, "--seconds", "0").returncode in (0, 1))
    open(os.path.join(d, "empty.bin"), "wb").close(); t.check("an empty file is refused", go(os.path.join(d, "empty.bin")).returncode != 0)
    return t
if __name__ == "__main__": sys.exit(0 if run().summary() else 1)
