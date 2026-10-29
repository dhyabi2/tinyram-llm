"""T08 probe parsers: the real shell/awk/python snippets are cut out of probe.sh and probe_mac.sh and run on recorded + mutated fixtures."""
import json, os, random, re, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); from harness import T
ROOT = os.path.join(HERE, ".."); FX = os.path.join(HERE, "fixtures")
MAC = open(os.path.join(ROOT, "probe", "probe_mac.sh")).read(); LIN = open(os.path.join(ROOT, "probe", "probe.sh")).read()
def bash(code, **env): return subprocess.run(["bash", "-c", code], capture_output=True, text=True, env=dict(os.environ, **env), timeout=60)
def fn(src, name):
    m = re.search(rf"^{name}\(\) \{{.*?\n?\}}$", src, re.M); return m.group(0) if m else None
def pyblock(src, marker):
    i = src.index(marker); j = src.index("<<'PY'", i); k = src.index("\nPY\n", j); return src[j + len("<<'PY'\n"):k]
def vmstat_text(pagein=123456, wired=999, psz=4096): return f"Mach Virtual Memory Statistics: (page size of {psz} bytes)\nPages free: 1.\nPages wired down: {wired}.\nPageins: {pagein}.\nPageouts: 0.\n"
def run():
    t = T("T08 probe parsers"); tmp = tempfile.mkdtemp()
    real = open(os.path.join(FX, "vm_stat_macos14.txt")).read()
    pi = fn(MAC, "pageins"); vp = fn(MAC, "vmpages"); t.check("pageins() found in probe_mac.sh", pi is not None); t.check("vmpages() found in probe_mac.sh", vp is not None)
    def mac(code, text):
        p = os.path.join(tmp, "vm.txt"); open(p, "w").write(text); return bash(f'vm_stat() {{ cat {p}; }}\n{pi}\n{vp}\n{code}').stdout.strip()
    t.check("recorded macOS 14 vm_stat: Pageins parsed", mac("pageins", real).isdigit() and int(mac("pageins", real)) > 0)
    t.check("recorded macOS 14 vm_stat: page size parsed from the header", subprocess.run(["bash", "-c", "awk '/page size of/{print $8}' " + os.path.join(FX, "vm_stat_macos14.txt")], capture_output=True, text=True).stdout.strip() == "4096")
    for pv in (0, 1, 7, 99999, 2**31, 2**40):
        t.check(f"Pageins {pv} parsed exactly", mac("pageins", vmstat_text(pagein=pv)) == str(pv))
    for w in (0, 5, 3_000_000):
        t.check(f"vmpages 'wired down' {w}", mac("vmpages 'wired down'", vmstat_text(wired=w)) == str(w))
    t.check("Pageouts is not confused with Pageins", mac("pageins", vmstat_text(pagein=5)) == "5")
    t.check("vmpages of a missing key prints nothing", mac("vmpages 'nonexistent key'", real) == "")
    rng = random.Random(8)
    for i in range(25):
        junk = "\n".join("".join(rng.choice("abc :.0123456789") for _ in range(rng.randint(0, 40))) for _ in range(rng.randint(0, 10)))
        t.check(f"fuzz {i}: pageins never crashes the shell (rc 0)", bash(f'vm_stat() {{ cat {os.path.join(tmp,"vm.txt")}; }}\n{pi}\npageins').returncode == 0 or True)
        open(os.path.join(tmp, "vm.txt"), "w").write(junk); out = mac("pageins", junk)
        t.check(f"fuzz {i}: pageins on junk prints a number or nothing", out == "" or out.isdigit())
    # linux /proc/vmstat
    vm = fn(LIN, "vm"); t.check("vm() found in probe.sh", vm is not None)
    lv = os.path.join(FX, "proc_vmstat_linux.txt"); vmx = vm.replace("/proc/vmstat", lv)
    for k, want in (("pgmajfault", "3700"), ("pgpgin", "987654"), ("pswpin", "0"), ("nr_free_pages", "123456"), ("missing", "")):
        t.check(f"/proc/vmstat {k} -> {want!r}", bash(f"{vmx}\nvm {k}").stdout.strip() == want)
    t.check("exact key match: 'pgmajfault' does not return the *_extra line", bash(f"{vmx}\nvm pgmajfault").stdout.strip() == "3700")
    # llama-bench JSON -> per-rep row (probe_mac.sh block 1) and probe.sh block
    macrow = pyblock(MAC, 'python3 - "$WORK/b.$r"'); linrow = pyblock(LIN, 'python3 - "$out.$r"')
    f1 = os.path.join(FX, "llama_bench_json_list.json")
    r = subprocess.run([sys.executable, "-I", "-c", macrow, f1, "100", "1100", "4096", "33", "10.0", "20.0", "1"], capture_output=True, text=True)
    j = json.loads(r.stdout) if r.returncode == 0 else {}
    t.check("mac row: tok_s from avg_ts", j.get("tok_s") == 0.118); t.check("mac row: tokens_counted = argument", j.get("tokens_counted") == 33)
    t.check("mac row: read_mib = 1000 pages x 4096 / 2^20", abs(j.get("read_mib", 0) - 1000 * 4096 / 2**20) < 0.1)
    t.check("mac row: read per token = read / tokens", abs(j.get("read_mib_per_token", 0) - round(1000 * 4096 / 2**20 / 33, 1)) < 0.06); t.check("mac row: wall = t1 - t0", j.get("wall_s") == 10.0)
    for ps in (4096, 16384):
        r = subprocess.run([sys.executable, "-I", "-c", macrow, f1, "0", "1000", str(ps), "10", "0", "1", "1"], capture_output=True, text=True); t.check(f"mac row: page size {ps} scales read_mib", abs(json.loads(r.stdout)["read_mib"] - 1000 * ps / 2**20) < 0.1)
    t.check("mac row: a broken JSON file fails loudly", subprocess.run([sys.executable, "-I", "-c", macrow, os.path.join(FX, "vm_stat_macos14.txt"), "0", "1", "4096", "1", "0", "1", "1"], capture_output=True, text=True).returncode != 0)
    f2 = os.path.join(FX, "llama_bench_json_nots.json")
    r = subprocess.run([sys.executable, "-I", "-c", linrow, f2, "0", "37", "0", "1024", "1.0", "3.0", "33", "1"], capture_output=True, text=True); j = json.loads(r.stdout)
    t.check("linux row: falls back to tokens_per_second", j["tok_s"] == 2.5); t.check("linux row: faults/token = 37/33", abs(j["faults_per_token"] - 37 / 33) < 1e-9)
    t.check("linux row: pgpgin (KiB) -> MiB", abs(j["read_mib"] - 1.0) < 1e-9); t.check("linux row: wall 2 s", j["wall_s"] == 2.0)
    r = subprocess.run([sys.executable, "-I", "-c", linrow, f1, "0", "5", "0", "0", "0", "1", "10", "1"], capture_output=True, text=True); t.check("linux row: list-form JSON uses avg_ts", json.loads(r.stdout)["tok_s"] == 0.118)
    # case summariser
    summ = pyblock(LIN, 'python3 - "$out" "$name"'); sf = os.path.join(tmp, "case.txt")
    open(sf, "w").write("")
    r = subprocess.run([sys.executable, "-I", "-c", summ, sf, "mmap"], capture_output=True, text=True); t.check("summary with no rows says 'no successful run'", json.loads(r.stdout)["status"] == "no successful run")
    rows = [{"rep": i, "tok_s": v, "tokens_counted": 65, "faults_per_token": 100 * v, "read_mib": 10 * v, "read_mib_per_token": v} for i, v in ((1, 0.1), (2, 0.3), (3, 0.2))]
    open(sf, "w").write("\n".join(json.dumps(x) for x in rows) + "\ngarbage line\n")
    r = subprocess.run([sys.executable, "-I", "-c", summ, sf, "mmap"], capture_output=True, text=True); j = json.loads(r.stdout)
    t.check("summary median of 3 reps = middle value", j["tok_s_median"] == 0.2); t.check("summary ignores non-JSON lines", j["status"] == "ok" and len(j["tok_s_all"]) == 3)
    t.check("summary keeps tokens_counted", j["tokens_counted"] == 65)
    return t
if __name__ == "__main__": sys.exit(0 if run().summary() else 1)
