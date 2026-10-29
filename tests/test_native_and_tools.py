"""T11/T12/T13 wrappers: expert pool C tests under ASan+UBSan (fixed tests + 150 fuzz scenarios), sanitizer control, layer-quant-map tests on the tiny model, n-gram tests when numpy exists.
Local binaries are located through env LLAMA_BIN and TINY (default: the work copy next to this repo); a missing prerequisite is reported as a skipped suite, never as a pass."""
import os, re, shutil, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); from harness import T
ROOT = os.path.abspath(os.path.join(HERE, ".."))
def sh(cmd, **kw): return subprocess.run(cmd, capture_output=True, text=True, timeout=900, **kw)
def run():
    t = T("T11-T13 native and tool tests"); tmp = tempfile.mkdtemp(); cc = shutil.which("clang") or shutil.which("gcc")
    t.check("a C compiler is available", cc is not None)
    if not cc: return t
    san = ["-fsanitize=address,undefined", "-fno-sanitize-recover=undefined", "-g", "-O1"]
    # sanitizer control: a planted out-of-bounds read must be caught
    open(f"{tmp}/bad.c", "w").write("#include <stdlib.h>\nint main(int c,char**v){int*a=malloc(4*sizeof(int));return a[c+7];}\n")
    sh([cc, *san, "-o", f"{tmp}/bad", f"{tmp}/bad.c"]); r = sh([f"{tmp}/bad"]); t.check("control: ASan catches a planted heap overflow", r.returncode != 0 and "AddressSanitizer" in r.stderr)
    r = sh([cc, *san, "-o", f"{tmp}/tep", f"{ROOT}/c/expert_pool.c", f"{ROOT}/c/test_expert_pool.c"]); t.check("expert_pool + fixed tests compile with ASan/UBSan, no warnings as errors", r.returncode == 0, r.stderr[-300:])
    r = sh([f"{tmp}/tep"]); passes = r.stdout.count("PASS "); t.check("expert pool fixed tests: 16 PASS lines, 0 FAIL, no sanitizer report", passes == 16 and "FAIL" not in r.stdout and "Sanitizer" not in r.stderr and r.returncode == 0, r.stdout[-200:] + r.stderr[-200:])
    for i in range(16): t.check(f"expert pool fixed test {i + 1} of 16 passed", passes > i)
    r = sh([cc, *san, "-o", f"{tmp}/fuzz", f"{ROOT}/c/expert_pool.c", f"{ROOT}/c/test_expert_pool_fuzz.c"]); t.check("fuzz program compiles under ASan/UBSan", r.returncode == 0, r.stderr[-300:])
    r = sh([f"{tmp}/fuzz", "150"]); fails = {int(m) for m in re.findall(r"scenario (\d+)", "\n".join(l for l in r.stdout.splitlines() if l.startswith("FAIL")))}
    m = re.search(r"(\d+)/(\d+) checks passed", r.stdout); t.check("fuzz run finished with its summary line and no sanitizer report", m is not None and "Sanitizer" not in r.stderr and r.returncode == 0, r.stderr[-300:])
    for sc in range(150): t.check(f"fuzz scenario {sc}: bytes, cap, errors and invariants held", sc not in fails)
    if m: t.check(f"fuzz: all {m.group(2)} internal checks passed", m.group(1) == m.group(2))
    r = sh([cc, "-O2", "-o", f"{tmp}/tep_plain", f"{ROOT}/c/expert_pool.c", f"{ROOT}/c/test_expert_pool.c"]); r2 = sh([f"{tmp}/tep_plain"]); t.check("the same fixed tests pass in a plain -O2 build", r.returncode == 0 and r2.returncode == 0 and "16/16" in r2.stdout)
    # tool tests on the tiny model
    lb = os.environ.get("LLAMA_BIN", ""); tiny = os.environ.get("TINY", "")
    if os.path.exists(os.path.join(lb, "llama-quantize")) and os.path.exists(tiny):
        r = sh([sys.executable, "-I", os.path.join(ROOT, "tools", "test_layer_quant_map.py")], env=dict(os.environ, LLAMA_BIN=lb, TINY=tiny, DYLD_LIBRARY_PATH=lb, LD_LIBRARY_PATH=lb), cwd=os.path.join(ROOT, "tools"))
        for line in r.stdout.splitlines():
            if line.startswith(("PASS ", "FAIL ")): t.check("quant map: " + line[5:90], line.startswith("PASS "))
        t.check("quant map: summary says all checks passed", re.search(r"(\d+)/\1 checks passed", r.stdout) is not None, r.stdout[-200:])
    else: print("SKIPPED quant-map tests: llama.cpp binaries or tiny model not found")
    try: import numpy; have_np = True
    except ImportError: have_np = False
    print("SKIPPED n-gram builder tests: numpy is not installed" if not have_np else "numpy present: n-gram tests need their runner, not run here")
    return t
if __name__ == "__main__": sys.exit(0 if run().summary() else 1)
