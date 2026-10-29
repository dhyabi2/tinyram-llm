"""T17 repo hygiene (leak scan, license, links, syntax) and T18 data integrity (hash-pinned fixtures, traces, result files)."""
import ast, glob, hashlib, json, os, re, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "tools")); from harness import T
import results_check as rc
ROOT = os.path.abspath(os.path.join(HERE, ".."))
# patterns are assembled from pieces so this file does not trip its own scan
BAD = {"internal vault name": "Internal" + "_not_public", "private project name": "divide" + "-and-" + "concur", "private project camel": "divide" + "AndConcur", "private tool dir": "methodology" + "-tree", "VPS address": "172." + "86.85.83",
       "home path": "/Users/" + "mac/", "root login": "root" + "@", "private key block": "BEGIN " + "PRIVATE KEY", "OpenAI-style key": r"sk-[A-Za-z0-9]{20,}", "bearer token": r"Bearer [A-Za-z0-9._-]{20,}",
       "key assignment": r"(?i)(api[_-]?key|secret|token|password)\s*[=:]\s*['\"][A-Za-z0-9_\-]{16,}['\"]", "nano key": "nano" + "gpt", "x-api-key": "x-api-" + "key"}
def scan(files):
    hits = []
    for f in files:
        try: s = open(f, errors="ignore").read()
        except Exception: continue
        for name, pat in BAD.items():
            if (re.search(pat, s) if any(c in pat for c in "[]\\(") else pat in s): hits.append((f, name))
    return hits
def tracked(): return [os.path.join(ROOT, f) for f in subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split()]
def run():
    t = T("T17/T18 hygiene + data"); files = tracked()
    t.check("tracked file list is non-trivial", len(files) >= 40, str(len(files)))
    # scanner control: it must catch planted leaks
    d = tempfile.mkdtemp()
    for name, planted in (("vault", BAD["internal vault name"]), ("ip", BAD["VPS address"]), ("key", "sk-" + "a" * 30), ("assign", "api_key = '" + "x" * 20 + "'"), ("home", "/Users/" + "mac/x"), ("pk", "BEGIN " + "PRIVATE KEY")):
        p = os.path.join(d, name); open(p, "w").write("hello " + planted + " world"); t.check(f"control: scanner catches a planted {name}", len(scan([p])) >= 1)
    p = os.path.join(d, "clean"); open(p, "w").write("tokens per second and expert cache sizes"); t.check("control: scanner passes clean text", scan([p]) == [])
    texts = [f for f in files if not f.endswith((".gguf", ".png"))]
    self_ = os.path.abspath(__file__)
    for name, pat in BAD.items():
        h = [f for f, n in scan([f for f in texts if os.path.abspath(f) != self_]) if n == name]
        t.check(f"no tracked file contains: {name}", not h, str(h[:3]))
    t.check("LICENSE exists and is non-empty", os.path.exists(os.path.join(ROOT, "LICENSE")) and os.path.getsize(os.path.join(ROOT, "LICENSE")) > 500)
    t.check("LICENSE keeps the copyright line", re.search(r"(?i)copyright", open(os.path.join(ROOT, "LICENSE")).read()) is not None)
    t.check("README exists", os.path.exists(os.path.join(ROOT, "README.md")))
    rd = open(os.path.join(ROOT, "README.md")).read()
    for m in re.finditer(r"\]\(([^)#\s]+)\)", rd):
        l = m.group(1)
        if not l.startswith(("http", "mailto")): t.check(f"README link resolves: {l}", os.path.exists(os.path.join(ROOT, l)))
    for f in files:
        n = os.path.relpath(f, ROOT)
        if f.endswith(".py"):
            try: ast.parse(open(f).read()); ok = True
            except SyntaxError: ok = False
            t.check(f"{n}: python syntax", ok)
        if f.endswith(".json"):
            try: json.load(open(f)); ok = True
            except Exception: ok = False
            t.check(f"{n}: valid JSON", ok)
        if f.endswith((".py", ".sh", ".md", ".c", ".h")): t.check(f"{n}: ends with newline, no tabs-only lines, no merge markers", open(f, errors="ignore").read().endswith("\n") and ("<" * 7) not in open(f, errors="ignore").read())
    t.check("no .gguf or large binary is tracked", not [f for f in files if f.endswith(".gguf") or os.path.getsize(f) > 20 * 2**20])
    t.check("no tracked file over 5 MB", not [f for f in files if os.path.getsize(f) > 5 * 2**20])
    # T18 data
    man = open(os.path.join(ROOT, "data", "MANIFEST.sha256")).read().splitlines(); t.check("manifest lists the traces, results and fixtures", len(man) >= 10, str(len(man)))
    for l in man:
        h, f = l.split("  ", 1); p = os.path.join(ROOT, f); t.check(f"hash pinned: {f}", os.path.exists(p) and hashlib.sha256(open(p, "rb").read()).hexdigest() == h)
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "traces", "*.txt"))):
        n = os.path.basename(f); L = [l.split() for l in open(f) if l.startswith("blk.")]
        t.check(f"{n}: every trace line has 8 expert ids", all(len(x) == 9 for x in L), str({len(x) for x in L}))
        t.check(f"{n}: ids within 0..127 and distinct per line", all(0 <= int(i) < 128 for x in L for i in x[1:]) and all(len(set(x[1:])) == 8 for x in L))
        t.check(f"{n}: layer names are blk.0..47 with the same tensor", all(re.fullmatch(r"blk\.(\d|[1-3]\d|4[0-7])\.ffn_down_exps\.weight", x[0]) for x in L))
        while L and not L[0][0].startswith("blk.0."): L = L[1:]
        t.check(f"{n}: after alignment the line count is a whole number of tokens (48 layers)", len(L) // 48 >= 200, str(len(L)))
        seq = [int(x[0].split(".")[1]) for x in L[: len(L) // 48 * 48]]
        t.check(f"{n}: layers cycle 0..47 in order inside whole tokens", all(seq[i] == i % 48 for i in range(len(seq))))
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "*", "results.json"))):
        n = os.path.relpath(f, ROOT); j = json.load(open(f))
        t.check(f"{n}: valid JSON object", isinstance(j, dict))
        t.check(f"{n}: has machine and model", "machine" in j and ("model" in j or "model_gb" in j or "tool" in j))
        for k in ("tok_s_median", "read_mib_per_token_median"):
            vals = []
            def walk(o):
                if isinstance(o, dict):
                    for kk, v in o.items():
                        if kk == k: vals.append(v)
                        walk(v)
                elif isinstance(o, list):
                    for v in o: walk(v)
            walk(j)
            t.check(f"{n}: every {k} is a finite non-negative number", all(isinstance(v, (int, float)) and v >= 0 for v in vals))
    return t
if __name__ == "__main__": sys.exit(0 if run().summary() else 1)
