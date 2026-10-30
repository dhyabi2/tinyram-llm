#!/usr/bin/env python3
"""mac_ballast_run: run llama-completion variants under a verified wired-memory ballast on macOS, interleaved, and report decode tok/s per variant.
usage: mac_ballast_run.py --model M.gguf --bin DIR --offsets OFF.txt [--ballast-gb 9] [--tokens 64] [--rounds 3] [--variants spec.json] [--out out.json]
A variants file is a JSON list of {"name":..., "env":{...}, "args":[...]}; default variants compare 8/6/4 experts per token with the overlapped pool (needs a model whose expert-count key is qwen3moe.expert_used_count, override with --kv-key).
The ballast is a mlock'ed anonymous allocation (mac/ballast.c); a round is skipped as invalid when it did not wire >= 95% of the target."""
import argparse, json, os, re, statistics as st, subprocess, sys, tempfile, time
HERE = os.path.dirname(os.path.abspath(__file__))
def wired_pages():
    out = subprocess.check_output(["vm_stat"]).decode(); return int(re.search(r"wired down:\s+(\d+)", out).group(1)), int(re.search(r"page size of (\d+)", out).group(1))
class Ballast:
    def __init__(self, gb): self.gb = gb; self.p = None; self.ok = gb == 0
    def __enter__(self):
        if self.gb == 0: return self
        d = tempfile.mkdtemp(); exe = os.path.join(d, "ballast"); subprocess.check_call(["cc", "-O2", "-o", exe, os.path.join(HERE, "..", "mac", "ballast.c")])
        w0, ps = wired_pages(); n = int(self.gb * 2**30); self.p = subprocess.Popen([exe, str(n)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(60):
            time.sleep(3); w1, _ = wired_pages()
            if (w1 - w0) * ps >= 0.97 * n: self.ok = True; break
        else: self.ok = (wired_pages()[0] - w0) * ps >= 0.95 * n
        return self
    def __exit__(self, *a):
        if self.p: self.p.terminate(); self.p.wait(); time.sleep(10)
def run_one(bin_dir, model, env, args, tokens, prompt):
    e = dict(os.environ, DYLD_LIBRARY_PATH=bin_dir, **env); e.pop("GGML_MOE_PREFETCH", None)
    cmd = [os.path.join(bin_dir, "llama-completion"), "-m", model, "-p", prompt, "-n", str(tokens), "--temp", "0", "-s", "1", "-ngl", "0", "-t", "4", "-lm", "mmap", "--no-repack", "--no-warmup", "-no-cnv"] + args
    t0 = time.time(); r = subprocess.run(cmd, env=e, capture_output=True, text=True, timeout=1800); wall = time.time() - t0
    err = r.stderr; m = re.search(r"^[^\n]*common_perf_print:\s+eval time =\s+([\d.]+) ms /\s+(\d+) runs", err, re.M)
    pm = re.search(r"prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens", err)
    hits = re.findall(r"moe-pool: ops=(\d+) hits=(\d+) misses=(\d+)", err)
    return dict(tok_s=(int(m.group(2)) / (float(m.group(1)) / 1000)) if m else None, runs=int(m.group(2)) if m else None, prompt_ms=float(pm.group(1)) if pm else None, wall_s=round(wall, 1), text_md5=__import__("hashlib").md5(r.stdout.encode()).hexdigest()[:8], pool=hits[-1] if hits else None, rc=r.returncode)
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True); ap.add_argument("--bin", required=True); ap.add_argument("--offsets", required=True); ap.add_argument("--ballast-gb", type=float, default=9.0); ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=3); ap.add_argument("--variants"); ap.add_argument("--kv-key", default="qwen3moe.expert_used_count"); ap.add_argument("--out", default="ballast_run.json")
    ap.add_argument("--prompt", default="Write a short explanation of how a hash table works and why resizing it is amortised constant time.")
    a = ap.parse_args(); pool = {"GGML_MOE_POOL": a.offsets, "GGML_MOE_POOL_MB": "1500", "GGML_MOE_POOL_ASYNC": "2", "GGML_MOE_POOL_LOG": "1"}
    variants = json.load(open(a.variants)) if a.variants else [dict(name=f"pool+overlap, {k} experts", env=pool, args=([] if k == 8 else ["--override-kv", f"{a.kv_key}=int:{k}"])) for k in (8, 6, 4)]
    rows = []
    for rnd in range(a.rounds):
        for v in variants:
            with Ballast(a.ballast_gb) as b:
                if not b.ok: rows.append(dict(variant=v["name"], round=rnd + 1, invalid="ballast did not wire")); print("INVALID ballast", v["name"], file=sys.stderr); continue
                r = run_one(a.bin, a.model, v["env"], v["args"], a.tokens, a.prompt); r.update(variant=v["name"], round=rnd + 1); rows.append(r)
                print(f"round {rnd+1} {v['name']:40} {r['tok_s'] and round(r['tok_s'], 3)} tok/s  text {r['text_md5']}  pool {r['pool']}", file=sys.stderr)
    summ = {}
    for v in variants:
        xs = [r["tok_s"] for r in rows if r.get("variant") == v["name"] and r.get("tok_s")]
        if xs: summ[v["name"]] = dict(median=round(st.median(xs), 3), values=[round(x, 3) for x in xs])
    base = next(iter(summ.values()))["median"] if summ else None
    print("\n| variant | median tok/s | runs | vs first |\n|---|---|---|---|")
    for k, s in summ.items(): print(f"| {k} | {s['median']} | {s['values']} | {s['median']/base:.2f}x |")
    json.dump(dict(rows=rows, summary=summ, ballast_gb=a.ballast_gb, tokens=a.tokens), open(a.out, "w"), indent=1)
if __name__ == "__main__": main()
