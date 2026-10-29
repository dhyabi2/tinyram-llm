#!/usr/bin/env python3
"""compare_run: measure plain mmap against the explicit expert read pool (and the overlapped pool) on THIS machine, interleaved, and print a table + write a JSON result.
usage: compare_run.py --model MODEL.gguf [--bin DIR] [--tokens 32] [--reps 3] [--out result.json] [--pool-mb auto|N] [--no-overlap]
Each round runs: plain mmap, pool, pool with 2 overlapping I/O threads (llama-bench, decode only, 4 threads = physical cores). Order is the same in every round, rounds alternate variants so drift hits all of them.
It records free memory, swap and how much of the model file is already in the page cache before each run, and marks a round NOT CLEAN when swap is above 2 GiB or more than 50% of the file is cached
(then the plain-mmap number only measures the cache, not the disk). For a machine whose RAM is bigger than the model, shrink the memory first (see docs/RUN_WITH_CLAUDE.md) or use a bigger model."""
import argparse, json, os, platform, statistics as st, subprocess, sys, tempfile, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import moe_run as mr, gguf_offsets as go

def one(variant, cmd, env, timeout):
    with tempfile.NamedTemporaryFile("w+", suffix=".json") as f:
        t0 = time.time(); r = subprocess.run(cmd + ["-o", "json"], env=env, stdout=f, stderr=subprocess.PIPE, timeout=timeout); wall = time.time() - t0
        if r.returncode != 0: return dict(variant=variant, error=r.stderr.decode(errors="replace")[-300:])
        f.seek(0); d = json.load(f)
    return dict(variant=variant, tok_s=d[-1]["avg_ts"], wall_s=round(wall, 1), pool_log=[l for l in r.stderr.decode(errors="replace").splitlines() if l.startswith("moe-pool: ops")][-1:])

def residency(model):
    try:
        import mincore_probe as mp; return round(mp.resident(model)["fraction"], 3)
    except Exception: return None

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True); ap.add_argument("--bin", default=os.environ.get("LLAMA_BIN", "./llama.cpp/build/bin")); ap.add_argument("--tokens", type=int, default=32); ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="compare_result.json"); ap.add_argument("--pool-mb", default="auto"); ap.add_argument("--no-overlap", action="store_true"); ap.add_argument("--timeout", type=int, default=1800); ap.add_argument("--ngl", type=int, default=0); ap.add_argument("--n-cpu-moe", type=int)
    a = ap.parse_args(argv)
    tot, avail, swap = mr.memory_state(); cores = mr.physical_cores(); eb, ne = go.expert_total(a.model); size = os.path.getsize(a.model)
    p = mr.plan(size, eb, tot, avail, swap, cores, a.pool_mb, system_has_no_pool=(platform.system() == "Windows" and not os.environ.get("TINYRAM_POOL_ON_WINDOWS")))
    if not p["use_pool"]: sys.exit("not a MoE model (no expert tensors): nothing to compare")
    off = mr.offsets_for(a.model); bench = ["-p", "0", "-n", str(a.tokens), "-r", "1"]
    variants = [("plain mmap", True, None)]
    variants.append(("pool", False, None))
    if not a.no_overlap: variants.append(("pool + overlap", False, p["io_threads"]))
    rows = []; clean = True
    print(f"machine: {tot//mr.MIB} MiB RAM, {cores} cores, model {size//mr.MIB} MiB (experts {eb//mr.MIB} MiB), pool {p['pool_mb']} MiB", file=sys.stderr)
    for n in p["notes"]: print("note:", n, file=sys.stderr)
    for rep in range(a.reps):
        for name, plain, io in variants:
            _, avail, swap = mr.memory_state(); res = residency(a.model)
            cmd, env = mr.build("bench", a.bin, a.model, p, off, bench, plain=plain, ngl=a.ngl, n_cpu_moe=a.n_cpu_moe)
            if io is None and not plain: env.pop("GGML_MOE_POOL_ASYNC", None)
            r = one(name, cmd, env, a.timeout); r.update(rep=rep + 1, avail_mib=avail // mr.MIB, swap_mib=swap // mr.MIB, file_cached_fraction=res)
            ok = swap <= 2 * 1024 * mr.MIB and (res is None or res <= 0.5); r["clean"] = ok; clean &= ok; rows.append(r)
            print(f"round {rep+1} {name:15} " + (f"{r['tok_s']:.3f} tok/s" if "tok_s" in r else "ERROR " + r["error"][-120:]) + ("" if ok else "   NOT CLEAN"), file=sys.stderr)
    summ = {}
    for name, _, _ in variants:
        v = [r["tok_s"] for r in rows if r["variant"] == name and "tok_s" in r]
        if v: summ[name] = dict(median=round(st.median(v), 3), values=[round(x, 3) for x in v])
    base = summ.get("plain mmap", {}).get("median")
    print("\n| variant | median tok/s | all runs | vs plain mmap |\n|---|---|---|---|")
    for name, s in summ.items(): print(f"| {name} | {s['median']} | {s['values']} | {('%.2fx' % (s['median']/base)) if base else '-'} |")
    if not clean: print("\nWARNING: at least one round was not clean (swap in use or the model mostly cached); the plain-mmap number is then too high or noisy. Close other apps, or shrink memory, and rerun.")
    json.dump(dict(machine=dict(system=platform.system(), machine=platform.machine(), ram_mib=tot // mr.MIB, cores=cores, ngl=a.ngl, n_cpu_moe=a.n_cpu_moe), model=os.path.basename(a.model), model_mib=size // mr.MIB, expert_mib=eb // mr.MIB, pool_mb=p["pool_mb"],
                   tokens=a.tokens, rows=rows, summary=summ, clean=clean), open(a.out, "w"), indent=1); print(f"\nwrote {a.out}", file=sys.stderr)
    return 0
if __name__ == "__main__": sys.exit(main())
