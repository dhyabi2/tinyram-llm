#!/usr/bin/env python3
"""moe_run: run any MoE GGUF with the explicit expert read pool, with settings chosen for THIS machine.
usage: moe_run.py --model MODEL.gguf [--bin DIR] [--tool completion|cli|bench|perplexity] [--pool-mb N|auto] [--io-threads N] [--threads N] [--dry-run] [--plain] -- <arguments for the llama tool>
  --bin       directory with the patched llama.cpp binaries (default $LLAMA_BIN or ./llama.cpp/build/bin)
  --pool-mb   cache for expert weights in MiB (default auto = free RAM minus dense weights minus a reserve)
  --plain     do not use the pool (stock mmap path) - for comparisons
What it does: reads the GGUF header, finds the expert tensors, writes/caches the offsets file (~/.cache/tinyram), picks threads = physical cores (never hyperthreads: 8 threads on 4 cores was 11x slower),
sizes the pool from free RAM, turns on 4 overlapping I/O threads (blocking reads that do not compete for cores), disables weight repacking (the pool needs the mmap path), warns when swap is already in use, then runs the tool."""
import argparse, hashlib, os, platform, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); import gguf_offsets as go
MIB = 1 << 20

def physical_cores(system=None, run=subprocess.check_output):
    system = system or platform.system()
    try:
        if system == "Darwin": return int(run(["sysctl", "-n", "hw.physicalcpu"]).strip())
        if system == "Windows":
            out = run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum"]).decode().strip()
            if out.isdigit(): return int(out)
        if system == "Linux":
            txt = open("/proc/cpuinfo").read(); pairs = set(re.findall(r"physical id\s*:\s*(\d+)\s*\n(?:.*\n)*?core id\s*:\s*(\d+)", txt)); n = len(pairs)
            if n: return n
            return max(1, (os.cpu_count() or 2) // 2)
    except Exception: pass
    return max(1, (os.cpu_count() or 2) // 2)

def memory_state(system=None, run=subprocess.check_output):
    """(total_bytes, available_bytes, swap_used_bytes). available = what a cache can use without pushing the machine into swap."""
    system = system or platform.system()
    if system == "Darwin":
        total = int(run(["sysctl", "-n", "hw.memsize"]).strip()); vm = run(["vm_stat"]).decode(); ps = int(re.search(r"page size of (\d+) bytes", vm).group(1))
        g = lambda k: int((re.search(k + r"[^\d]*(\d+)\.", vm) or [0, 0])[1]) if re.search(k + r"[^\d]*(\d+)\.", vm) else 0
        avail = (g("Pages free") + g("Pages inactive") + g("Pages speculative") + g("Pages purgeable")) * ps
        sw = re.search(r"used = ([\d.]+)([MG])", run(["sysctl", "-n", "vm.swapusage"]).decode()); swap = int(float(sw.group(1)) * (MIB if sw.group(2) == "M" else 1024 * MIB)) if sw else 0
        return total, avail, swap
    if system == "Windows":
        import ctypes
        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS(); m.dwLength = ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        swap = max(0, (m.ullTotalPageFile - m.ullAvailPageFile) - (m.ullTotalPhys - m.ullAvailPhys))   # committed memory beyond physical RAM: an estimate of what lives in the page file
        return m.ullTotalPhys, m.ullAvailPhys, swap
    info = {k: int(v) * 1024 for k, v in re.findall(r"^(\w+):\s+(\d+) kB", open("/proc/meminfo").read(), re.M)}
    return info["MemTotal"], info.get("MemAvailable", info["MemFree"]), info.get("SwapTotal", 0) - info.get("SwapFree", 0)

def plan(file_bytes, expert_bytes, total_ram, avail_ram, swap_used, cores, pool_mb=None, io_threads=None, system_has_no_pool=False):
    """Pure function: the settings for a machine. Returns a dict (threads, io_threads, pool_mb, notes[])."""
    notes = []; dense = max(0, file_bytes - expert_bytes)
    reserve = max(int(1.5 * 1024 * MIB), int(0.10 * total_ram))      # OS, KV cache, compute buffers
    budget = avail_ram - dense - reserve
    auto = max(256 * MIB, min(budget, expert_bytes)) if expert_bytes else 0
    if budget < 256 * MIB: notes.append(f"little free memory: about {max(budget,0)//MIB} MiB left for the expert cache after the dense weights ({dense//MIB} MiB) and a {reserve//MIB} MiB reserve; the pool is set to the 256 MiB minimum and most experts will be read from disk")
    if swap_used > 2 * 1024 * MIB: notes.append(f"swap already holds {swap_used/1024/MIB:.1f} GiB: close memory-heavy apps first, results and speed suffer under memory pressure")
    if expert_bytes == 0: notes.append("no expert tensors found: this is not a MoE model (or uses an unsupported layout); running without the pool")
    pm = int(pool_mb) if pool_mb not in (None, "auto") else int(auto // MIB)
    if pool_mb not in (None, "auto") and pm * MIB > avail_ram: notes.append(f"the requested pool ({pm} MiB) is larger than the memory that is currently free ({avail_ram//MIB} MiB)")
    if system_has_no_pool:
        notes.append("the read pool is not available on this platform yet (see docs/WINDOWS_PORT_PROMPT.md): running the stock mmap path with tuned threads"); expert_bytes_for_pool = 0
    else: expert_bytes_for_pool = expert_bytes
    return dict(threads=cores, io_threads=(io_threads if io_threads is not None else (4 if cores >= 4 else 2 if cores >= 2 else 1)), pool_mb=pm, use_pool=expert_bytes_for_pool > 0, notes=notes, dense_mb=dense // MIB, expert_mb=expert_bytes // MIB)

# Presets from the measurements (Qwen3-30B-A3B and OLMoE on a 16 GB Mac, memory squeezed 8x). Quality = perplexity change on documentation / code text.
PRESETS = {
    "safe":     dict(cache_bias=None, use_experts=None, note="pool + overlap only: output identical to the stock path (about 4-5x over plain mmap when the model is far larger than RAM)"),
    "balanced": dict(cache_bias=1.0,  use_experts=None, note="+ routing prefers cached experts: about 1.4x more, perplexity +0.07%/+1% (Qwen3-30B docs / OLMoE code)"),
    "fast":     dict(cache_bias=1.0,  use_experts=6,    note="+ 6 of 8 experts: about 2x over safe, perplexity +3.5% docs / +5.3% code (Qwen3-30B)"),
    "fastest":  dict(cache_bias=3.0,  use_experts=6,    note="bias 3 + 6 experts: about 2.6x over safe, perplexity +2% docs; bias 3 alone cost +17% on OLMoE and +4.6% on Qwen3 code - check your model first"),
}

def offsets_for(model):
    st = os.stat(model); key = hashlib.sha1(f"{os.path.abspath(model)}:{st.st_size}:{int(st.st_mtime)}".encode()).hexdigest()[:16]
    d = os.path.join(os.path.expanduser("~"), ".cache", "tinyram"); os.makedirs(d, exist_ok=True); path = os.path.join(d, f"offsets-{key}.txt")
    if not os.path.exists(path): go.main(model, path)
    return path

def build(tool, bin_dir, model, p, offsets, extra, plain=False, ngl=0, n_cpu_moe=None, cache_bias=None, use_experts=None, adapt=False, warm=None):
    exe = os.path.join(bin_dir, "llama-" + tool + (".exe" if platform.system() == "Windows" else "")); env = dict(os.environ)
    if bin_dir and platform.system() == "Windows": env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    elif bin_dir: env["DYLD_LIBRARY_PATH"] = bin_dir + os.pathsep + env.get("DYLD_LIBRARY_PATH", ""); env["LD_LIBRARY_PATH"] = bin_dir + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    pool = p["use_pool"] and not plain
    if pool: env.update(GGML_MOE_POOL=offsets, GGML_MOE_POOL_MB=str(p["pool_mb"]), GGML_MOE_POOL_ASYNC=str(p["io_threads"]), GGML_MOE_POOL_LOG="1")
    if pool and adapt: env["GGML_MOE_POOL_ADAPT"] = "1"           # shrink the pool when the machine runs short of memory, grow back later
    if pool and warm: env["GGML_MOE_POOL_WARM"] = warm              # save the hot experts at exit and preload them at the next start (no gain across different prompts)
    if pool and cache_bias: env["GGML_MOE_CACHE_BIAS"] = str(cache_bias)   # routing prefers experts already in the pool: fewer disk reads, small quality cost (see README)
    if use_experts: env["GGML_MOE_USE_K"] = str(use_experts)         # compute only the first K of the selected experts (weights stay normalised over all of them)
    for k in ("GGML_MOE_PREFETCH", "GGML_MOE_PIN"): env.pop(k, None) if pool else None   # prefetch/pin touch the mmap pages the pool replaces
    argv = [exe, "-m", model, "-t", str(p["threads"]), "-ngl", str(ngl)]
    if n_cpu_moe is not None: argv += ["-ncmoe", str(n_cpu_moe)]   # keep the expert tensors of the first N layers on the CPU (the pool serves them); the rest of the model can sit on the GPU
    argv += ["-lm", "mmap"]
    argv += (["--repack", "0"] if tool in ("bench",) else ["--no-repack"])
    return argv + list(extra), env

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True); ap.add_argument("--bin", default=os.environ.get("LLAMA_BIN", "./llama.cpp/build/bin")); ap.add_argument("--tool", default="completion", choices=["completion", "cli", "bench", "perplexity"])
    ap.add_argument("--pool-mb", default="auto"); ap.add_argument("--io-threads", type=int); ap.add_argument("--threads", type=int); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--plain", action="store_true"); ap.add_argument("--cache-bias", type=float, help="routing bias toward experts resident in the pool, e.g. 1.0 (conservative) or 3.0 (faster, up to ~5%% perplexity on code)"); ap.add_argument("--use-experts", type=int, help="compute only K of the selected experts per token (anchored; K=6 of 8 costs ~5%% perplexity)"); ap.add_argument("--preset", choices=sorted(PRESETS), help="speed/quality preset: " + "; ".join(f"{k}: {v['note']}" for k, v in PRESETS.items()).replace("%", "%%")); ap.add_argument("--adapt", action="store_true", help="adaptive pool size under memory pressure"); ap.add_argument("--warm", help="file for warm start of the pool"); ap.add_argument("--ngl", type=int, default=0, help="layers to offload to the GPU (GPU builds only)"); ap.add_argument("--n-cpu-moe", type=int, help="keep the experts of the first N layers on the CPU (use 999 for all) so the GPU holds attention/dense weights and the CPU read pool serves the experts")
    ap.add_argument("rest", nargs=argparse.REMAINDER); a = ap.parse_args(argv); extra = a.rest[1:] if a.rest[:1] == ["--"] else a.rest
    if not os.path.isfile(a.model): sys.exit(f"model not found: {a.model}")
    tot, avail, swap = memory_state(); cores = a.threads or physical_cores()
    expert_bytes, n_exp = go.expert_total(a.model); p = plan(os.path.getsize(a.model), expert_bytes, tot, avail, swap, cores, a.pool_mb, a.io_threads, system_has_no_pool=(platform.system() == "Windows" and not os.environ.get("TINYRAM_POOL_ON_WINDOWS")))
    print(f"machine: {tot//MIB} MiB RAM, {avail//MIB} MiB available, swap used {swap//MIB} MiB, {cores} physical cores", file=sys.stderr)
    print(f"model: {os.path.getsize(a.model)//MIB} MiB, experts {p['expert_mb']} MiB ({n_exp} per layer), dense {p['dense_mb']} MiB", file=sys.stderr)
    for n in p["notes"]: print("note:", n, file=sys.stderr)
    if a.preset:
        pr = PRESETS[a.preset]; a.cache_bias = a.cache_bias if a.cache_bias is not None else pr["cache_bias"]; a.use_experts = a.use_experts or pr["use_experts"]; print(f"preset {a.preset}: {pr['note']}", file=sys.stderr)
    off = offsets_for(a.model) if p["use_pool"] and not a.plain else ""
    cmd, env = build(a.tool, a.bin, a.model, p, off, extra, a.plain, a.ngl, a.n_cpu_moe, a.cache_bias, a.use_experts, a.adapt, a.warm)
    print(f"settings: threads={p['threads']} pool={'off' if a.plain or not p['use_pool'] else str(p['pool_mb']) + ' MiB, ' + str(p['io_threads']) + ' io threads'}", file=sys.stderr)
    if a.dry_run: print(" ".join(cmd)); return 0
    if not os.path.exists(cmd[0]): sys.exit(f"missing {cmd[0]}: run tools/setup_llama.sh first (or pass --bin)")
    return subprocess.call(cmd, env=env)
if __name__ == "__main__": sys.exit(main())
