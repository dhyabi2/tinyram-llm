"""moe_run: settings logic (pure functions), command construction, patches present. No model is run."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "tools")); from harness import T
import moe_run as mr
G = 1024 * mr.MIB
TOOLS_DIR = os.path.join(HERE, "..", "tools")
def run():
    t = T("T20 moe_run launcher"); ROOT = os.path.join(HERE, "..")
    for tot, avail, file_gb, exp_gb, want_pool_gb in ((16, 12, 18.6, 17.3, 5.6), (8, 6, 69, 66, 1.9), (32, 28, 69, 66, 25.3), (64, 60, 18.6, 17.3, 17.3)):
        p = mr.plan(int(file_gb * G), int(exp_gb * G), tot * G, avail * G, 0, 4)
        dense = file_gb - exp_gb; reserve = max(1.5, 0.1 * tot); exp = max(256 / 1024, min(avail - dense - reserve, exp_gb))
        t.check(f"RAM {tot} GB avail {avail} GB model {file_gb} GB: pool ~ {exp:.1f} GB", abs(p["pool_mb"] - exp * 1024) < 40, str(p["pool_mb"]))
        t.check(f"RAM {tot} GB: pool never above the expert bytes", p["pool_mb"] * mr.MIB <= int(exp_gb * G) + mr.MIB)
        t.check(f"RAM {tot} GB: pool never below the 256 MiB minimum", p["pool_mb"] >= 256)
    p = mr.plan(10 * G, 9 * G, 8 * G, 1 * G, 0, 4); t.check("almost no free memory: minimum pool and a warning", p["pool_mb"] == 256 and any("little free memory" in n for n in p["notes"]))
    p = mr.plan(10 * G, 9 * G, 16 * G, 12 * G, 3 * G, 4); t.check("swap above 2 GiB gives a warning", any("swap" in n for n in p["notes"]))
    p = mr.plan(10 * G, 0, 16 * G, 12 * G, 0, 4); t.check("a dense model (no expert tensors) does not use the pool", not p["use_pool"] and any("not a MoE" in n for n in p["notes"]))
    t.check("threads equal the physical core count passed in", mr.plan(G, G // 2, 8 * G, 6 * G, 0, 6)["threads"] == 6)
    t.check("io threads: 4 on >= 4 cores, 2 on 2-3 cores, 1 on one core", mr.plan(G, G // 2, 8 * G, 6 * G, 0, 4)["io_threads"] == 4 and mr.plan(G, G // 2, 8 * G, 6 * G, 0, 2)["io_threads"] == 2 and mr.plan(G, G // 2, 8 * G, 6 * G, 0, 1)["io_threads"] == 1)
    t.check("an explicit pool size is honoured", mr.plan(10 * G, 9 * G, 16 * G, 12 * G, 0, 4, pool_mb=1500)["pool_mb"] == 1500)
    t.check("an explicit pool above the free memory warns", any("larger than" in n for n in mr.plan(10 * G, 9 * G, 4 * G, 2 * G, 0, 4, pool_mb=9000)["notes"]))
    p = mr.plan(10 * G, 9 * G, 16 * G, 12 * G, 0, 4)
    for tool, repack in (("completion", "--no-repack"), ("cli", "--no-repack"), ("perplexity", "--no-repack"), ("bench", "--repack")):
        cmd, env = mr.build(tool, "/bin/x", "m.gguf", p, "/tmp/off.txt", ["-p", "hi"])
        t.check(f"{tool}: threads, mmap load mode and repack off, user arguments last", cmd[:2] == ["/bin/x/llama-" + tool, "-m"] and "-t" in cmd and repack in cmd and "mmap" in cmd and cmd[-2:] == ["-p", "hi"])
        t.check(f"{tool}: pool environment set", env["GGML_MOE_POOL"] == "/tmp/off.txt" and env["GGML_MOE_POOL_ASYNC"] == "4" and int(env["GGML_MOE_POOL_MB"]) == p["pool_mb"])
    cmd, env = mr.build("completion", "/bin/x", "m.gguf", p, "/tmp/off.txt", [], plain=True); t.check("--plain leaves the pool variables out", "GGML_MOE_POOL" not in env or env.get("GGML_MOE_POOL") == os.environ.get("GGML_MOE_POOL"))
    os.environ["GGML_MOE_PREFETCH"] = "1"; cmd, env = mr.build("completion", "/bin/x", "m.gguf", p, "/tmp/off.txt", []); t.check("prefetch switch is removed when the pool is on (it would touch the pages the pool replaces)", "GGML_MOE_PREFETCH" not in env); del os.environ["GGML_MOE_PREFETCH"]
    t.check("physical_cores parses macOS sysctl output", mr.physical_cores("Darwin", run=lambda *a, **k: b"4\n") == 4)
    vm = b"Mach Virtual Memory Statistics: (page size of 4096 bytes)\nPages free:                              1000.\nPages inactive:                          2000.\nPages speculative:                       500.\nPages purgeable:                         500.\n"
    outs = {"hw.memsize": b"17179869184\n", "vm.swapusage": b"total = 2048.00M  used = 1024.00M  free = 1024.00M  (encrypted)\n"}
    def fake(cmd, **k): return vm if cmd[0] == "vm_stat" else outs[cmd[-1]]
    total, avail, swap = mr.memory_state("Darwin", run=fake); t.check("memory_state on macOS output: total, available = free+inactive+speculative+purgeable pages, swap", total == 17179869184 and avail == 4000 * 4096 and swap == 1024 * mr.MIB, f"{avail} {swap}")
    # Windows branches (no Windows here: fake the command runner)
    t.check("physical_cores on Windows parses the PowerShell count", mr.physical_cores("Windows", run=lambda *a, **k: b"6\r\n") == 6)
    t.check("physical_cores falls back to half the logical CPUs when the query fails", mr.physical_cores("Windows", run=lambda *a, **k: (_ for _ in ()).throw(OSError("x"))) >= 1)
    pw = mr.plan(10 * G, 9 * G, 16 * G, 12 * G, 0, 4, system_has_no_pool=True); t.check("a platform without the pool plans the stock path and says why", not pw["use_pool"] and any("not available on this platform" in n for n in pw["notes"]))
    import platform as _pf; real = _pf.system
    _pf.system = lambda: "Windows"
    try: cmd, env = mr.build("completion", "C:\\llama\\bin", "m.gguf", p, "off.txt", []); ok = cmd[0].endswith("llama-completion.exe") and env["PATH"].startswith("C:\\llama\\bin")
    finally: _pf.system = real
    t.check("on Windows the executable gets .exe and the bin directory goes on PATH", ok)
    cmd, env = mr.build("bench", "/bin/x", "m.gguf", p, "/tmp/off.txt", [], ngl=99, n_cpu_moe=999); t.check("GPU options: -ngl and -ncmoe are passed through", cmd[cmd.index("-ngl") + 1] == "99" and cmd[cmd.index("-ncmoe") + 1] == "999")
    cmd, env = mr.build("bench", "/bin/x", "m.gguf", p, "/tmp/off.txt", []); t.check("default stays CPU only: -ngl 0 and no -ncmoe", cmd[cmd.index("-ngl") + 1] == "0" and "-ncmoe" not in cmd)
    cmd, env = mr.build("completion", "/bin/x", "m.gguf", p, "/tmp/off.txt", [], cache_bias=1.5, use_experts=6, adapt=True, warm="/tmp/w.txt")
    t.check("new switches reach the environment", env.get("GGML_MOE_CACHE_BIAS") == "1.5" and env.get("GGML_MOE_USE_K") == "6" and env.get("GGML_MOE_POOL_ADAPT") == "1" and env.get("GGML_MOE_POOL_WARM") == "/tmp/w.txt")
    cmd, env = mr.build("completion", "/bin/x", "m.gguf", p, "/tmp/off.txt", [], plain=True, cache_bias=1.5, adapt=True)
    t.check("--plain ignores the pool-only switches (bias, adapt, warm)", "GGML_MOE_CACHE_BIAS" not in env and "GGML_MOE_POOL_ADAPT" not in env)
    cmd, env = mr.build("completion", "/bin/x", "m.gguf", p, "/tmp/off.txt", []); t.check("defaults leave the new switches unset", not any(k in env for k in ("GGML_MOE_CACHE_BIAS", "GGML_MOE_USE_K", "GGML_MOE_POOL_ADAPT", "GGML_MOE_POOL_WARM")) or all(os.environ.get(k) == env.get(k) for k in ("GGML_MOE_CACHE_BIAS", "GGML_MOE_USE_K", "GGML_MOE_POOL_ADAPT", "GGML_MOE_POOL_WARM")))
    t.check("presets: safe changes nothing, fastest sets bias 3 and 6 experts", mr.PRESETS["safe"]["cache_bias"] is None and mr.PRESETS["safe"]["use_experts"] is None and mr.PRESETS["fastest"]["cache_bias"] == 3.0 and mr.PRESETS["fastest"]["use_experts"] == 6)
    t.check("presets are ordered by risk: bias and expert cuts never decrease from safe to fastest", [(mr.PRESETS[k]["cache_bias"] or 0, 8 - (mr.PRESETS[k]["use_experts"] or 8)) for k in ("safe", "balanced", "fast", "fastest")] == sorted([(mr.PRESETS[k]["cache_bias"] or 0, 8 - (mr.PRESETS[k]["use_experts"] or 8)) for k in ("safe", "balanced", "fast", "fastest")]))
    import subprocess as _sp
    q = _sp.run([sys.executable, os.path.join(TOOLS_DIR, "moe_run.py"), "--help"], capture_output=True, text=True); t.check("--help lists the presets", "--preset" in q.stdout and "balanced" in q.stdout)
    for f in ("0001-expert-trace-prefetch-pin-pool.patch", "0002-phase-timing.patch", "0003-routing-experiments.patch", "README.md"): t.check(f"patches/{f} present and non-empty", os.path.getsize(os.path.join(ROOT, "patches", f)) > 200)
    pt = open(os.path.join(ROOT, "patches", "0001-expert-trace-prefetch-pin-pool.patch")).read()
    for k in ("GGML_MOE_POOL", "GGML_MOE_POOL_ASYNC", "GGML_MOE_PREFETCH", "GGML_MOE_TRACE", "GGML_MOE_TRACE_PREFILL"): t.check(f"patch 0001 defines {k}", k in pt)
    t.check("setup script has valid syntax and pins the commit", os.system(f"bash -n {os.path.join(ROOT, 'tools', 'setup_llama.sh')}") == 0 and "eec18f5" in open(os.path.join(ROOT, "tools", "setup_llama.sh")).read())
    return t
if __name__ == "__main__": sys.exit(0 if run().summary() else 1)
