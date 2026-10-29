"""T09 shell scripts + T10 preflight against a local mock server (no network)."""
import glob, http.server, json, os, shutil, stat, subprocess, sys, tempfile, threading
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); from harness import T
ROOT = os.path.abspath(os.path.join(HERE, ".."))
class H(http.server.BaseHTTPRequestHandler):
    mode = "ok"; size = 5000
    def log_message(self, *a): pass
    def do_HEAD(self):
        m, rng = H.mode, self.headers.get("Range")
        if m == "redirect" and self.path == "/model": self.send_response(302); self.send_header("Location", "/real"); self.end_headers(); return
        if rng and m != "no206": self.send_response(206); self.send_header("Content-Range", f"bytes 0-0/{H.size}"); self.send_header("Content-Length", "1")
        else: self.send_response(200); self.send_header("Content-Length", str(H.size if m != "wrongsize" else H.size + 7))
        self.end_headers()
    do_GET = do_HEAD
def serve():
    s = http.server.HTTPServer(("127.0.0.1", 0), H); threading.Thread(target=s.serve_forever, daemon=True).start(); return s
def fake_bench(d, flags):
    p = os.path.join(d, "bench"); open(p, "w").write("#!/bin/sh\necho 'usage: llama-bench " + " ".join(flags) + "'\n"); os.chmod(p, 0o755); return p
def pre(url, want, d, bench):
    r = subprocess.run(["bash", os.path.join(ROOT, "tools", "preflight.sh"), url, str(want), d, bench], capture_output=True, text=True, timeout=120)
    chk = {}
    for l in r.stdout.splitlines():
        if l.startswith("{") and '"check"' in l: j = json.loads(l); chk[j["check"]] = j["ok"]
    return r.returncode, chk, r.stdout
def run():
    t = T("T09/T10 shell + preflight"); tmp = tempfile.mkdtemp()
    scripts = glob.glob(os.path.join(ROOT, "**", "*.sh"), recursive=True); t.check("scripts found", len(scripts) >= 4, str(len(scripts)))
    sc = shutil.which("shellcheck")
    for s in scripts:
        n = os.path.relpath(s, ROOT); src = open(s).read(); code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
        t.check(f"{n}: bash -n passes", subprocess.run(["bash", "-n", s], capture_output=True).returncode == 0)
        t.check(f"{n}: has a bash shebang", src.startswith("#!/usr/bin/env bash") or src.startswith("#!/bin/bash"))
        t.check(f"{n}: no CRLF line endings", "\r" not in src)
        t.check(f"{n}: no 'rm -rf /' or 'rm -rf $HOME' or sudo/purge call", not any(x in code for x in ("rm -rf /\n", "rm -rf $HOME", "rm -rf ~", "sudo ")))
        t.check(f"{n}: no hard-coded credential-looking assignment", not any(k in src.lower() for k in ("password=", "api_key=", "apikey=", "secret=")))
        if sc: t.check(f"{n}: shellcheck has no errors", subprocess.run([sc, "-S", "error", s], capture_output=True).returncode == 0)
    for s in ("probe/probe.sh", "probe/probe_mac.sh"):
        r = subprocess.run(["bash", os.path.join(ROOT, s), "--bogus-option"], capture_output=True, text=True, timeout=30); t.check(f"{s}: unknown option -> exit 2", r.returncode == 2 and "unknown option" in r.stderr)
    r = subprocess.run(["bash", os.path.join(ROOT, "probe/probe_mac.sh")], capture_output=True, text=True, timeout=30); t.check("probe_mac.sh without --bench/--model -> exit 2", r.returncode == 2)
    r = subprocess.run(["bash", os.path.join(ROOT, "probe/probe_mac.sh"), "--bench", "/nonexistent", "--model", "/nonexistent"], capture_output=True, text=True, timeout=30); t.check("probe_mac.sh with a missing binary -> exit 2", r.returncode == 2)
    r = subprocess.run(["bash", os.path.join(ROOT, "probe/probe.sh"), "--dry-run", "--workdir", os.path.join(tmp, "w"), "--model-gb", "1"], capture_output=True, text=True, timeout=60)
    t.check("probe.sh --dry-run stops before any download", r.returncode == 0 and "dry run: stopping here" in r.stdout, r.stdout[-200:])
    t.check("probe.sh --dry-run created nothing but the work dir files", sorted(os.listdir(os.path.join(tmp, "w"))) == ["probe.log"])
    # preflight vs mock
    srv = serve(); port = srv.server_address[1]; good = fake_bench(tmp, ["-lm", "--repack", "-ncmoe", "dio"]); url = f"http://127.0.0.1:{port}/real"
    H.size = 5000
    for mode, key, want_ok in (("ok", "range_support", True), ("no206", "range_support", False), ("ok", "remote_size", True), ("wrongsize", "remote_size", False), ("redirect", "range_support", True), ("redirect", "remote_size", True)):
        H.mode = mode; rc, chk, out = pre(url if mode != "redirect" else f"http://127.0.0.1:{port}/model", 5000, os.path.join(tmp, "dl"), good)
        t.check(f"mock {mode}: {key} -> {want_ok}", chk.get(key) is want_ok, out[-300:])
        t.check(f"mock {mode}: exit code is {'0' if all(chk.values()) else '11'} and matches the failed checks", rc == (0 if all(chk.values()) else 11), f"rc={rc}")
    H.mode = "ok"; rc, chk, out = pre(url, 5000, os.path.join(tmp, "dl"), good)
    for k in ("os_arch", "cpu_flags", "ram", "size_parse", "disk_headroom", "write_test", "download_tool", "range_support", "remote_size", "bench_binary", "flag-lm", "flag--repack", "flag-ncmoe", "lm_dio"):
        t.check(f"clean mock run: {k} ok", chk.get(k) is True, k)
    t.check("clean mock run exits 0 with '0 failed checks'", rc == 0 and "0 failed checks" in out)
    t.check("the preflight write test leaves no file behind", not os.path.exists(os.path.join(tmp, "dl", ".preflight_write")))
    nb = fake_bench(tmp, ["-lm", "--repack"]); rc, chk, _ = pre(url, 5000, os.path.join(tmp, "dl"), nb); t.check("bench help lacking -ncmoe/dio is flagged and exit is 11", chk.get("flag-ncmoe") is False and chk.get("lm_dio") is False and rc == 11)
    rc, chk, _ = pre(url, 5000, os.path.join(tmp, "dl"), "/nonexistent/bench"); t.check("missing bench binary is flagged", chk.get("bench_binary") is False and rc == 11)
    rc, chk, _ = pre(url, 10**15, os.path.join(tmp, "dl"), good); t.check("a model larger than free disk is flagged (disk_headroom)", chk.get("disk_headroom") is False and rc == 11)
    rc, chk, _ = pre(f"http://127.0.0.1:1/x", 5000, os.path.join(tmp, "dl"), good); t.check("connection refused is flagged for range support and size", chk.get("range_support") is False and chk.get("remote_size") is False)
    rc, chk, _ = pre(url, 5000, "/proc/nonexistent/dl", good); t.check("an unwritable download dir is flagged (write_test)", chk.get("write_test") is False)
    srv.shutdown(); return t
if __name__ == "__main__": sys.exit(0 if run().summary() else 1)
