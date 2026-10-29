#!/usr/bin/env bash
# probe.sh - measure how a big MoE GGUF behaves on a small-RAM Linux box (Colab, Kaggle, Oracle Always Free, any VM).
#
# What it answers (the unmeasured numbers behind the 100x claim):
#   B_mmap        decode tok/s with file-backed pages (-lm mmap)         the strong baseline / the design's runtime
#   B_swap        decode tok/s with weights as anonymous memory + swap   the brief's naive baseline (skipped if swap cannot be enabled)
#   faults/token  major page faults per generated token (gate G3)
#   GFLOP/s       effective compute rate = tok/s x 2 x active parameters (calibrates the 60 GFLOP/s assumption)
#   disk          sequential and 1 MiB random read speed of the disk holding the model (gate G5)
# It never claims a number it did not measure: a step that cannot run is reported as SKIPPED with the reason.
#
# usage: bash probe.sh [--ram-gb 8] [--model-repo ggml-org/gpt-oss-120b-GGUF] [--model-file gpt-oss-120b-MXFP4.gguf]
#                      [--active-b 5.1] [--model-gb 64] [--workdir ~/moe-probe] [--tokens 64] [--reps 2] [--dry-run]
#   --dry-run    only report the machine and what would happen (nothing is downloaded or built)
#   --swap-test  also run the B_swap case. OFF by default: it turns swap off/on and creates a swap file, which you must not do on a shared server
set -uo pipefail

RAM_GB=8; REPO=ggml-org/gpt-oss-120b-GGUF; FILE=gpt-oss-120b-MXFP4.gguf; ACTIVE_B=5.1; MODEL_GB=64; SWAP_TEST=0; WORK=$HOME/moe-probe; TOK=64; REPS=2; DRY=0
while [ $# -gt 0 ]; do case "$1" in
  --ram-gb) RAM_GB=$2; shift 2;; --model-repo) REPO=$2; shift 2;; --model-file) FILE=$2; shift 2;; --active-b) ACTIVE_B=$2; shift 2;; --model-gb) MODEL_GB=$2; shift 2;;
  --workdir) WORK=$2; shift 2;; --tokens) TOK=$2; shift 2;; --reps) REPS=$2; shift 2;; --dry-run) DRY=1; shift;; --swap-test) SWAP_TEST=1; shift;;
  *) echo "unknown option $1" >&2; exit 2;; esac; done

mkdir -p "$WORK" && cd "$WORK" || exit 1
RESULT=$WORK/results.json; LOG=$WORK/probe.log; : > "$LOG"
say() { echo "[probe] $*" | tee -a "$LOG"; }
skip() { say "SKIPPED: $*"; SKIPS+=("$*"); }
SKIPS=()

# ------------------------------------------------------------------ 1. the machine
OS=$(uname -s); ARCH=$(uname -m)
CORES=$(lscpu -p=Core 2>/dev/null | grep -v '^#' | sort -u | wc -l); [ "${CORES:-0}" -ge 1 ] || CORES=$(nproc 2>/dev/null || echo 2)
MEM_KB=$(awk '/^MemTotal/{print $2}' /proc/meminfo 2>/dev/null || echo 0); MEM_GB=$(python3 -c "print(round($MEM_KB/1048576,1))")
DISK_GB=$(df -BG --output=avail "$WORK" 2>/dev/null | tail -1 | tr -dc 0-9); DISK_GB=${DISK_GB:-0}
CPU_MODEL=$(lscpu 2>/dev/null | awk -F: '/Model name/{gsub(/^ +/,"",$2); print $2; exit}')
CG=none; [ -f /sys/fs/cgroup/cgroup.controllers ] && CG=v2
say "machine: $OS $ARCH, $CORES physical cores, $MEM_GB GB RAM, $DISK_GB GB free in $WORK, cpu='$CPU_MODEL', cgroup=$CG, root=$([ "$(id -u)" = 0 ] && echo yes || echo no)"
SWAP_ON=$(swapon --show --noheadings 2>/dev/null | wc -l)
# MODEL_GB: gpt-oss-120b-MXFP4.gguf is ~63.4 GB (from the repo listing); set --model-gb for another model (it sizes the disk check and the swap file)
MODEL_GB=$(python3 -c "import math;print(math.ceil(float('$MODEL_GB')))")
NEED=$((MODEL_GB + 8))
say "plan: cap usable RAM at ${RAM_GB} GB, run $FILE ($REPO), $REPS reps x $TOK tokens; needs ~${NEED} GB free disk (have $DISK_GB)"
[ "$DISK_GB" -ge "$NEED" ] || say "WARNING: not enough free disk for the model on this box"
if [ "$DRY" = 1 ]; then say "dry run: stopping here"; exit 0; fi
[ "$DISK_GB" -ge "$NEED" ] || { say "abort: free disk $DISK_GB GB < $NEED GB"; exit 3; }
[ "$OS" = Linux ] || { say "abort: needs Linux (/proc/vmstat)"; exit 3; }

# ------------------------------------------------------------------ 2. tools
if ! command -v cmake >/dev/null || ! command -v git >/dev/null; then
  say "installing build tools"; (apt-get update -qq && apt-get install -y -qq cmake git build-essential fio python3-pip libcurl4-openssl-dev) >>"$LOG" 2>&1 || skip "could not install build tools (need root/apt)"
fi
command -v fio >/dev/null || (apt-get install -y -qq fio >>"$LOG" 2>&1 || true)
pip install -q -U "huggingface_hub[cli]" >>"$LOG" 2>&1 || say "pip install huggingface_hub failed"

# ------------------------------------------------------------------ 3. build llama.cpp (CPU only)
if [ ! -x llama.cpp/build/bin/llama-bench ]; then
  say "building llama.cpp (CPU only)"
  [ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp >>"$LOG" 2>&1 || { say "clone failed"; exit 4; }
  (cd llama.cpp && cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF >>"$LOG" 2>&1 \
    && cmake --build build -j"$(nproc)" --target llama-bench llama-completion >>"$LOG" 2>&1) || { say "build failed, see $LOG"; exit 4; }
fi
BIN=$WORK/llama.cpp/build/bin; COMMIT=$(git -C llama.cpp log --oneline -1 2>/dev/null)
say "llama.cpp: $COMMIT"

# ------------------------------------------------------------------ 4. model
if [ ! -f "$FILE" ]; then
  say "downloading $REPO/$FILE (~${MODEL_GB} GB)"
  if command -v hf >/dev/null; then hf download "$REPO" "$FILE" --local-dir . >>"$LOG" 2>&1
  elif command -v huggingface-cli >/dev/null; then huggingface-cli download "$REPO" "$FILE" --local-dir . >>"$LOG" 2>&1
  else curl -fL -C - --retry 5 --retry-delay 5 -o "$FILE.part" "https://huggingface.co/$REPO/resolve/main/$FILE" >>"$LOG" 2>&1 && mv "$FILE.part" "$FILE"; fi
  [ -f "$FILE" ] || { say "download failed"; exit 5; }
fi
SIZE=$(stat -c %s "$FILE")
REMOTE=$(curl -sIL -m 20 "https://huggingface.co/$REPO/resolve/main/$FILE" 2>/dev/null | awk 'tolower($1)=="content-length:"{v=$2} END{gsub(/\r/,"",v);print v}')
if [ -n "$REMOTE" ] && [ "$REMOTE" != "$SIZE" ]; then say "abort: $FILE is $SIZE bytes but the server says $REMOTE (truncated download?). Delete it and rerun to resume."; exit 5; fi
say "model: $FILE $(python3 -c "print(round($SIZE/1e9,2))") GB"

# ------------------------------------------------------------------ 5. disk speed (cache-bypassing, no fio needed)
cat > "$WORK/disk_bench.py" <<'DISKPY'
#!/usr/bin/env python3
"""Cache-bypassing read benchmark for a large file; works on macOS (F_NOCACHE) and Linux (O_DIRECT). No fio needed.

usage: disk_bench.py FILE [--seconds 10] [--block-kib 1024]
Prints one JSON line: {"seq_mib_s": .., "rand_mib_s": .., "block_kib": .., "method": ..}
Sequential: reads blocks from a random start. Random: reads block-aligned blocks at random offsets. The file should be larger than RAM,
and the numbers are only as cold as the bypass is: if the filesystem ignores the hint, a second pass over the same blocks would be cached,
so each pass here starts at a fresh random region and never rereads a block.
"""
import argparse, json, mmap, os, random, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("file"); ap.add_argument("--seconds", type=float, default=10); ap.add_argument("--block-kib", type=int, default=1024, help="read size; 4096-byte multiples (O_DIRECT needs alignment)")
a = ap.parse_args()
BLK = a.block_kib * 1024
size = os.path.getsize(a.file)
if size < 8 * BLK:
    sys.exit("file too small for a read benchmark")

if sys.platform == "darwin":
    import fcntl
    fd = os.open(a.file, os.O_RDONLY)
    fcntl.fcntl(fd, 48, 1)                      # F_NOCACHE
    method = "F_NOCACHE"
else:
    fd = os.open(a.file, os.O_RDONLY | getattr(os, "O_DIRECT", 0))
    method = "O_DIRECT"
buf = mmap.mmap(-1, BLK)                          # page-aligned, as O_DIRECT needs
nblk = size // BLK


def run(offsets):
    n, t0 = 0, time.time()
    for off in offsets:
        if time.time() - t0 >= a.seconds:
            break
        got = os.preadv(fd, [buf], off)
        if got <= 0:
            break
        n += got
    return n / 2**20 / max(time.time() - t0, 1e-9)


rng = random.Random(1234)
start = rng.randrange(0, max(1, nblk - 1))
seq = run(((start + i) % nblk) * BLK for i in range(nblk))
order = list(range(nblk)); rng.shuffle(order)
rnd = run(b * BLK for b in order)
print(json.dumps({"seq_mib_s": round(seq), "rand_mib_s": round(rnd), "block_kib": a.block_kib, "method": method}))
DISKPY
DISK_JSON="null"
say "disk: cache-bypassing reads on the model file at 128 KiB, 1 MiB and 4 MiB (the read size decides the speed), 6 s per pass"
DJ=""
for K in 128 1024 4096; do
  R=$(python3 -I "$WORK/disk_bench.py" "$FILE" --seconds 6 --block-kib $K 2>>"$LOG") && DJ="$DJ${DJ:+,}\"$K\":$R"
done
if [ -n "$DJ" ]; then DISK_JSON="{$DJ}"; say "disk: $DJ"; else skip "disk benchmark failed (see log)"; fi

# ------------------------------------------------------------------ 6. cap the usable RAM
BALLAST_PID=""; CAP_LABEL="UNCAPPED"; EFF_CACHE_GB=null
avail_gb() { awk '/MemAvailable/{printf "%.2f", $2/1048576}' /proc/meminfo; }
cap_ram() {
  local want=$1 avail extra
  sync; [ "$(id -u)" = 0 ] && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null
  avail=$(avail_gb)
  # sized from what is AVAILABLE now (free + reclaimable cache), not from total RAM: sizing from total left the cache larger than the target
  extra=$(python3 -c "print(max(0, int(($avail - $want) * 1024)))")
  if [ "$extra" -le 0 ]; then say "RAM cap: only $avail GB available <= $want GB, no ballast needed"; CAP_LABEL="NATURAL"; EFF_CACHE_GB=$avail; return; fi
  # hold `extra` MiB, check what is left, and resize once if the first attempt missed the target by more than 10%
  for attempt in 1 2; do
    python3 - "$extra" <<'PY' &
import sys, mmap, ctypes, time
n = int(sys.argv[1]) * 1024 * 1024
buf = mmap.mmap(-1, n)
for i in range(0, n, 4096): buf[i] = 1
try: ctypes.CDLL("libc.so.6").mlockall(3)
except Exception: pass
print("ballast ready", flush=True)
time.sleep(10**6)
PY
    BALLAST_PID=$!; sleep 6
    EFF_CACHE_GB=$(avail_gb)
    if python3 -c "import sys;sys.exit(0 if abs($EFF_CACHE_GB - $want) <= 0.10 * $want else 1)"; then CAP_LABEL="MEASURED-CAP"; break; fi
    CAP_LABEL="SOFT-CAP"
    [ "$attempt" = 2 ] && break
    kill "$BALLAST_PID" 2>/dev/null; wait "$BALLAST_PID" 2>/dev/null; sleep 2
    extra=$(python3 -c "print(max(0, int($extra - ($want - $EFF_CACHE_GB) * 1024)))")     # available was too low -> hold less; too high -> hold more
    [ "$extra" -gt 0 ] || break
    say "RAM cap: attempt 1 left $EFF_CACHE_GB GB available against $want GB; retrying with ${extra} MiB"
  done
  say "RAM cap: ballast ${extra} MiB, available now $EFF_CACHE_GB GB against a target of $want GB -> $CAP_LABEL"
}
trap '[ -n "$BALLAST_PID" ] && kill "$BALLAST_PID" 2>/dev/null' EXIT

vm() { awk -v k="$1" '$1==k{print $2}' /proc/vmstat; }
run_case() {   # name, extra flags, drop-cache?
  local name=$1 flags=$2 out=$WORK/case_$1.txt tps="" worst=""
  : > "$out"
  for r in $(seq "$REPS"); do
    sync; [ "$(id -u)" = 0 ] && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null
    local m0 p0 m1 p1 t0 t1
    m0=$(vm pgmajfault); p0=$(vm pgpgin); t0=$(date +%s.%N)
    "$BIN/llama-bench" -m "$FILE" -ngl 0 -t "$CORES" -p 0 -n "$TOK" -r 1 --repack 0 -o json $flags > "$out.$r" 2>>"$LOG" || { say "case $name rep $r failed"; continue; }
    t1=$(date +%s.%N); m1=$(vm pgmajfault); p1=$(vm pgpgin)
    python3 - "$out.$r" "$m0" "$m1" "$p0" "$p1" "$t0" "$t1" "$((TOK + 1))" "$r" >> "$out" <<'PY'
import sys, json
f, m0, m1, p0, p1, t0, t1, tok, r = sys.argv[1:]
d = json.load(open(f)); row = d[-1] if isinstance(d, list) else d
tps = row.get("avg_ts") or row.get("tokens_per_second") or 0
print(json.dumps({"rep": int(r), "tok_s": tps, "tokens_counted": int(tok), "major_faults": int(m1) - int(m0), "faults_per_token": (int(m1) - int(m0)) / int(tok),
                  "read_mib": (int(p1) - int(p0)) / 1024, "read_mib_per_token": (int(p1) - int(p0)) / 1024 / int(tok), "wall_s": float(t1) - float(t0)}))
PY
  done
  python3 - "$out" "$name" <<'PY'
import sys, json, statistics as st
rows = [json.loads(l) for l in open(sys.argv[1]) if l.startswith("{")]
if not rows: print(json.dumps({"case": sys.argv[2], "status": "no successful run"})); sys.exit()
t = [r["tok_s"] for r in rows]
print(json.dumps({"case": sys.argv[2], "status": "ok", "tok_s_median": st.median(t), "tok_s_all": t,
                  "tokens_counted": rows[0]["tokens_counted"], "faults_per_token_median": st.median([r["faults_per_token"] for r in rows]), "read_mib_median": st.median([r["read_mib"] for r in rows]), "read_mib_per_token_median": st.median([r["read_mib_per_token"] for r in rows])}))
PY
}

cap_ram "$RAM_GB"
RES=()
say "case 1/2: B_mmap  (-lm mmap, file-backed pages, swap OFF)"
[ "$SWAP_TEST" = 1 ] && swapoff -a 2>/dev/null; RES+=("$(run_case mmap "-lm mmap")")
say "$(echo "${RES[-1]}")"

say "case 2/2: B_swap  (-lm none: weights as anonymous memory, swap ON)"
SWAPFILE=$WORK/swapfile; SW_OK=0
if [ "$SWAP_TEST" != 1 ]; then skip "B_swap: not requested (pass --swap-test on a box where changing swap is safe)"
elif [ "$(id -u)" = 0 ]; then
  (fallocate -l "$((MODEL_GB + 4))G" "$SWAPFILE" && chmod 600 "$SWAPFILE" && mkswap "$SWAPFILE" >/dev/null && swapon "$SWAPFILE") 2>>"$LOG" && SW_OK=1
fi
if [ "$SWAP_TEST" != 1 ]; then :
elif [ "$SW_OK" = 1 ]; then sysctl -w vm.swappiness=60 >/dev/null 2>&1; RES+=("$(run_case swap "-lm none")"); say "$(echo "${RES[-1]}")"; swapoff "$SWAPFILE" 2>/dev/null; rm -f "$SWAPFILE"
else skip "B_swap: could not create/enable a swap file (needs root and a kernel that allows swapon; many notebook containers do not)"; fi

# ------------------------------------------------------------------ 7. result
python3 - "$RESULT" "$ACTIVE_B" "$CORES" "$MEM_GB" "$RAM_GB" "$DISK_JSON" "$COMMIT" "$FILE" "$CPU_MODEL" "$ARCH" "$CAP_LABEL" "$EFF_CACHE_GB" "$OS" "${RES[@]}" -- "${SKIPS[@]:-}" <<'PY'
import sys, json
out, active, cores, mem, cap, disk, commit, model, cpu, arch, cap_label, eff, osname = sys.argv[1:14]
rest = sys.argv[14:]; i = rest.index("--"); cases = [json.loads(x) for x in rest[:i]]; skips = [s for s in rest[i + 1:] if s]
for c in cases:
    if c.get("status") == "ok": c["effective_gflops"] = round(c["tok_s_median"] * 2 * float(active), 1)   # tok/s x 2 FLOP x active params (an upper-bound style estimate)
res = {"schema": 2, "os": osname, "machine": {"cpu": cpu, "arch": arch, "physical_cores": int(cores), "ram_gb": float(mem), "ram_cap_gb": float(cap), "cap_label": cap_label, "available_gb_after_cap": None if eff == "null" else float(eff)}, "llama_cpp": commit, "model": model,
       "disk": json.loads(disk), "cases": cases, "skipped": skips, "note": "faults_per_token and tok_s are measured; effective_gflops assumes the run is compute-bound"}
json.dump(res, open(out, "w"), indent=1); print(json.dumps(res, indent=1))
PY
say "done. Send back: $RESULT  and  $LOG"
