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
SIZE=$(stat -c %s "$FILE"); say "model: $FILE $(python3 -c "print(round($SIZE/1e9,2))") GB"

# ------------------------------------------------------------------ 5. disk speed (gate G5)
DISK_JSON="null"
if command -v fio >/dev/null; then
  say "disk: sequential 1 MiB read, then random 1 MiB read (direct I/O, 20 s each) on the model file"
  sync; [ "$(id -u)" = 0 ] && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null
  SEQ=$(fio --name=seq --filename="$FILE" --rw=read --bs=1M --iodepth=8 --direct=1 --runtime=20 --time_based --readonly --output-format=json 2>/dev/null | sed -n '/^{/,$p' | python3 -c "import sys,json;print(round(json.load(sys.stdin)['jobs'][0]['read']['bw']/1024))" 2>/dev/null)
  RND=$(fio --name=rnd --filename="$FILE" --rw=randread --bs=1M --iodepth=8 --direct=1 --runtime=20 --time_based --readonly --output-format=json 2>/dev/null | sed -n '/^{/,$p' | python3 -c "import sys,json;print(round(json.load(sys.stdin)['jobs'][0]['read']['bw']/1024))" 2>/dev/null)
  say "disk: sequential ${SEQ:-?} MiB/s, random 1 MiB ${RND:-?} MiB/s"; DISK_JSON="{\"seq_mib_s\":${SEQ:-null},\"rand1m_mib_s\":${RND:-null}}"
else skip "fio not available, disk speed not measured"; fi

# ------------------------------------------------------------------ 6. cap the usable RAM
BALLAST_PID=""
cap_ram() {
  local total_gb=$1 want=$2 extra
  extra=$(python3 -c "print(max(0, int(($total_gb - $want) * 1024)))")
  if [ "$extra" -le 0 ]; then say "RAM cap: machine has $total_gb GB <= $want GB, no ballast needed"; return; fi
  # a process that holds ${extra} MiB of anonymous, touched (and locked if allowed) memory: the page cache then has only ${want} GB to work with
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
  BALLAST_PID=$!; sleep 5; say "RAM cap: ballast of ${extra} MiB held by pid $BALLAST_PID (MemAvailable now $(awk '/MemAvailable/{printf "%.1f GB", $2/1048576}' /proc/meminfo))"
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
    python3 - "$out.$r" "$m0" "$m1" "$p0" "$p1" "$t0" "$t1" "$TOK" "$r" >> "$out" <<'PY'
import sys, json
f, m0, m1, p0, p1, t0, t1, tok, r = sys.argv[1:]
d = json.load(open(f)); row = d[-1] if isinstance(d, list) else d
tps = row.get("avg_ts") or row.get("tokens_per_second") or 0
print(json.dumps({"rep": int(r), "tok_s": tps, "major_faults": int(m1) - int(m0), "faults_per_token": (int(m1) - int(m0)) / int(tok),
                  "read_mib": (int(p1) - int(p0)) / 1024, "wall_s": float(t1) - float(t0)}))
PY
  done
  python3 - "$out" "$name" <<'PY'
import sys, json, statistics as st
rows = [json.loads(l) for l in open(sys.argv[1]) if l.startswith("{")]
if not rows: print(json.dumps({"case": sys.argv[2], "status": "no successful run"})); sys.exit()
t = [r["tok_s"] for r in rows]
print(json.dumps({"case": sys.argv[2], "status": "ok", "tok_s_median": st.median(t), "tok_s_all": t,
                  "faults_per_token_median": st.median([r["faults_per_token"] for r in rows]), "read_mib_median": st.median([r["read_mib"] for r in rows])}))
PY
}

cap_ram "$MEM_GB" "$RAM_GB"
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
python3 - "$RESULT" "$ACTIVE_B" "$CORES" "$MEM_GB" "$RAM_GB" "$DISK_JSON" "$COMMIT" "$FILE" "$CPU_MODEL" "$ARCH" "${RES[@]}" -- "${SKIPS[@]:-}" <<'PY'
import sys, json
out, active, cores, mem, cap, disk, commit, model, cpu, arch = sys.argv[1:11]
rest = sys.argv[11:]; i = rest.index("--"); cases = [json.loads(x) for x in rest[:i]]; skips = [s for s in rest[i + 1:] if s]
for c in cases:
    if c.get("status") == "ok": c["effective_gflops"] = round(c["tok_s_median"] * 2 * float(active), 1)   # tok/s x 2 FLOP x active params (an upper-bound style estimate)
res = {"machine": {"cpu": cpu, "arch": arch, "physical_cores": int(cores), "ram_gb": float(mem), "ram_cap_gb": float(cap)}, "llama_cpp": commit, "model": model,
       "disk": json.loads(disk), "cases": cases, "skipped": skips, "note": "faults_per_token and tok_s are measured; effective_gflops assumes the run is compute-bound"}
json.dump(res, open(out, "w"), indent=1); print(json.dumps(res, indent=1))
PY
say "done. Send back: $RESULT  and  $LOG"
