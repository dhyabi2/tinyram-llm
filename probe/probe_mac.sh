#!/usr/bin/env bash
# probe_mac.sh - decode speed, page-ins per token and disk speed for a GGUF run with -lm mmap on macOS.
# usage: bash probe_mac.sh --bench /path/to/llama-bench --model file.gguf [--tokens 32] [--reps 2] [--cap-gb 8] [--out results.json]
#   --cap-gb N  emulate a machine with N GB of usable memory: a wired (mlock) ballast takes the rest, and the cap is VERIFIED from the rise in
#               wired pages. No sudo needed when the wire limit (sysctl vm.global_user_wire_limit) is high enough. Without it the run is UNCAPPED.
# Page-ins come from `vm_stat` and are system-wide, so close other apps first. The result file uses the same fields as probe.sh (schema 2).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BENCH=; MODEL=; TOK=32; REPS=2; OUT=results.json; CAP=
while [ $# -gt 0 ]; do case "$1" in
  --bench) BENCH=$2; shift 2;; --model) MODEL=$2; shift 2;; --tokens) TOK=$2; shift 2;; --reps) REPS=$2; shift 2;; --out) OUT=$2; shift 2;; --cap-gb) CAP=$2; shift 2;;
  *) echo "unknown option $1" >&2; exit 2;; esac; done
[ -x "$BENCH" ] && [ -f "$MODEL" ] || { echo "need --bench (executable) and --model (file)" >&2; exit 2; }
say() { echo "[probe] $*"; }
pageins() { vm_stat | awk '/^Pageins/{gsub(/\./,"",$2); print $2}'; }
vmpages() { vm_stat | awk -v k="$1" '$0 ~ k {gsub(/\./,"",$NF); print $NF}'; }
PSZ=$(vm_stat | awk '/page size of/{print $8}')
RAM_GB=$(python3 -c "import subprocess;print(round(int(subprocess.check_output(['sysctl','-n','hw.memsize']))/2**30,1))")
CPU=$(sysctl -n machdep.cpu.brand_string); CORES=$(sysctl -n hw.physicalcpu); ARCH=$(uname -m)
SIZE=$(stat -f %z "$MODEL")
say "$CPU, $CORES cores, ${RAM_GB} GB RAM, model $(python3 -c "print(round($SIZE/1e9,2))") GB"

# ---- disk: cache-bypassing reads at three read sizes (the read size decides the speed)
WORK=$(mktemp -d); trap 'rm -rf "$WORK"; [ -n "${BP:-}" ] && kill "$BP" 2>/dev/null' EXIT
DJ=""
for K in 128 1024 4096; do
  R=$(python3 -I "$HERE/../tools/disk_bench.py" "$MODEL" --seconds 6 --block-kib $K 2>/dev/null) && DJ="$DJ${DJ:+,}\"$K\":$R"
done
DISK_JSON="null"; [ -n "$DJ" ] && { DISK_JSON="{$DJ}"; say "disk: $DJ"; }

# ---- optional verified memory cap
CAP_LABEL="UNCAPPED"; EFF=null
if [ -n "$CAP" ]; then
  if cc -O2 -o "$WORK/ballast" "$HERE/ballast_mac.c" 2>/dev/null; then
    RECL=$(python3 - <<PY
import subprocess,re
t=subprocess.check_output(['vm_stat']).decode()
g=lambda k:int(re.search(k+r'.*?(\d+)\.',t).group(1))
print(int((g('Pages free')+g('Pages inactive')+g('Pages speculative')+g('Pages purgeable'))*$PSZ))
PY
)
    B=$(python3 -c "print(max(0,int($RECL - $CAP*2**30)))")
    if [ "$B" -gt 0 ]; then
      W0=$(vmpages "wired down"); "$WORK/ballast" "$B" > "$WORK/ballast.out" 2>&1 & BP=$!; sleep 8; W1=$(vmpages "wired down")
      RISE=$((W1 - W0)); NEED=$((B / PSZ * 95 / 100)); FRAC=$(python3 -c "print(round(100*$RISE/($B/$PSZ),1))")
      EFF=$(python3 -c "print(round(($RECL - $B)/2**30,2))")
      if [ "$RISE" -ge "$NEED" ]; then CAP_LABEL="MEASURED-WIRED-CAP"; say "cap: wired $((B/1048576)) MiB (${FRAC}% of the pages are wired, threshold 95%), about $EFF GB left for cache and apps"
      else CAP_LABEL="UNCAPPED"; say "cap FAILED: wired rise $RISE pages < $NEED (see $WORK/ballast.out); continuing uncapped"; kill "$BP" 2>/dev/null; BP=; EFF=null; fi
    else say "cap: only $(python3 -c "print(round($RECL/2**30,1))") GB reclaimable <= $CAP GB, no ballast needed"; CAP_LABEL="NATURAL"; EFF=$(python3 -c "print(round($RECL/2**30,2))"); fi
  else say "cap: could not compile the ballast; continuing uncapped"; fi
fi

# ---- timed runs. llama-bench runs a 1-token warmup before the timed tokens, so tokens_counted = TOK + 1
LINES=$(mktemp)
for r in $(seq 1 "$REPS"); do
  P0=$(pageins); T0=$(python3 -c "import time;print(time.time())")
  "$BENCH" -m "$MODEL" -ngl 0 -t "$CORES" -p 0 -n "$TOK" -r 1 -lm mmap --repack 0 -o json > "$WORK/b.$r" 2>"$WORK/b.$r.err" || { say "rep $r failed: $(tail -2 "$WORK/b.$r.err" | tr '\n' ' ')"; continue; }
  P1=$(pageins); T1=$(python3 -c "import time;print(time.time())")
  python3 - "$WORK/b.$r" "$P0" "$P1" "$PSZ" "$((TOK + 1))" "$T0" "$T1" "$r" >> "$LINES" <<'PY'
import sys, json
f, p0, p1, psz, tok, t0, t1, rep = sys.argv[1:9]
d = json.load(open(f)); ts = d[0]["avg_ts"]
mib = (int(p1) - int(p0)) * int(psz) / 2**20
print(json.dumps({"rep": int(rep), "tok_s": round(ts, 6), "tokens_counted": int(tok), "read_mib": round(mib, 1), "read_mib_per_token": round(mib / int(tok), 1), "wall_s": round(float(t1) - float(t0), 1)}))
PY
  say "$(tail -1 "$LINES")"
done
python3 - "$OUT" "$CPU" "$CORES" "$RAM_GB" "$(basename "$MODEL")" "$SIZE" "$LINES" "$CAP_LABEL" "$EFF" "$CAP" "$DISK_JSON" "$ARCH" <<'PY'
import sys, json, statistics as s
out, cpu, cores, ram, model, size, lines, label, eff, cap, disk, arch = sys.argv[1:13]
runs = [json.loads(x) for x in open(lines) if x.strip()]
case = {"case": "mmap", "status": "ok" if runs else "no successful run", "runs": runs}
if runs:
    case.update(tok_s_median=s.median([x["tok_s"] for x in runs]), tokens_counted=runs[0]["tokens_counted"], read_mib_per_token_median=s.median([x["read_mib_per_token"] for x in runs]))
r = {"schema": 2, "os": "Darwin", "machine": {"cpu": cpu, "arch": arch, "physical_cores": int(cores), "ram_gb": float(ram), "ram_cap_gb": float(cap) if cap else None, "cap_label": label,
     "available_gb_after_cap": None if eff == "null" else float(eff)}, "model": model, "model_gb": round(int(size) / 1e9, 2), "disk": json.loads(disk), "cases": [case],
     "note": "page-ins are system-wide (vm_stat), so other apps add noise"}
json.dump(r, open(out, "w"), indent=1); print("[probe] wrote", out)
PY
