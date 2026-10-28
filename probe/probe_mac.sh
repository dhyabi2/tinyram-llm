#!/usr/bin/env bash
# probe_mac.sh - decode speed and page-ins per token for a GGUF run with -lm mmap on macOS.
# usage: bash probe_mac.sh --bench /path/to/llama-bench --model file.gguf [--tokens 32] [--reps 2] [--out results.json]
# Page-ins come from `vm_stat` (system-wide, so close other apps first). No RAM cap is applied: use a model larger than RAM to test streaming.
set -uo pipefail
BENCH=; MODEL=; TOK=32; REPS=2; OUT=results.json
while [ $# -gt 0 ]; do case "$1" in
  --bench) BENCH=$2; shift 2;; --model) MODEL=$2; shift 2;; --tokens) TOK=$2; shift 2;; --reps) REPS=$2; shift 2;; --out) OUT=$2; shift 2;;
  *) echo "unknown option $1" >&2; exit 2;; esac; done
[ -x "$BENCH" ] && [ -f "$MODEL" ] || { echo "need --bench and --model" >&2; exit 2; }
pageins() { vm_stat | awk '/^Pageins/{gsub(/\./,"",$2); print $2}'; }
PSZ=$(vm_stat | awk '/page size of/{print $8}')
RAM_GB=$(python3 -c "import subprocess;print(round(int(subprocess.check_output(['sysctl','-n','hw.memsize']))/2**30,1))")
CPU=$(sysctl -n machdep.cpu.brand_string); CORES=$(sysctl -n hw.physicalcpu)
SIZE=$(stat -f %z "$MODEL")
echo "[probe] $CPU, $CORES cores, ${RAM_GB} GB RAM, model $(python3 -c "print(round($SIZE/1e9,2))") GB"
LINES=$(mktemp)
for r in $(seq 1 "$REPS"); do
  P0=$(pageins); T0=$(python3 -c "import time;print(time.time())")
  "$BENCH" -m "$MODEL" -ngl 0 -t "$CORES" -p 0 -n "$TOK" -r 1 -lm mmap --repack 0 -o json > "$OUT.$r" 2>"$OUT.$r.err" || { echo "[probe] rep $r failed (see $OUT.$r.err)"; continue; }
  P1=$(pageins); T1=$(python3 -c "import time;print(time.time())")
  python3 - "$OUT.$r" "$P0" "$P1" "$PSZ" "$TOK" "$T0" "$T1" "$r" >> "$LINES" <<'PY'
import sys, json
f, p0, p1, psz, tok, t0, t1, rep = sys.argv[1:9]
d = json.load(open(f)); ts = d[0]["avg_ts"]
mib = (int(p1) - int(p0)) * int(psz) / 2**20
print(json.dumps({"rep": int(rep), "tok_s": round(ts, 6), "pageins_mib": round(mib, 1), "read_mib_per_token": round(mib / int(tok), 1), "wall_s": round(float(t1) - float(t0), 1)}))
PY
  echo "[probe] $(tail -1 "$LINES")"
done
python3 - "$OUT" "$CPU" "$CORES" "$RAM_GB" "$(basename "$MODEL")" "$SIZE" "$LINES" <<'PY'
import sys, json, statistics as s
out, cpu, cores, ram, model, size, lines = sys.argv[1:]
runs = [json.loads(x) for x in open(lines) if x.strip()]
r = {"machine": {"cpu": cpu, "physical_cores": int(cores), "ram_gb": float(ram)}, "model": model, "model_gb": round(int(size) / 1e9, 2), "case": "mmap",
     "runs": runs, "tok_s_median": s.median([x["tok_s"] for x in runs]) if runs else None,
     "read_mib_per_token_median": s.median([x["read_mib_per_token"] for x in runs]) if runs else None,
     "note": "page-ins are system-wide (vm_stat), so other apps add noise; no RAM cap applied"}
json.dump(r, open(out, "w"), indent=1); print("[probe] wrote", out)
PY
