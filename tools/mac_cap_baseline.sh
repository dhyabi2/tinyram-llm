#!/bin/bash
# Naive-baseline collapse on macOS: plain mmap (repack off), a wired mlock ballast squeezes memory, residency of the model file is measured with mincore afterwards.
# usage: mac_cap_baseline.sh BENCH MODEL "BALLAST_GB ..." TOKENS OUT.jsonl ["ENV=VALUE ..."]   (the optional last argument is exported to llama-bench, e.g. the expert read pool)
# Safety: skips a stage if swap used is above 9 GB or the free-memory percentage is below 12; every run has a hard timeout; the ballast is always released.
BENCH=$1; MODEL=$2; SIZES=$3; TOK=${4:-8}; OUT=${5:-mac-cap-baseline.jsonl}; XENV=${6:-}
HERE=$(cd "$(dirname "$0")" && pwd); W=$(mktemp -d); trap 'pkill -f "$W/ballast" 2>/dev/null; rm -rf "$W"' EXIT
cc -O2 -o $W/ballast "$HERE/../mac/ballast.c" || exit 2
pageins() { vm_stat | awk '/^Pageins/{gsub(/\./,"",$2); print $2}'; }
wired() { vm_stat | awk '/wired down/{gsub(/\./,"",$4); print $4}'; }
swap_gb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); printf "%.1f", $6/1024}'; }
freepct() { memory_pressure 2>/dev/null | awk '/free percentage/{gsub("%","",$NF); print $NF}'; }
PSZ=$(vm_stat | awk '/page size of/{print $8}')
for gb in $SIZES; do
  sw=$(swap_gb); fp=$(freepct)
  if python3 -c "import sys;sys.exit(0 if float('$sw')<9 and float('${fp:-50}')>=12 else 1)"; then :; else echo "{\"ballast_gb\":$gb,\"skipped\":\"swap ${sw} GB or free ${fp}% outside the safe range\"}" | tee -a $OUT; continue; fi
  BP=; W0=$(wired)
  if [ "$gb" != "0" ]; then B=$(python3 -c "print(int($gb*2**30))"); $W/ballast $B > $W/b.out 2>&1 & BP=$!; for _ in $(seq 60); do sleep 3; W1=$(wired); python3 -c "import sys;sys.exit(0 if ($W1-$W0)>=0.97*($B/$PSZ) else 1)" && break; done; FR=$(python3 -c "print(round(100*($W1-$W0)/($B/$PSZ),1))"); else FR=0; fi
  if [ "$gb" != "0" ] && python3 -c "import sys;sys.exit(0 if float('$FR')>=95 else 1)"; then :; elif [ "$gb" != "0" ]; then echo "{\"ballast_gb\":$gb,\"invalid\":\"ballast wired only ${FR}% of the target (memory too fragmented or full: swap $(swap_gb) GB); row skipped\"}" | tee -a $OUT; [ -n "$BP" ] && kill $BP 2>/dev/null && wait $BP 2>/dev/null; sleep 15; continue; fi
  p0=$(pageins); t0=$(date +%s)
  perl -e 'alarm shift; exec @ARGV' 900 env $XENV $BENCH -m $MODEL -ngl 0 -t 4 -p 0 -n $TOK -r 1 --repack 0 -lm mmap -o json > $W/o.json 2> $W/o.err; rc=$?
  t1=$(date +%s); p1=$(pageins)
  res=$(python3 "$HERE/mincore_probe.py" $MODEL --json); sw2=$(swap_gb); fp2=$(freepct)
  python3 - "$gb" "$FR" "$rc" "$p0" "$p1" "$PSZ" "$((TOK+1))" "$t0" "$t1" "$res" "$sw" "$sw2" "$fp" "$fp2" "$W/o.json" "$XENV" >> $OUT <<'PY'
import sys, json
gb, fr, rc, p0, p1, psz, tok, t0, t1, res, sw, sw2, fp, fp2, jf = sys.argv[1:16]
try: tps = json.load(open(jf))[0]["avg_ts"]
except Exception: tps = None
r = json.loads(res)
print(json.dumps({"ballast_gb": float(gb), "ballast_wired_pct": float(fr), "rc": int(rc), "tok_s": tps, "tokens_counted": int(tok), "extra_env": sys.argv[16] if len(sys.argv) > 16 else "", "pagein_gib_per_token": round((int(p1) - int(p0)) * int(psz) / 2**30 / int(tok), 3),
  "wall_s": int(t1) - int(t0), "model_resident_gib_after": round(r["resident_bytes"] / 2**30, 2), "model_resident_pct_after": round(100 * r["fraction"], 1), "swap_gb_before": float(sw), "swap_gb_after": float(sw2), "free_pct_before": fp, "free_pct_after": fp2}))
PY
  tail -1 $OUT
  [ -n "$BP" ] && kill $BP 2>/dev/null && wait $BP 2>/dev/null; sleep 15
done
