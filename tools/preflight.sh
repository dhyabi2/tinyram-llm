#!/usr/bin/env bash
# preflight.sh <model_url> <expected_bytes> <download_dir> <llama_bench_binary>
# Fast checks that surface known failures in seconds. One JSON line per check on stdout; exit 11 on a fetch problem, 12 on a runtime problem.
URL=$1; WANT=$2; DIR=$3; BENCH=$4; FAIL=0
emit() { printf '{"check":"%s","ok":%s,"detail":"%s"}\n' "$1" "$2" "$3"; [ "$2" = true ] || FAIL=$((FAIL+1)); }
printf 'PREFLIGHT/1 %s %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(uname -s)" "$(uname -m)"
OS=$(uname -s); ARCH=$(uname -m)
case "$OS/$ARCH" in Darwin/x86_64|Darwin/arm64|Linux/x86_64|Linux/aarch64) emit os_arch true "$OS/$ARCH";; *) emit os_arch false "unsupported $OS/$ARCH";; esac
if [ "$OS" = Darwin ]; then FL=$(sysctl -n machdep.cpu.features machdep.cpu.leaf7_features 2>/dev/null | tr 'A-Z' 'a-z'); MEM=$(sysctl -n hw.memsize)
else FL=$(grep -m1 -E '^(flags|Features)' /proc/cpuinfo | cut -d: -f2); MEM=$(awk '/^MemTotal:/{printf "%d",$2*1024}' /proc/meminfo)
  [ -r /sys/fs/cgroup/memory.max ] && { M2=$(cat /sys/fs/cgroup/memory.max); [ "$M2" != max ] && [ "$M2" -lt "$MEM" ] && MEM=$M2; }; fi
if echo "$FL" | grep -qwE 'avx2|asimd|neon'; then emit cpu_flags true "avx2_or_neon$(echo "$FL" | grep -qw avx512f && echo +avx512f)"; else emit cpu_flags false "no avx2/neon: llama.cpp CPU build may crash with SIGILL"; fi
emit ram true "$MEM bytes usable"
python3 - <<'PY' > /tmp/ps.out 2>&1
import re
def parse_size(t):
    m=re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*(GB|GiB|MB|MiB)\s*",t)
    if not m: raise ValueError(t)
    v=float(m.group(1)); u={"GB":1e9,"GiB":2**30,"MB":1e6,"MiB":2**20}[m.group(2)]
    return int(round(v*u))
assert parse_size("69 GB")==69000000000 and parse_size("14.25 GB")==14250000000 and parse_size("18.6 GB")==18600000000
print("ok")
PY
[ "$(cat /tmp/ps.out)" = ok ] && emit size_parse true "decimal sizes parse exactly" || emit size_parse false "$(tail -1 /tmp/ps.out)"
mkdir -p "$DIR" 2>/dev/null
FREE=$(python3 -c "import os,sys;v=os.statvfs(sys.argv[1]);print(v.f_bavail*v.f_frsize)" "$DIR")
NEED=$(python3 -c "w=$WANT;print(int(w+max(0.10*w,2*2**30)))")
[ "$FREE" -ge "$NEED" ] && emit disk_headroom true "free $FREE >= need $NEED" || emit disk_headroom false "free $FREE < need $NEED"
T="$DIR/.preflight_write"; python3 -c "import os,sys;f=open(sys.argv[1],'wb');f.write(os.urandom(1048576));f.flush();os.fsync(f.fileno())" "$T" 2>/dev/null && [ "$(stat -f%z "$T" 2>/dev/null || stat -c%s "$T")" = 1048576 ] && emit write_test true "1 MiB fsynced" || emit write_test false "cannot write in $DIR"; rm -f "$T"
TOOL=$(command -v curl wget aria2c 2>/dev/null | head -1); [ -n "$TOOL" ] && emit download_tool true "$(basename "$TOOL")" || emit download_tool false "no curl/wget/aria2c"
H=$(curl -sIL -m 20 -r 0-0 "$URL" | grep -E '^HTTP' | tail -1 | tr -d '\r'); echo "$H" | grep -q 206 && emit range_support true "$H" || emit range_support false "expected 206, got: $H"
REM=$(curl -sIL -m 20 "$URL" | awk 'tolower($1)=="content-length:"{v=$2} END{gsub(/\r/,"",v);print v}')
[ "$REM" = "$WANT" ] && emit remote_size true "$REM == expected" || emit remote_size false "remote $REM != expected $WANT"
[ -x "$BENCH" ] && emit bench_binary true "$BENCH" || emit bench_binary false "llama-bench not executable"
HELP=$("$BENCH" -h 2>&1); for f in -lm --repack -ncmoe; do echo "$HELP" | grep -q -- "$f" && emit "flag$f" true present || emit "flag$f" false "missing in llama-bench -h"; done
echo "$HELP" | grep -q "dio" && emit lm_dio true "dio listed" || emit lm_dio false "dio not listed"
echo "{\"summary\":\"$FAIL failed checks\"}"; [ "$FAIL" = 0 ] || exit 11
