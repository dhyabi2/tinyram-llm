#!/bin/bash
cd "$(dirname "$0")"
wired() { vm_stat | awk '/wired down/{gsub(/\./,"",$4);print $4}'; }
comp()  { vm_stat | awk '/occupied by compressor/{gsub(/\./,"",$5);print $5}'; }
B=$(python3 -c "print(int($1*1024**3))")
echo "ulimit -l: $(ulimit -l)"; sysctl vm.global_user_wire_limit vm.user_wire_limit hw.pagesize 2>&1
W0=$(wired); C0=$(comp)
./ballast $B > ballast.out 2>&1 &
BP=$!
sleep 8
W1=$(wired); C1=$(comp)
cat ballast.out
echo "wired pages: $W0 -> $W1 (rise $((W1-W0)), need >= $((B/4096*98/100)))   compressor: $C0 -> $C1"
memory_pressure 2>/dev/null | tail -1
kill $BP 2>/dev/null; sleep 2
echo "after release wired: $(wired)"
