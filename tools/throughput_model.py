#!/usr/bin/env python3
"""Throughput model: tok/s = 1 / max(t_ssd, t_ram, t_compute). Every input is an ASSUMPTION, printed below.
Hit rates come from hit_rate_sim.py (synthetic trace; pool=1200 experts, plain LRU)."""
L, E, K = 48, 512, 10
D = 3.4e9                                     # always-on (dense) params implied by 125B total / 6B active (Scenario B)
EXPERT_PARAMS = (125e9 - D) / (L * E)         # ~4.95M params per expert
BPW_DENSE = 7.2                               # attention Q6_K, shared Q8_0, router BF16, lm_head Q6_K -> 3.06 GB
BPW_EXPERT = 2.256                            # parameter-weighted tiered average (mixed-quant-plan)
EXPERT_MB = EXPERT_PARAMS * BPW_EXPERT / 8 / 1e6
DRIVES = {"SATA SSD 0.45GB/s": 0.45, "NVMe PCIe3 2.0GB/s": 2.0, "NVMe PCIe4 3.5GB/s": 3.5}   # effective random-read
RAM_GBS, GFLOPS = 25.0, 60.0
TAU = 0.78 + 0.78 * 0.66 + 0.78 * 0.66 * 0.55 + 1      # accepted tokens per MTP round incl. bonus = 2.578
K_DRAFT = 3
t_comp = 12e9 / (GFLOPS * 1e9)
dense_gb = D * BPW_DENSE / 8 / 1e9
print(f"expert = {EXPERT_PARAMS/1e6:.2f}M params = {EXPERT_MB:.2f} MB; routed reads/token = {L*K}; dense read from RAM = {dense_gb:.2f} GB/token")
print(f"compute ceiling = {1/t_comp:.1f} tok/s at {GFLOPS:.0f} GFLOP/s (6e9 active params x 2 FLOP); RAM {RAM_GBS:.0f} GB/s (DDR4 dual-channel)")
print(f"MTP: modelled per round (4 positions verified together, {TAU:.3f} accepted tokens): dense read once, expert UNION reads, 4x compute per round\n")

def tps(h, ssd, mtp, overlap=True, persist=0.6):
    """overlap=True : drive reads fully hidden behind compute/RAM work   -> t = max(ts, tr, tc)   (optimistic)
       overlap=False: misses are on the critical path (layer L+1's experts are unknown until layer L finishes)  -> t = ts + max(tr, tc)
       mtp=True: one ROUND = (k+1)=4 positions verified in one batched pass, yielding TAU accepted tokens.
         - dense weights are read from RAM once per round (not per position)
         - expert reads = UNION of the 4 positions' experts per layer = 10*(1+3*(1-persist)), capped at 26
         - compute is 4 positions of arithmetic per round (rejected positions are wasted), i.e. ~1.55x per accepted token
         - the MTP draft's own compute and weights are NOT counted (optimistic for MTP)"""
    if not mtp:
        miss = L * K * (1 - h) * EXPERT_MB * 1e6
        ram = D * BPW_DENSE / 8 + L * K * h * EXPERT_MB * 1e6
        ts, tr, tc, per = miss / (ssd * 1e9), ram / (RAM_GBS * 1e9), t_comp, 1.0
    else:
        reads = L * min(26, K * (1 + K_DRAFT * (1 - persist)))
        miss = reads * (1 - h) * EXPERT_MB * 1e6
        ram = D * BPW_DENSE / 8 + reads * h * EXPERT_MB * 1e6
        ts, tr, tc, per = miss / (ssd * 1e9), ram / (RAM_GBS * 1e9), (K_DRAFT + 1) * t_comp, TAU
    if overlap:
        t = max(ts, tr, tc)
        return per / t, "compute" if t == tc else "SSD" if t == ts else "RAM"
    return per / (ts + max(tr, tc)), "serial"

print("token-to-token expert overlap (persist) -> pool-1200 LRU hit rate (simulated) -> tok/s  [no MTP | MTP]  (bound)")
SIM = [(0.20, 0.349), (0.35, 0.454), (0.50, 0.582), (0.70, 0.746), (0.85, 0.872)]
print(f"{'overlap':>8} {'hit':>6} | " + " | ".join(f"{d:>26}" for d in DRIVES))
for pers, h in SIM:
    cells = []
    for g in DRIVES.values():
        a, wa = tps(h, g, False); b, wb = tps(h, g, True, persist=pers)
        cells.append(f"{a:5.2f} ({wa:7}) | {b:5.2f}")
    print(f"{pers:8.2f} {h:6.3f} | " + " | ".join(f"{c:>26}" for c in cells))
print("\nSERIALIZED bound (misses NOT hidden behind compute):  t = t_ssd + max(t_ram, t_compute)   [no MTP | MTP]")
print(f"{'overlap':>8} {'hit':>6} | " + " | ".join(f"{d:>20}" for d in DRIVES))
for pers, h in SIM:
    cells = []
    for g in DRIVES.values():
        a, _ = tps(h, g, False, overlap=False); b, _ = tps(h, g, True, overlap=False, persist=pers)
        cells.append(f"{a:5.2f} | {b:5.2f}")
    print(f"{pers:8.2f} {h:6.3f} | " + " | ".join(f"{c:>20}" for c in cells))
print("\n100x condition: F/B >= 100 AND F >= 2 tok/s")
for B in (0.01, 0.03, 0.05, 0.10): print(f"  baseline B = {B:4.2f} tok/s -> need F >= {max(100*B, 2.0):4.1f} tok/s")
