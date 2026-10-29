#!/usr/bin/env python3
"""Cluster scaling model: decode tokens/s of a ~125B MoE (48 layers, 512 experts, top-10) on N cheap nodes. Parameters are estimates unless marked [M] measured; use it to see which bound stops scaling.
Schemes for ONE interactive stream:
  single   : one node.
  pipeline : contiguous layer ranges per node. Layers stay sequential, so compute does not shrink; what scales is the aggregate expert cache (more RAM -> fewer disk reads) and disk bandwidth in total.
  expert   : the experts of every layer are sharded over the nodes; the dense part (attention, router, shared tensors) stays on a master unless dense_split=True (tensor-parallel attention, 2 extra messages per layer).
Time per token (serial, pessimistic) = dense compute + expert compute (parallel part, with straggler factor) + disk misses (per node, reads in parallel across nodes) + network (messages x latency + bytes / bandwidth).
usage: cluster_model.py [--gflops 55] [--ram-gb 16] [--nvme 3.1] [--rtt-ms 0.3] [--bw-gbs 0.11] [--max-nodes 8] [--overlap] [--real-hit] [--dense-split]"""
import argparse, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
FLOP_TOT, DENSE_FRAC = 12.0, 3.4 / 6.0          # [E] 12 GFLOP/token, dense share of active parameters
EXP_GB, DENSE_GB, MODEL_GB = 1.296, 1.9, 69.0   # expert bytes touched per token, resident dense weights, file size
LAYERS, ACT_KB = 48, 4.0                        # activation per token per layer ~4 KB
STRAGGLER = 1.27                                # [M] mean max-load over mean load for experts split over nodes (trace study)

def lru(k, real):
    import combo_model as cm
    saved = cm.LRU
    if real: cm.LRU = cm.LRU_REAL
    try: return cm.lru(k)
    finally: cm.LRU = saved

def hit_for(cache_gb, bytes_per_token_gb, real):
    return lru(cache_gb * 1.074 / max(bytes_per_token_gb, 1e-9), real) if cache_gb > 0 else 0.0

def tok_s(n, scheme, gflops=55.0, ram_gb=16.0, nvme=3.1, rtt_ms=0.3, bw_gbs=0.11, overlap=False, real=False, dense_split=False, reserve_gb=None):
    if reserve_gb is None: reserve_gb = 3.6 if ram_gb <= 8 else 4.95    # matches the 2.5 / 9.15 / 25.15 GiB expert cache of an 8 / 16 / 32 GB machine used elsewhere in this repo
    dense_f, exp_f = FLOP_TOT * DENSE_FRAC, FLOP_TOT * (1 - DENSE_FRAC); cache_node = max(0.0, ram_gb - reserve_gb - (DENSE_GB if scheme != "pipeline" or n == 1 else DENSE_GB / n))
    if scheme == "single" or n == 1:
        hit = hit_for(cache_node, EXP_GB, real); io = (1 - hit) * EXP_GB / nvme; comp = FLOP_TOT / gflops; net = 0.0
    elif scheme == "pipeline":
        share = EXP_GB / n; hit = hit_for(cache_node, share, real); io = n * (1 - hit) * share / nvme    # nodes work one after another
        comp = FLOP_TOT / gflops; net = n * (rtt_ms / 2000.0 + ACT_KB * 1e-6 / bw_gbs)                    # n hops per token (forward + return)
    elif scheme == "expert":
        share = EXP_GB / n; hit = hit_for(cache_node, share, real); io = (1 - hit) * share / nvme            # all nodes read their own share in parallel
        dense_t = dense_f / gflops / (n if dense_split else 1); comp = dense_t + exp_f / gflops / n * (STRAGGLER if n > 1 else 1.0)
        msgs = LAYERS * (2 + (2 if dense_split else 0)); net = msgs * (rtt_ms / 2000.0 + ACT_KB * 1e-6 / bw_gbs)
    else: raise ValueError(scheme)
    t = max(comp, io) + net if overlap else comp + io + net
    return 1.0 / t, dict(comp=comp, io=io, net=net, hit=hit)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gflops", type=float, default=55.0); ap.add_argument("--ram-gb", type=float, default=16.0); ap.add_argument("--nvme", type=float, default=3.1); ap.add_argument("--rtt-ms", type=float, default=0.3)
    ap.add_argument("--bw-gbs", type=float, default=0.11); ap.add_argument("--max-nodes", type=int, default=8); ap.add_argument("--overlap", action="store_true"); ap.add_argument("--real-hit", action="store_true"); ap.add_argument("--dense-split", action="store_true")
    a = ap.parse_args()
    print(f"node: {a.gflops} GFLOP/s, {a.ram_gb} GB RAM, NVMe {a.nvme} GB/s; net RTT {a.rtt_ms} ms, {a.bw_gbs} GB/s; overlap={a.overlap}; hit curve={'real-model' if a.real_hit else '30B-trace'}; dense_split={a.dense_split}")
    print(f"{'nodes':>5} | " + " | ".join(f"{s:>28}" for s in ("pipeline", "expert")))
    for n in range(1, a.max_nodes + 1):
        row = []
        for s in ("pipeline", "expert"):
            v, d = tok_s(n, s, a.gflops, a.ram_gb, a.nvme, a.rtt_ms, a.bw_gbs, a.overlap, a.real_hit, a.dense_split); row.append(f"{v:5.2f} (c{d['comp']*1000:3.0f} io{d['io']*1000:3.0f} n{d['net']*1000:3.0f} h{d['hit']:.2f})")
        print(f"{n:>5} | " + " | ".join(f"{r:>28}" for r in row))
if __name__ == "__main__": main()
