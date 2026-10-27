#!/usr/bin/env python3
"""Reference implementations + self-tests for the three runtime patches (numpy, no ggml).

These are NOT the ggml/C++ patches. They are the executable specification and correctness oracle each patch must match:
  P-A  gated delta rule (Gated DeltaNet recurrence)             -> test: associative recall, state size independent of length
  P-B  block-sparse attention (sink + local + top-k blocks)     -> test: equals masked dense attention; equals dense when k covers all blocks (gate G6)
  P-C  expert pool: byte-capacity LRU over an extent file       -> test: LRU order, byte accounting, crc check, 4 KiB-aligned reads, hit rate agrees with the simulator
"""
import numpy as np, os, struct, zlib, tempfile, collections, time

rng = np.random.default_rng(7)
RESULTS = []
def check(name, ok, detail=""):
    RESULTS.append(ok); print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))

# ======================================================================================================
# P-A  Gated delta rule:  S_t = a_t * S_{t-1} (I - b_t k_t k_t^T) + b_t v_t k_t^T ;  o_t = S_t q_t
# ======================================================================================================
def gated_delta_rule(q, k, v, a, b, S0=None):
    """q,k: [T,dk]  v: [T,dv]  a,b: [T] (gate alpha in (0,1], write strength beta in [0,1]).  Returns outputs [T,dv] and final state [dv,dk].
    `a` here is the already-exponentiated gate; the ggml op receives g RAW (a = exp(g)) per the runtime-config stage."""
    T, dk = k.shape; dv = v.shape[1]
    S = np.zeros((dv, dk), dtype=np.float32) if S0 is None else S0.copy()
    out = np.empty((T, dv), dtype=np.float32)
    for t in range(T):
        kt = k[t]
        S = a[t] * (S - b[t] * np.outer(S @ kt, kt)) + b[t] * np.outer(v[t], kt)
        out[t] = S @ q[t]
    return out, S

dk = dv = 128
keys = np.linalg.qr(rng.standard_normal((dk, dk)))[0][:, :16].T.astype(np.float32)        # 16 orthonormal keys
vals = rng.standard_normal((16, dv)).astype(np.float32)
T = 16
_, S = gated_delta_rule(keys, keys, vals, np.ones(T, np.float32), np.ones(T, np.float32))
recall = max(np.abs(S @ keys[j] - vals[j]).max() for j in range(16))
check("P-A associative recall: with a=1, b=1 and orthonormal keys, S k_j == v_j", recall < 1e-4, f"max err {recall:.2e}")

def unit(x): return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
def rnd(n, d=128): return rng.standard_normal((n, d)).astype(np.float32)
_, S_short = gated_delta_rule(rnd(8), unit(rnd(8)), rnd(8), np.full(8, .9, np.float32), np.full(8, .5, np.float32))
_, S_long = gated_delta_rule(rnd(600), unit(rnd(600)), rnd(600), np.full(600, .9, np.float32), np.full(600, .5, np.float32))
state_bytes = 36 * 32 * 128 * 128 * 4
check("P-A state size is independent of sequence length (8 vs 600 tokens)", S_short.shape == S_long.shape == (128, 128) and np.isfinite(S_long).all(), f"{state_bytes} B total for 36 layers x 32 heads fp32 = {state_bytes/1e6:.1f} MB")

Kd, Vd, Qd = unit(rnd(200)), rnd(200), rnd(200)
Sd = gated_delta_rule(Qd, Kd, Vd, np.full(200, .5, np.float32), np.full(200, .5, np.float32))[1]
Sl = gated_delta_rule(Qd, Kd, Vd, np.full(200, 1.0, np.float32), np.full(200, .5, np.float32))[1]
check("P-A gate decays the state (alpha=0.5 has a smaller norm than alpha=1)", np.linalg.norm(Sd) < np.linalg.norm(Sl), f"{np.linalg.norm(Sd):.1f} < {np.linalg.norm(Sl):.1f}")

with np.errstate(all="ignore"):
    Sx = gated_delta_rule(rnd(200), rnd(200), rnd(200), np.full(200, 1.0, np.float32), np.full(200, .5, np.float32))[1]
check("P-A the recurrence DIVERGES unless keys are L2-normalised: the kernel must normalise k", not np.isfinite(Sx).all() or np.abs(Sx).max() > 1e6, f"max |S| with raw keys {np.abs(Sx).max():.2e}")
# fp32 vs bf16-rounded state drift over many steps: why the stage keeps the state in fp32
def bf16(x): return (x.view(np.uint32) & 0xFFFF0000).view(np.float32)
q_ = rng.standard_normal((2000, 64)).astype(np.float32); k_ = rng.standard_normal((2000, 64)).astype(np.float32); k_ /= np.linalg.norm(k_, axis=1, keepdims=True); v_ = rng.standard_normal((2000, 64)).astype(np.float32)
o32, _ = gated_delta_rule(q_, k_, v_, np.full(2000, .99, np.float32), np.full(2000, .3, np.float32))
S = np.zeros((64, 64), np.float32); o16 = np.empty_like(o32)
for t in range(2000):
    S = bf16(.99 * (S - .3 * np.outer(S @ k_[t], k_[t])) + .3 * np.outer(v_[t], k_[t])); o16[t] = S @ q_[t]
drift = np.abs(o16 - o32).mean() / np.abs(o32).mean()
check("P-A bf16 state storage drifts measurably from fp32 over 2,000 steps (supports the fp32 decision)", drift > 1e-3, f"mean relative drift {drift:.3%}")

# ======================================================================================================
# P-B  Block-sparse attention.  64-token KV blocks, selection by mean-key score, + 1 sink block + 2 local blocks.
# ======================================================================================================
BS = 64
def block_select(q, K, n_sel, qpos, n_local=2):
    """Which KV blocks the query at absolute position qpos attends to (causal)."""
    nb = (qpos // BS) + 1
    chosen = {0} | set(range(max(0, nb - n_local), nb))                 # sink + local
    if nb > len(chosen):
        mean_k = np.stack([K[i * BS:min((i + 1) * BS, qpos + 1)].mean(0) for i in range(nb)])
        score = mean_k @ q
        for i in np.argsort(-score):
            if len(chosen) >= n_sel + 3: break
            chosen.add(int(i))
    return sorted(chosen)

def attn_masked_dense(Q, K, V, n_sel):
    """Oracle: dense softmax attention with an explicit [T,T] mask built from block_select."""
    T, d = Q.shape; out = np.zeros_like(V)
    for t in range(T):
        keep = np.zeros(t + 1, bool)
        for b_ in block_select(Q[t], K, n_sel, t): keep[b_ * BS:min((b_ + 1) * BS, t + 1)] = True
        s = (K[:t + 1] @ Q[t]) / np.sqrt(d); s[~keep] = -np.inf
        p = np.exp(s - s.max()); out[t] = (p / p.sum()) @ V[:t + 1]
    return out

def attn_block_sparse(Q, K, V, n_sel):
    """Implementation under test: gathers ONLY the selected blocks and runs a single-pass ONLINE softmax in ascending block order.
    No [T,T] mask is ever materialised."""
    T, d = Q.shape; out = np.zeros_like(V)
    for t in range(T):
        m, l, acc = -np.inf, 0.0, np.zeros(V.shape[1], np.float64)
        for b_ in block_select(Q[t], K, n_sel, t):
            lo, hi = b_ * BS, min((b_ + 1) * BS, t + 1)
            s = (K[lo:hi] @ Q[t]) / np.sqrt(d)
            m_new = max(m, s.max()); scale = np.exp(m - m_new) if np.isfinite(m) else 0.0
            p = np.exp(s - m_new); acc = acc * scale + p @ V[lo:hi]; l = l * scale + p.sum(); m = m_new
        out[t] = acc / l
    return out

def attn_dense_causal(Q, K, V):
    T, d = Q.shape; out = np.zeros_like(V)
    for t in range(T):
        s = (K[:t + 1] @ Q[t]) / np.sqrt(d); p = np.exp(s - s.max()); out[t] = (p / p.sum()) @ V[:t + 1]
    return out

T, d = 640, 64
Q, K, V = (rng.standard_normal((T, d)).astype(np.float64) for _ in range(3))
sparse, masked = attn_block_sparse(Q, K, V, n_sel=3), attn_masked_dense(Q, K, V, n_sel=3)
check("P-B online-softmax block-sparse == explicitly masked dense (gate G6 harness)", np.abs(sparse - masked).max() < 1e-9, f"max |diff| {np.abs(sparse - masked).max():.2e}")
full = attn_block_sparse(Q, K, V, n_sel=10_000)
check("P-B selecting every block reproduces full causal attention", np.abs(full - attn_dense_causal(Q, K, V)).max() < 1e-9, f"max |diff| {np.abs(full - attn_dense_causal(Q, K, V)).max():.2e}")
touched = np.mean([sum(min((b_ + 1) * BS, t + 1) - b_ * BS for b_ in block_select(Q[t], K, 3, t)) for t in range(T)])
check("P-B partial selection reads fewer KV rows than dense", touched < (T + 1) / 2, f"mean rows read per query {touched:.0f} vs dense mean {(T + 1) / 2:.0f}")

# ======================================================================================================
# P-C  Expert pool: byte-capacity LRU over an extent file, O_DIRECT-style 4 KiB-aligned reads (pread; io_uring is a drop-in)
# ======================================================================================================
ALIGN = 4096
class ExpertPool:
    def __init__(self, path, manifest, capacity_bytes):
        self.fd = os.open(path, os.O_RDONLY); self.man = manifest; self.cap = capacity_bytes
        self.lru = collections.OrderedDict(); self.used = 0
        self.hits = self.misses = self.bytes_read = 0
    def get(self, layer, expert):
        key = (layer, expert)
        if key in self.lru: self.lru.move_to_end(key); self.hits += 1; return self.lru[key]
        off, length, crc = self.man[key]
        assert off % ALIGN == 0, "extent not 4 KiB aligned"
        buf = os.pread(self.fd, length, off); self.bytes_read += length
        if zlib.crc32(buf) != crc: raise IOError(f"crc mismatch for expert {key}")
        self.misses += 1
        while self.used + length > self.cap and self.lru:
            _, ev = self.lru.popitem(last=False); self.used -= len(ev)
        self.lru[key] = buf; self.used += length
        return buf

# build a small extent file: 8 layers x 64 experts with tier-dependent sizes (0.98..2.63 MB scaled down 100x)
L_, E_ = 8, 64
sizes = {t: int(mb * 1e4) for t, mb in enumerate([2.63, 2.13, 1.28, 1.08, 0.97])}
tier_of = lambda l, e: min(int((e / E_) ** 0.8 * 5), 4)
tmp = tempfile.mkdtemp(); path = os.path.join(tmp, "experts.bin"); man = {}; off = 0
with open(path, "wb") as f:
    for l in range(L_):
        for e in range(E_):
            n = sizes[tier_of(l, e)]; payload = rng.integers(0, 256, n, dtype=np.uint8).tobytes()
            f.seek(off); f.write(payload); man[(l, e)] = (off, n, zlib.crc32(payload))
            off += (n + ALIGN - 1) // ALIGN * ALIGN
f_size = off
pool = ExpertPool(path, man, capacity_bytes=int(sum(m[1] for m in man.values()) * 0.25))
a = pool.get(0, 0); b = pool.get(0, 0)
check("P-C second access is a hit and returns identical bytes", pool.hits == 1 and a == b)
pool2 = ExpertPool(path, man, capacity_bytes=sizes[0] * 2 + 10)
pool2.get(1, 0); pool2.get(1, 1); pool2.get(1, 2)                    # capacity holds ~2 of the largest: (1,0) must be evicted
check("P-C evicts the least recently used expert when the byte budget is exceeded", (1, 0) not in pool2.lru and (1, 2) in pool2.lru, f"used {pool2.used} <= cap {pool2.cap}")
check("P-C never exceeds its byte capacity", pool2.used <= pool2.cap)
import shutil
bad_path = path + ".corrupt"; shutil.copy(path, bad_path)
bad = open(bad_path, "r+b"); o, n, c = man[(2, 3)]; bad.seek(o + 5); bad.write(b"\xff\xff"); bad.close()
try: ExpertPool(bad_path, man, 1 << 20).get(2, 3); detected = False
except IOError: detected = True
check("P-C detects a corrupted extent by crc32", detected)
check("P-C every extent starts on a 4 KiB boundary", all(o % ALIGN == 0 for o, _, _ in man.values()))

# cross-check against the simulator: replay a persistence trace through the real pool and through a plain item-LRU
def trace(n_tok, persist, seed=3):
    r = np.random.default_rng(seed); w = 1 / np.arange(1, E_ + 1) ** .8; w /= w.sum(); sel = [r.choice(E_, 6, replace=False, p=w) for _ in range(L_)]; out = []
    for _ in range(n_tok):
        for l in range(L_):
            keep = r.random(6) < persist; kept = sel[l][keep]; need = 6 - len(kept)
            if need:
                pool_ = np.setdiff1d(np.arange(E_), kept); pp = w[pool_] / w[pool_].sum(); sel[l] = np.concatenate([kept, r.choice(pool_, need, replace=False, p=pp)])
            out += [(l, int(e)) for e in sel[l]]
    return out
ids = trace(300, 0.7)
big = ExpertPool(path, man, capacity_bytes=int(sum(m[1] for m in man.values()) * 0.25)); item = collections.OrderedDict(); ih = 0
cap_items = 128
for i, k in enumerate(ids):
    big.get(*k)
    if k in item: item.move_to_end(k); ih += 1
    else:
        item[k] = 1
        if len(item) > cap_items: item.popitem(last=False)
hr_pool = big.hits / (big.hits + big.misses); hr_item = ih / len(ids)
check("P-C byte-capacity LRU hit rate is close to the item-count LRU of the simulator at equal capacity", abs(hr_pool - hr_item) < 0.10, f"pool {hr_pool:.3f} vs item-LRU {hr_item:.3f}; bytes read {big.bytes_read/1e6:.1f} MB for {len(ids)} accesses")

print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
raise SystemExit(0 if all(RESULTS) else 1)
