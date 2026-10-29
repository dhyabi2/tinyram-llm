"""Published-technique family tests, group D: families fam_s2_2 .. fam_s2_9 (second brainstorm screen, ideas 2-9).
Each function re-implements the family's decisive FREE test on the recorded router traces (Qwen3-30B-A3B, 128 experts, top-8, 48 layers; data/traces/*.txt)
or, where the decisive test needs a weight tensor, a kernel benchmark on specific hardware, or a model run, computes the closed-form / trace part and says so.
Traces are a stand-in for the 125B target (512 experts, top-10): cache entries = token-footprints x 384 accesses; 8/16/32 GB machines hold about 2.1/7.6/20.8 footprints.
Weight-tensor probes (families 5, 8, 9) read ONE tensor of a Q8_0 GGUF of a small MoE (path in the environment variable PUBTECH_GGUF) with a pure-python reader; without it they are skipped, never faked.
Pure standard library, deterministic (fixed seeds). Nothing here runs a model."""
import collections, math, os, random, struct, sys
HERE = os.path.dirname(os.path.abspath(__file__)); TOOLS = os.path.abspath(os.path.join(HERE, ".."))
if TOOLS not in sys.path: sys.path.insert(0, TOOLS)
import cachesim, combo_model as cm, combo_trace_sim as cs

TRACES = cs.TRACES
NL, TOP, FOOT, E128 = 48, 8, 384, 128                 # layers, experts per token, accesses of one token, experts per layer of the trace model
SL125, SLICE125 = 480, 2.7e6                          # slices per token and bytes per slice of the 125B target at 4.3 bpw
TOK_BYTES = SL125 * SLICE125                          # 1.296e9 expert bytes per token at 4.3 bpw
COMP_S = 1 / 4.57                                     # decode compute time per token (55 GFLOP/s over 12 GFLOP)
GIB = 1.074e9
CACHE_GIB = {8: 2.5, 16: 9.15, 32: 25.15}             # expert cache on 8/16/32 GB machines
RATE_LO, RATE_HI = 2.38e9, 3.1e9                      # explicit-read disk rates quoted by the screen (bytes/s)

# ------------------------------------------------------------------ shared helpers
_ACC = {}; _HIT = {}
def accesses(name):
    """Flat (layer, expert) access stream of a trace, tokens in order, layers in order, experts in recorded order."""
    if name not in _ACC: _ACC[name] = [(l, e) for tok in cs.tr(name) for l, ids in enumerate(tok) for e in ids]
    return _ACC[name]

def lru_hit(name, entries):
    """Hit rate (cold misses included) of one fully associative LRU of `entries` over the trace."""
    entries = int(entries)
    if (name, entries) not in _HIT: a = accesses(name); _HIT[(name, entries)] = cachesim.lru_hits(a, entries) / len(a)
    return _HIT[(name, entries)]

def mean_hit(footprints):
    return sum(lru_hit(n, footprints * FOOT) for n in TRACES) / len(TRACES)

def footprints_for(ram_gib, bpw, resident_gb=0.0):
    """Token-footprints of the 125B that fit in the expert cache at `bpw` bits per weight (a footprint = 480 slices)."""
    cache_b = max(0.0, CACHE_GIB[ram_gib] * GIB - resident_gb * 1e9)
    return cache_b / (TOK_BYTES * bpw / 4.3)

def tokps(hit, bytes_tok, rate, comp_s=COMP_S):
    """(serial, overlap) tokens/s: serial = 1/(io+compute), overlap = 1/max(io, compute); io = miss bytes / rate."""
    io = (1 - hit) * bytes_tok / rate
    return 1 / (io + comp_s), 1 / max(io, comp_s)

def lever(bpw, ram, rate=RATE_LO, kappa=1.0, extra_flop=0.0, resident_gb=0.0, hit_override=None):
    """Closed model of a byte lever: experts stored at `bpw` bits, kernel speed ratio kappa, extra compute fraction, resident GB taken from the cache."""
    fp = footprints_for(ram, bpw, resident_gb); hit = mean_hit(fp) if hit_override is None else hit_override
    bt = TOK_BYTES * bpw / 4.3; comp = COMP_S * (1 + extra_flop) / kappa; s, o = tokps(hit, bt, rate, comp)
    return dict(footprints=fp, hit=hit, disk_gb=(1 - hit) * bt / 1e9, serial=s, overlap=o)

def gain(a, b): return dict(serial=a["serial"] / b["serial"], overlap=a["overlap"] / b["overlap"])
def r3(x): return round(x, 3)

# ------------------------------------------------------------------ weight-tensor helpers (pure python)
_GT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
def _rd(f, fmt): return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
def _rstr(f): return f.read(_rd(f, "<Q")).decode("utf-8", "replace")
def _rval(f, t):
    if t == 8: return _rstr(f)
    if t == 9:
        et = _rd(f, "<I"); n = _rd(f, "<Q")
        if et in _GT: f.seek(struct.calcsize(_GT[et]) * n, 1)
        else:
            for _ in range(n): _rval(f, et)
        return ("array", n)
    return _rd(f, _GT[t])

def read_gguf(path):
    """Header of a GGUF v2/v3 file: {'meta': {key: value}, 'tensors': {name: (dims, type_id, absolute_offset)}}. Arrays are skipped (kept as ('array', n))."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF": raise ValueError("not a GGUF file")
        ver = _rd(f, "<I"); nt = _rd(f, "<Q"); nkv = _rd(f, "<Q"); meta = {}
        for _ in range(nkv): k = _rstr(f); meta[k] = _rval(f, _rd(f, "<I"))
        infos = []
        for _ in range(nt):
            name = _rstr(f); nd = _rd(f, "<I"); dims = tuple(_rd(f, "<Q") for _ in range(nd)); ty = _rd(f, "<I"); off = _rd(f, "<Q"); infos.append((name, dims, ty, off))
        al = int(meta.get("general.alignment", 32)); pos = f.tell(); start = (pos + al - 1) // al * al
    return dict(version=ver, meta=meta, tensors={n: (d, t, start + o) for n, d, t, o in infos})

Q8_0 = 8
def read_q8_0(path, info, expert, row0, nrows, col0, ncols):
    """Dequantised sub-matrix (list of rows) of a Q8_0 tensor stored as dims = (cols, rows, experts): rows row0.., columns col0.. (multiples of 32)."""
    dims, ty, off = info
    if ty != Q8_0: raise ValueError("tensor is not Q8_0")
    ne0, ne1 = dims[0], dims[1]; rb = ne0 // 32 * 34; out = []
    if col0 % 32 or ncols % 32: raise ValueError("columns must be multiples of 32")
    with open(path, "rb") as f:
        for r in range(row0, row0 + nrows):
            f.seek(off + (expert * ne1 + r) * rb + col0 // 32 * 34); raw = f.read(ncols // 32 * 34); row = []
            for b in range(ncols // 32):
                blk = raw[b * 34:(b + 1) * 34]; d = struct.unpack("<e", blk[:2])[0]; row.extend(d * q for q in struct.unpack("<32b", blk[2:]))
            out.append(row)
    return out

# ---- block quantisers used as stand-ins for the K-quants (not bit-exact llama.cpp formats)
def quant_uniform(vec, block, bits):
    """Asymmetric uniform quantiser, one (min, scale) per block of `block` values, exact scales. Returns the dequantised list."""
    lv = (1 << bits) - 1; out = []
    for i in range(0, len(vec), block):
        b = vec[i:i + block]; lo, hi = min(b), max(b); s = (hi - lo) / lv
        if s == 0: out.extend([lo] * len(b)); continue
        out.extend(lo + min(lv, max(0, round((x - lo) / s))) * s for x in b)
    return out

def quant_q2k_like(vec):
    """2-bit quantiser shaped like Q2_K: 256-value super-blocks, 16 sub-blocks of 16, 4-bit sub-block scale and min relative to a per-super-block scale and min (2.625 bpw)."""
    out = []
    for s0 in range(0, len(vec), 256):
        sb = vec[s0:s0 + 256]; subs = [sb[i:i + 16] for i in range(0, len(sb), 16)]; sc = []; mn = []
        for b in subs: lo, hi = min(0.0, min(b)), max(b); sc.append((hi - lo) / 3); mn.append(-lo)
        d = max(sc) / 15 or 1.0; dm = max(mn) / 15 or 1.0
        for b, s_, m_ in zip(subs, sc, mn):
            qs = max(1, min(15, round(s_ / d))) * d; qm = min(15, round(m_ / dm)) * dm
            out.extend(min(3, max(0, round((x + qm) / qs))) * qs - qm for x in b)
    return out

def quant_q4_like(vec): return quant_uniform(vec, 32, 4)
def rel_err(a, b):
    num = sum((x - y) ** 2 for x, y in zip(a, b)); den = sum(x * x for x in a); return math.sqrt(num / den) if den else 0.0

def jacobi_eigh(A, sweeps=60, tol=1e-12):
    """Eigenvalues (descending) and eigenvectors (rows) of a small symmetric matrix by cyclic Jacobi rotations."""
    n = len(A); a = [row[:] for row in A]; v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(sweeps):
        off = sum(a[i][j] ** 2 for i in range(n) for j in range(n) if i != j)
        if off < tol * max(1e-300, sum(a[i][i] ** 2 for i in range(n))): break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(a[p][q]) < 1e-300: continue
                th = (a[q][q] - a[p][p]) / (2 * a[p][q]); t = (1 if th >= 0 else -1) / (abs(th) + math.sqrt(th * th + 1)); c = 1 / math.sqrt(t * t + 1); s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]; a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]; a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]; v[k][p] = c * vkp - s * vkq; v[k][q] = s * vkp + c * vkq
    order = sorted(range(n), key=lambda i: -a[i][i])
    return [a[i][i] for i in order], [[v[k][i] for k in range(n)] for i in order]

def dot(a, b): return sum(x * y for x, y in zip(a, b))

def shared_basis_analysis(X, ranks, quant, qname="q"):
    """Experts X (list of equal-length vectors) are written as mean + rank-r combination of r shared basis vectors (PCA over the expert axis) + a quantised residual.
    Returns per rank: share of the centred energy captured, rel error of the unquantised low-rank part, and the rel error after quantising the residual with `quant`.
    Also the direct quantisation error and the share of total energy held by the mean."""
    E = len(X); D = len(X[0]); m = [sum(x[i] for x in X) / E for i in range(D)]; C = [[xi - mi for xi, mi in zip(x, m)] for x in X]
    tot = sum(dot(x, x) for x in X); cen = sum(dot(c, c) for c in C); mean_share = E * dot(m, m) / tot
    G = [[dot(C[i], C[j]) if j >= i else 0.0 for j in range(E)] for i in range(E)]
    for i in range(E):
        for j in range(i): G[i][j] = G[j][i]
    lam, vec = jacobi_eigh(G); rmax = max(ranks); U = []
    for j in range(rmax):
        acc = [0.0] * D
        for i in range(E):
            w = vec[j][i]
            if w: acc = [a + w * c for a, c in zip(acc, C[i])]
        nrm = math.sqrt(max(dot(acc, acc), 1e-300)); U.append([a / nrm for a in acc])
    out = {}
    for r in ranks:
        rec_err2 = 0.0; q_err2 = 0.0
        for e in range(E):
            res = C[e][:]
            for j in range(r):
                c = dot(C[e], U[j]); res = [a - c * u for a, u in zip(res, U[j])]
            rec_err2 += dot(res, res); qr = quant(res); q_err2 += sum((a - b) ** 2 for a, b in zip(res, qr))
        out[r] = dict(energy_captured_of_centred=1 - rec_err2 / cen if cen else 1.0, rel_err_lowrank_only=math.sqrt(rec_err2 / tot), rel_err_with_quantised_residual=math.sqrt(q_err2 / tot))
    direct = math.sqrt(sum(sum((a - b) ** 2 for a, b in zip(x, quant(x))) for x in X) / tot)
    return dict(mean_share_of_energy=mean_share, centred_share=cen / tot, eig_top=[l for l in lam[:rmax]], per_rank=out, direct_quant_rel_err=direct)

def dist_sq(x, c): return sum((a - b) ** 2 for a, b in zip(x, c))
def kmeans(blocks, K, iters, seed=0):
    """Lloyd k-means over equal-length blocks; deterministic (seeded random init, empty clusters re-seeded). Returns the centroid list."""
    rng = random.Random(seed); d = len(blocks[0]); cents = [b[:] for b in rng.sample(blocks, K)]
    for _ in range(iters):
        codes = assign(blocks, cents)[0]; sums = [[0.0] * d for _ in range(K)]; cnt = [0] * K
        for b, c in zip(blocks, codes):
            cnt[c] += 1; s = sums[c]
            for i in range(d): s[i] += b[i]
        for k in range(K): cents[k] = [v / cnt[k] for v in sums[k]] if cnt[k] else rng.choice(blocks)[:]
    return cents

def assign(blocks, cents):
    """Nearest centroid per block. Returns (codes, total squared error)."""
    d = len(cents[0]); cn = [sum(v * v for v in c) for c in cents]; codes = []; tot = 0.0
    if d == 8:
        cl = [(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]) for c in cents]
        for x in blocks:
            x0, x1, x2, x3, x4, x5, x6, x7 = x; best = 1e300; bi = 0
            for i, (c0, c1, c2, c3, c4, c5, c6, c7) in enumerate(cl):
                v = cn[i] - 2 * (x0 * c0 + x1 * c1 + x2 * c2 + x3 * c3 + x4 * c4 + x5 * c5 + x6 * c6 + x7 * c7)
                if v < best: best = v; bi = i
            codes.append(bi); tot += dist_sq(x, cents[bi])
    else:
        for x in blocks:
            best = 1e300; bi = 0
            for i, c in enumerate(cents):
                v = cn[i] - 2 * dot(x, c)
                if v < best: best = v; bi = i
            codes.append(bi); tot += dist_sq(x, cents[bi])
    return codes, tot

def vq_eval(train, test, K, iters, stages=1, seed=0):
    """Trains `stages` additive codebooks of K entries (each stage on the previous residual), returns held-out normalised MSE after each stage."""
    tr_res = [b[:] for b in train]; te_res = [b[:] for b in test]; den = sum(v * v for b in test for v in b); out = []
    for s in range(stages):
        cents = kmeans(tr_res, K, iters, seed + s)
        ctr = assign(tr_res, cents)[0]; cte, err = assign(te_res, cents); out.append(err / den)
        tr_res = [[a - b for a, b in zip(x, cents[c])] for x, c in zip(tr_res, ctr)]; te_res = [[a - b for a, b in zip(x, cents[c])] for x, c in zip(te_res, cte)]
    return out

def blocks_of(rows, d=8):
    return [r[i:i + d] for r in rows for i in range(0, len(r) - d + 1, d)]

def output_error_check(W, quant, nx=24, seed=5):
    """Relative output error ||Wx - W'x|| / ||Wx|| over `nx` random Gaussian x next to the weight (Frobenius) error: they agree in expectation for Gaussian x."""
    Wq = [quant(r) for r in W]; rng = random.Random(seed); num = den = 0.0
    for _ in range(nx):
        x = [rng.gauss(0, 1) for _ in range(len(W[0]))]; y = [dot(r, x) for r in W]; yq = [dot(r, x) for r in Wq]
        num += sum((a - b) ** 2 for a, b in zip(y, yq)); den += sum(a * a for a in y)
    fro = math.sqrt(sum((a - b) ** 2 for r, q in zip(W, Wq) for a, b in zip(r, q)) / sum(a * a for r in W for a in r))
    return dict(q2k_like_output_rel_err_gaussian_x=math.sqrt(num / den), q2k_like_weight_rel_err_same_matrix=fro)

def probe_path(): return os.environ.get("PUBTECH_GGUF")
def probe_matrix(layer, tensor, experts, rows, cols):
    """Sub-matrices (list per expert of row lists) of the probe GGUF, or None when no probe file is configured / readable."""
    p = probe_path()
    if not p or not os.path.exists(p): return None
    g = read_gguf(p); name = f"blk.{layer}.{tensor}.weight"
    if name not in g["tensors"]: return None
    return [read_q8_0(p, g["tensors"][name], e, 0, rows, 0, cols) for e in experts], g["tensors"][name][0]

# ================================================================== family 2: vector quantisation with lookup-table dot products
def lut_ops(rows, cols, books=2, block=8, ncode=256):
    """Operation count (multiply or add = 1 op, a MAC = 2) of a dense dot GEMV against a codebook-LUT GEMV on a rows x cols matrix with `books` additive codebooks.
    LUT kernel: per input chunk of `block` values build the table of all `ncode` codeword dot products (ncode*block MACs per book), then per output row one lookup+add per chunk per book."""
    chunks = cols // block; base = 2 * rows * cols; table = books * ncode * block * 2 * chunks; look = books * rows * chunks * 2
    return dict(base=base, table=table, lookups=look, lut=table + look, ratio=base / (table + look))

def lut_ops_shared(rows, cols, books=2, n_share=1, block=8, ncode=256):
    """Like lut_ops, but one table build is shared by `n_share` matrices that read the same input with the same codebook (e.g. gate and up of the 10 chosen experts when the codebook is shared per layer)."""
    chunks = cols // block; base = n_share * 2 * rows * cols; table = books * ncode * block * 2 * chunks; look = n_share * books * rows * chunks * 2
    return dict(base=base, lut=table + look, ratio=base / (table + look))

def min_rows_for_op_ratio(target, books=2, block=8, ncode=256):
    """Smallest number of output rows at which the LUT kernel needs target-times fewer operations than the dense dot (None if the asymptote 2*block/books is below target)."""
    # lut_ops: base = 2RC, lut = books*(2*ncode*C + 2*R*C/block)
    # ratio >= t  <=>  2R >= t*books*(2*ncode + 2R/block)  <=>  R*(2 - 2*t*books/block) >= 2*t*books*ncode
    den = 2 - 2 * target * books / block
    return None if den <= 0 else 2 * target * books * ncode / den

def table_bytes(cols, books=1, block=8, ncode=256, entry=2): return books * ncode * entry * (cols // block)

def vq_bpw(books, ncode=256, block=8, scale_bpw=0.0): return books * math.log2(ncode) / block + scale_bpw

def lut_gemv(codes, cbs, x, block=8):
    """Reference LUT GEMV. codes[r][chunk][book] -> index; cbs[book][index] -> list of `block` floats; weight block = sum of the chosen codewords (additive VQ). Returns (y, counters)."""
    nb = len(cbs); chunks = len(x) // block; tab_macs = 0; looks = 0; tabs = []
    for c in range(chunks):
        xc = x[c * block:(c + 1) * block]; tabs.append([[dot(xc, w) for w in cbs[b]] for b in range(nb)]); tab_macs += nb * len(cbs[0]) * block
    y = []
    for row in codes:
        acc = 0.0
        for c in range(chunks):
            for b in range(nb): acc += tabs[c][b][row[c][b]]; looks += 1
        y.append(acc)
    return y, dict(table_macs=tab_macs, lookups=looks)

def dense_gemv(codes, cbs, x, block=8):
    y = []
    for row in codes:
        w = []
        for chunk in row:
            blk = [0.0] * block
            for b, ix in enumerate(chunk): blk = [a + v for a, v in zip(blk, cbs[b][ix])]
            w.extend(blk)
        y.append(dot(w, x))
    return y

def fam_s2_2():
    out = {}
    # 1) op counts for the expert shapes of the trace model (gate/up: 768 x 2048, down: 2048 x 768) and the screen's claim of ~8x fewer ops
    shapes = {"gate_up_768x2048": (768, 2048), "down_2048x768": (2048, 768)}
    out["op_ratio"] = {f"{sh}_books{b}": r3(lut_ops(r, c, b)["ratio"]) for sh, (r, c) in shapes.items() for b in (1, 2)}
    out["op_ratio_shared_table_gate_up_20_consumers"] = {f"books{b}": r3(lut_ops_shared(768, 2048, b, 20)["ratio"]) for b in (1, 2)}
    dn = lambda b: lut_ops(2048, 768, b); gu = lambda b: lut_ops_shared(768, 2048, b, 2)
    out["op_ratio_whole_expert_gate_up_down_own_tables"] = {f"books{b}": r3((2 * lut_ops(768, 2048, b)["base"] + dn(b)["base"]) / (2 * lut_ops(768, 2048, b)["lut"] + dn(b)["lut"])) for b in (1, 2)}
    out["op_ratio_asymptote_books1"] = 8.0 / 1; out["op_ratio_asymptote_books2"] = 4.0
    out["min_rows_for_2x_fewer_ops"] = {f"books{b}": min_rows_for_op_ratio(2.0, b) for b in (1, 2)}
    out["table_bytes_cols2048_one_book"] = table_bytes(2048); out["table_fits_32KiB_L1_up_to_cols"] = 32768 // 64
    out["bpw"] = {"1x8_index_only": vq_bpw(1), "2x8_index_only": vq_bpw(2), "2x8_with_0.4_scale": vq_bpw(2, scale_bpw=0.4)}
    out["byte_cut_4.3_to_2.4"] = r3(4.3 / 2.4)
    # 2) byte lever on the traces: 4.3 bpw vs 2.4 bpw (kernel ratio 1), then the kernel ratio alone
    thr = {}
    for ram in (8, 16, 32):
        base = lever(4.3, ram); row = {"base_serial": r3(base["serial"]), "base_overlap": r3(base["overlap"]), "base_hit": r3(base["hit"])}
        for kap in (0.7, 1.0, 1.5):
            v = lever(2.4, ram, kappa=kap); g = gain(v, base); row[f"bpw2.4_kernel{kap}"] = dict(hit=r3(v["hit"]), serial=r3(v["serial"]), overlap=r3(v["overlap"]), gain_serial=r3(g["serial"]), gain_overlap=r3(g["overlap"]))
        row["kernel_only_1.5x_at_4.3bpw"] = {k: r3(v) for k, v in gain(lever(4.3, ram, kappa=1.5), base).items()}
        thr[f"{ram}GB"] = row
    out["throughput"] = thr
    out["claimed_8x_ceiling_tokps"] = r3(8 * 4.57)
    # 3) the lookup scheme itself is exact: reference check on a tiny random instance
    rng = random.Random(7); R, C, B = 12, 32, 2; cbs = [[[rng.gauss(0, 1) for _ in range(8)] for _ in range(16)] for _ in range(B)]
    codes = [[[rng.randrange(16) for _ in range(B)] for _ in range(C // 8)] for _ in range(R)]; x = [rng.gauss(0, 1) for _ in range(C)]
    y1, cnt = lut_gemv(codes, cbs, x); y2 = dense_gemv(codes, cbs, x)
    out["lut_vs_dense_max_abs_diff"] = max(abs(a - b) for a, b in zip(y1, y2)); out["lut_counters_tiny"] = cnt
    out["unmeasured"] = "LUT-gather vs int8-dot kernel speed on AVX2 (the family's decisive microbenchmark); 2-bit VQ model quality; the 125B has no fp checkpoint"
    return out

# ================================================================== family 3: persist the KV cache of a finished prompt
def kv_bytes_per_token(layers, kv_heads, head_dim, bytes_per=2): return 2 * layers * kv_heads * head_dim * bytes_per

def fam_s2_3():
    out = {}
    kv30 = kv_bytes_per_token(48, 4, 128); out["kv_bytes_per_token_30B_shape"] = kv30; out["kv_MB_500_tokens_30B_shape"] = r3(500 * kv30 / 1e6)
    out["kv_MB_500_tokens_if_125B_is_2x_3x"] = [r3(500 * kv30 * s / 1e6) for s in (2, 3)]
    out["family_claim_kv_GB_500_tokens"] = [1, 2]; out["claim_over_estimate"] = {f"claim{g}GB_vs_est_x{s}": r3(g * 1e9 / (500 * kv30 * s)) for g in (1, 2) for s in (1, 3)}
    out["kv_bytes_per_token_small_moe_16layers_16kvheads_128dim"] = kv_bytes_per_token(16, 16, 128)
    flop_tok = 12e9; prompt = 500; pre_flop = flop_tok * prompt; comp_decode, comp_prefill = 55e9, 311e9
    out["prefill_compute_s"] = {"at_decode_rate_55": r3(pre_flop / comp_decode), "at_prefill_rate_311": r3(pre_flop / comp_prefill)}
    store = 69e9; out["expert_stream_io_s"] = {"3.1GBps": r3(store / 3.1e9), "2.38GBps": r3(store / 2.38e9)}
    cold = max(pre_flop / comp_prefill, store / 3.1e9); out["cold_ttft_s_best_path_overlap"] = r3(cold); out["cold_ttft_s_serial"] = r3(pre_flop / comp_prefill + store / 3.1e9)
    step = TOK_BYTES / 3.1e9; res = {}
    for s in (1, 2, 3):
        load = prompt * kv30 * s / 3.1e9; hit = load + step; res[f"kv_x{s}"] = dict(load_s=r3(load), hit_ttft_s=r3(hit), ttft_ratio=r3(cold / hit), request_ratio_200_tokens_at_2p18=r3((cold + 200 / 2.18) / (hit + 200 / 2.18)))
    out["hit_case"] = res; out["decode_tokps_change"] = 1.0
    # 3) cross-prompt expert reuse on the traces (does saving the expert set help the next, different prompt?)
    cross = {}
    for n in TRACES:
        acc = accesses(n); others = [a for m in TRACES if m != n for a in accesses(m)]; row = {}
        for fp in (1.0, 2.1, 4.6, 7.5):
            npin = int(fp * FOOT); row[f"{fp}fp"] = dict(cross_prompt=r3(cachesim.pinned_hits(acc, npin, others) / len(acc)), in_sample=r3(cachesim.pinned_hits(acc, npin) / len(acc)))
        cross[n[-5:]] = row
    out["pinned_hit_cross_prompt_vs_in_sample"] = cross
    out["unmeasured"] = "prompt-cache plumbing time/size on a small MoE (needs a model run; stock llama.cpp already ships it); real 125B KV shape; hit rate of real chat workloads"
    return out

# ================================================================== family 4: cross-layer expert prefetch
_PRED = {}
def train_cooc(names):
    """Layer-to-layer co-occurrence counts over the traces `names`: cnt[l][a][b] = tokens where expert a was chosen at layer l and b at layer l+1."""
    cnt = [[[0] * E128 for _ in range(E128)] for _ in range(NL - 1)]
    for n in names:
        for tok in cs.tr(n):
            for l in range(NL - 1):
                rows = cnt[l]; nxt = tok[l + 1]
                for a in tok[l]:
                    r = rows[a]
                    for b in nxt: r[b] += 1
    return cnt

def predict_next(cnt, l, chosen, m):
    sc = [sum(c) for c in zip(*(cnt[l][a] for a in chosen))]
    return sorted(range(E128), key=lambda b: (-sc[b], b))[:m]

def predictions(name, held_out=True):
    """pred[t][l] = ranked list of 8 experts predicted for layer l+1 from the experts chosen at layer l of token t."""
    key = (name, held_out)
    if key not in _PRED:
        cnt = train_cooc([n for n in TRACES if n != name] if held_out else [name]); T = cs.tr(name)
        _PRED[key] = [[predict_next(cnt, l, tok[l], TOP) for l in range(NL - 1)] for tok in T]
    return _PRED[key]

def precision(name, m, held_out=True):
    T = cs.tr(name); P = predictions(name, held_out); h = n = 0
    for t, tok in enumerate(T):
        for l in range(NL - 1): h += len(set(P[t][l][:m]) & set(tok[l + 1])); n += m
    return h / n

def persistence(name):
    T = cs.tr(name); h = n = 0
    for t in range(1, len(T)):
        for l in range(NL): h += len(set(T[t][l]) & set(T[t - 1][l])); n += TOP
    return h / n

def prefetch_sim(name, fp, m, pred=None, oracle=False, rate=RATE_HI, comp_s=COMP_S, tokens=None):
    """Event simulation: per-layer LRU of int(fp*8) entries, one FIFO disk, compute split evenly over layers. Demand reads of layer l and then prefetches for layer l+1
    are queued when layer l's experts become known; a layer computes once all its experts have arrived. Returns tokens/s, bytes read per token, useful prefetches."""
    T = tokens if tokens is not None else cs.tr(name); cap = max(1, int(fp * TOP)); slice_b = TOK_BYTES / FOOT; st = slice_b / rate; cl = comp_s / NL
    caches = [collections.OrderedDict() for _ in range(NL)]; t = 0.0; disk = 0.0; reads = 0; useful = 0; issued = 0
    def put(c, e, ready, pf):
        c[e] = [ready, pf]
        while len(c) > cap: c.popitem(last=False)
    for ti, tok in enumerate(T):
        for l in range(NL):
            c = caches[l]; ready_max = t
            for e in tok[l]:
                if e in c:
                    c.move_to_end(e); ent = c[e]
                    if ent[1]: useful += 1; ent[1] = False
                    r = ent[0]
                else:
                    start = max(disk, t); r = start + st; disk = r; reads += 1; put(c, e, r, False)
                if r > ready_max: ready_max = r
            if m and l + 1 < NL:
                cand = tok[l + 1][:m] if oracle else pred[ti][l][:m]; c2 = caches[l + 1]
                for e in cand:
                    if e in c2: c2.move_to_end(e); continue
                    start = max(disk, t); r = start + st; disk = r; reads += 1; issued += 1; put(c2, e, r, True)
            t = ready_max + cl
    return dict(tokps=len(T) / t, gb_per_token=reads * slice_b / len(T) / 1e9, useful=useful, issued=issued)

def fam_s2_4():
    out = {"precision_held_out": {}, "precision_in_sample": {}, "persistence": {}}
    for n in TRACES:
        k = n[-5:]
        out["precision_held_out"][k] = {f"top{m}": r3(precision(n, m, True)) for m in (4, 8)}
        out["precision_in_sample"][k] = {f"top{m}": r3(precision(n, m, False)) for m in (4, 8)}
        out["persistence"][k] = r3(persistence(n))
    out["random_precision_expected"] = r3(TOP / E128)
    sim = {}
    for fp in (2.1, 4.0, 7.5):
        row = {}
        for n in TRACES:
            P = predictions(n, True); b = prefetch_sim(n, fp, 0)
            cell = {"no_prefetch_tokps": r3(b["tokps"]), "no_prefetch_gb_tok": r3(b["gb_per_token"])}
            for m in (4, 8):
                s = prefetch_sim(n, fp, m, P); cell[f"top{m}"] = dict(speedup=r3(s["tokps"] / b["tokps"]), gb_tok=r3(s["gb_per_token"]), useful_share=r3(s["useful"] / s["issued"]) if s["issued"] else 0.0)
            if fp == 2.1:
                for m in (4, 8): s = prefetch_sim(n, fp, m, oracle=True); cell[f"oracle_top{m}_speedup"] = r3(s["tokps"] / b["tokps"])
            row[n[-5:]] = cell
        sim[f"{fp}fp"] = row
    out["sim"] = sim
    out["layer_lead_ms"] = r3(COMP_S / NL * 1000); out["one_layer_read_ms"] = r3(TOK_BYTES / FOOT * TOP / RATE_HI * 1000)
    out["unmeasured"] = "hidden-state (pre-attention) predictors above 90% accuracy need a real model's router weights; 512-expert precision; Apple-silicon prefetch behaviour"
    return out

# ================================================================== family 5: shared low-rank basis + low-bit residual
def basis_resident_gb(rank, bpw=4.3, layers=NL, slice_bytes=SLICE125):
    """Resident bytes of a rank-`rank` shared basis per layer: `rank` expert-sized matrices at `bpw` bits per layer."""
    return rank * slice_bytes * (bpw / 4.3) * layers / 1e9

def fam_s2_5():
    out = {"claimed_bytes_per_token_GB": 0.36, "implied_residual_bpw_for_claim": r3(0.36e9 / TOK_BYTES * 4.3)}
    out["bytes_ratio_2.0_2.5_3.0bpw"] = [r3(b / 4.3) for b in (2.0, 2.5, 3.0)]; out["compute_ceiling_tokps"] = 4.57; out["claimed_tokps"] = 6.6
    tab = {}
    for ram in (8, 16, 32):
        base = lever(4.3, ram); row = {"base_serial": r3(base["serial"]), "base_overlap": r3(base["overlap"])}
        for bpw in (2.0, 2.5, 3.0):
            for ex in (0.0, 0.25):
                for rk in (0, 4, 16):
                    v = lever(bpw, ram, extra_flop=ex, resident_gb=basis_resident_gb(rk, 4.3)); g = gain(v, base)
                    row[f"resid{bpw}_extraflop{ex}_rank{rk}"] = dict(hit=r3(v["hit"]), gain_serial=r3(g["serial"]), gain_overlap=r3(g["overlap"]))
        tab[f"{ram}GB"] = row
    out["throughput"] = tab; out["basis_resident_GB_rank_4_16_64_at_4.3bpw"] = [r3(basis_resident_gb(r)) for r in (4, 16, 64)]
    pm = probe_matrix(8, "ffn_up_exps", range(64), 32, 1024)
    if pm is None: out["weights_probe"] = {"ran": False, "reason": "no PUBTECH_GGUF Q8_0 file configured"}
    else:
        mats, dims = pm; X = [[v for row in m for v in row] for m in mats]
        out["weights_probe"] = dict(ran=True, tensor="ffn_up_exps layer 8", dims=list(dims), sub_block="32 rows x 1024 cols of each of 64 experts", **shared_basis_analysis(X, [0, 4, 16], quant_q2k_like, "q2k_like"))
        out["weights_probe"]["direct_q4_like_rel_err"] = math.sqrt(sum(sum((a - b) ** 2 for a, b in zip(x, quant_q4_like(x))) for x in X) / sum(dot(x, x) for x in X))
    out["unmeasured"] = "model quality of any shared-basis scheme; 512-expert spectra (the probe uses 64 experts of a small MoE); training-free fitting at 125B scale"
    return out

# ================================================================== family 6: run only ~16 of 48 layers per token
def expected_jaccard_random(universe, k):
    """Exact expected Jaccard index of two independent uniformly random k-subsets of a `universe`-element set."""
    tot = math.comb(universe, k); e = 0.0
    for i in range(0, k + 1):
        p = math.comb(k, i) * math.comb(universe - k, k - i) / tot
        if p: e += p * i / (2 * k - i)
    return e

def layer_jaccard(name, tokens=None):
    """Mean over tokens of the Jaccard index between the expert sets of layer l and l+1, for l = 0..46."""
    T = tokens if tokens is not None else cs.tr(name); out = []
    for l in range(NL - 1):
        s = 0.0
        for tok in T: a, b = set(tok[l]), set(tok[l + 1]); s += len(a & b) / len(a | b)
        out.append(s / len(T))
    return out

def dynamic_skip_stream(name, frac, seed=11):
    """Per-token random choice of `frac` of the layers to skip (the router-chosen-layers variant), as an access stream."""
    rng = random.Random(seed); T = cs.tr(name); n = round(NL * frac); s = []
    for tok in T:
        sk = set(rng.sample(range(NL), n))
        for l in range(NL):
            if l not in sk: s.extend((l, e) for e in tok[l])
    return s

def skip_tokps(name, frac, mode, ram, rate=RATE_HI):
    """Tokens/s (serial, overlap) with a fraction of layers skipped: static = same layers for every token, dynamic = random layers per token. Cache size stays that of the unskipped model."""
    fp = footprints_for(ram, 4.3); cap = int(fp * FOOT); T = cs.tr(name)
    if mode == "static": sk = cs.skipped(frac); s = [(l, e) for tok in T for l in range(NL) if l not in sk for e in tok[l]]
    else: s = dynamic_skip_stream(name, frac)
    miss = len(s) - cachesim.lru_hits(s, cap); per_tok = miss / len(T); bytes_tok = per_tok * (TOK_BYTES / FOOT)
    io = bytes_tok / rate; comp = COMP_S * (1 - frac); return 1 / (io + comp), 1 / max(io, comp)

def fam_s2_6():
    out = {}; chance = expected_jaccard_random(E128, TOP); out["random_expected_jaccard"] = r3(chance)
    jt = {}
    for n in TRACES:
        j = layer_jaccard(n); jt[n[-5:]] = dict(mean=r3(sum(j) / len(j)), max=r3(max(j)), pairs_above_0p3=sum(1 for v in j if v > 0.3), pairs_above_0p1=sum(1 for v in j if v > 0.1), pairs=len(j))
    out["adjacent_layer_jaccard"] = jt
    res = {}
    for ram in (8, 16, 32):
        row = {}
        for frac in (0.25, 0.5, 0.67):
            cell = {"ideal_1_over_kept": r3(1 / (1 - round(NL * frac) / NL))}
            for mode in ("static", "dynamic"):
                base = [skip_tokps(n, 0.0, "static", ram) for n in TRACES]; sk = [skip_tokps(n, frac, mode, ram) for n in TRACES]
                cell[mode] = dict(gain_serial=r3(sum(a[0] / b[0] for a, b in zip(sk, base)) / 3), gain_overlap=r3(sum(a[1] / b[1] for a, b in zip(sk, base)) / 3))
            row[f"skip{int(frac * 100)}"] = cell
        res[f"{ram}GB"] = row
    out["throughput_gain_vs_no_skip"] = res
    cmf = {}
    for ex in (0.75, 0.5, 0.33):
        sc = cm.sc_at(1); c0 = dict(K=10, bits=4.3, lowrank=False, route=False, overlap=False, spec=0, gpu=0, lut=False, exit=1.0)
        cmf[f"keep{ex}"] = {f"{r}GB": r3(cm.tok_s({**c0, "exit": ex}, f"{r} GB", sc) / cm.tok_s(c0, f"{r} GB", sc)) for r in (8, 16, 32)}
    out["analytic_model_exit_gain_likely"] = cmf
    out["unmeasured"] = "perplexity/quality under any layer mask (decisive test; needs a model run with a layer mask), router distillation, KV for skipped layers"
    return out

# ================================================================== family 7: pin the prefill-streamed experts for decode
def pin_vs_lru(name, W, n_per_layer, H, mode="first", tokens=None):
    """Hit rate over tokens [W, W+H) of (a) a static per-layer set of the n most frequent experts of tokens [0, W), (b) a per-layer LRU of n entries warmed on tokens [0, W).
    mode 'cross' learns the static set from the other traces instead. Returns (pin_hit, lru_hit)."""
    T = tokens if tokens is not None else cs.tr(name); end = min(len(T), W + H); L = len(T[0]); lru = [collections.OrderedDict() for _ in range(L)]
    cnt = [collections.Counter() for _ in range(L)]
    srcs = [cs.tr(n) for n in TRACES if n != name] if mode == "cross" else [T[:W]]
    for toks in srcs:
        for tok in toks:
            for l in range(L): cnt[l].update(tok[l])
    pins = [{e for e, _ in sorted(cnt[l].items(), key=lambda kv: (-kv[1], kv[0]))[:n_per_layer]} for l in range(L)]
    hp = hl = tot = 0
    for t in range(end):
        for l in range(L):
            c = lru[l]
            for e in T[t][l]:
                hit = e in c
                if hit: c.move_to_end(e)
                else:
                    c[e] = 1
                    if len(c) > n_per_layer: c.popitem(last=False)
                if t >= W:
                    tot += 1; hl += hit; hp += e in pins[l]
    return hp / tot, hl / tot

def fam_s2_7():
    out = {}; res = {}
    for n in TRACES:
        row = {}
        for W in (64, 128):
            for frac_label, npl in (("15%", 19), ("29%", 37), ("50%", 64)):
                p, l = pin_vs_lru(n, W, npl, 20); row[f"W{W}_{frac_label}"] = dict(pin=r3(p), lru=r3(l), diff=r3(p - l))
        p, l = pin_vs_lru(n, 64, 37, 10 ** 6); row["W64_29%_rest_of_trace"] = dict(pin=r3(p), lru=r3(l), diff=r3(p - l))
        p, l = pin_vs_lru(n, 64, 37, 20, "cross"); row["cross_prompt_29%_W64"] = dict(pin=r3(p), lru=r3(l), diff=r3(p - l))
        res[n[-5:]] = row
    out["by_trace"] = res
    out["max_diff_pin_minus_lru_29pct_W64_20tok"] = max(res[k]["W64_29%"]["diff"] for k in res); out["kill_threshold"] = 0.05
    out["lru_curve_footprints_hit"] = {str(fp): r3(mean_hit(fp)) for fp in (1.0, 2.1, 4.0, 7.5, 20.8)}
    sec = {}
    for ram in (8, 16, 32):
        h = mean_hit(footprints_for(ram, 4.3)); sec[f"{ram}GB"] = dict(footprints=r3(footprints_for(ram, 4.3)), lru_hit=r3(h), serial=r3(tokps(h, TOK_BYTES, RATE_HI)[0]), overlap=r3(tokps(h, TOK_BYTES, RATE_HI)[1]))
    out["best_path_by_ram"] = sec
    out["claimed_gain_vs_baseline_2_4tokps"] = [r3(4.57 / 4), r3(4.57 / 2)]
    out["unmeasured"] = "prefill routing (only decode traces exist here); the 125B working-set size; effect of expert-major prefill order on LRU state"
    return out

# ================================================================== family 8: 8-dim vector quantisation, 256 codewords, ~1.3 bpw
def fam_s2_8():
    out = {"bpw_index_only_256_entries": vq_bpw(1), "bpw_claimed": 1.3, "codebook_overhead_bpw_per_1.5M_weights": round(256 * 8 * 2 * 8 / 1.5e6, 4), "bytes_cut_claimed": r3(4.3 / 1.3), "bytes_cut_index_only": r3(4.3 / 1.0)}
    out["bpw_2x8"] = vq_bpw(2); out["bpw_16bit_index_65536_entries"] = vq_bpw(1, ncode=65536)
    tab = {}
    for ram in (8, 16, 32):
        base = lever(4.3, ram); row = {"base_serial": r3(base["serial"]), "base_overlap": r3(base["overlap"])}
        for bpw in (1.0, 1.3, 2.0, 2.4, 2.625):
            v = lever(bpw, ram); g = gain(v, base); row[f"bpw{bpw}"] = dict(hit=r3(v["hit"]), gain_serial=r3(g["serial"]), gain_overlap=r3(g["overlap"]))
        tab[f"{ram}GB"] = row
    out["throughput"] = tab
    # reference points for a Gaussian source: Shannon rate-distortion bound D = 2^(-2R), and 1-/2-bit Lloyd-Max scalar quantisers
    out["gaussian_reference_normalised_mse"] = {"shannon_1bpw": 0.25, "lloyd_max_1bit": r3(1 - 2 / math.pi), "shannon_2bpw": 0.0625, "lloyd_max_2bit": 0.1175}
    pm = probe_matrix(8, "ffn_down_exps", range(4), 64, 512)
    if pm is None: out["weights_probe"] = {"ran": False, "reason": "no PUBTECH_GGUF Q8_0 file configured"}
    else:
        mats, dims = pm; train = blocks_of(mats[0] + mats[1]); test_rows = mats[2] + mats[3]; test = blocks_of(test_rows)
        mse = vq_eval(train, test, 256, 8, stages=2); flat = [v for r in test_rows for v in r]
        q2 = sum(rel_err(r, quant_q2k_like(r)) ** 2 * sum(v * v for v in r) for r in test_rows) / sum(v * v for v in flat)
        q4 = sum(rel_err(r, quant_q4_like(r)) ** 2 * sum(v * v for v in r) for r in test_rows) / sum(v * v for v in flat)
        out["weights_probe"] = dict(ran=True, tensor="ffn_down_exps layer 8", dims=list(dims), train_blocks=len(train), test_blocks=len(test), vq_1x8_256_rel_err=math.sqrt(mse[0]), vq_2x8_256_rel_err=math.sqrt(mse[1]),
                                    q2k_like_rel_err=math.sqrt(q2), q4_like_rel_err=math.sqrt(q4), **output_error_check(mats[2], quant_q2k_like))
    out["unmeasured"] = "model-level quality (perplexity/KL) of any VQ; real K-quant bit-exact errors (stand-in scalar quantisers are used); k-means with 65,536 entries; fine-tuning recovery"
    return out

# ================================================================== family 9: 2-bit base + 2-bit residual enhancement, top-4 get the enhancement
_BR = {}
def replay_bytes(name, k, enh="first4"):
    """Bytes per access in Q4-slice units: (Q4 LRU of C=k*384 entries, base+enhancement LRU of 2C half-size entries). enh: first4 / last4 / all8 / none."""
    key = (name, k, enh)
    if key in _BR: return _BR[key]
    T = cs.tr(name); C = int(k * FOOT); bq = 1 - lru_hit(name, C); _BR[key] = (bq, scheme_bytes(T, C, enh)); return _BR[key]

def scheme_bytes(T, C, enh):
    """Bytes per access (Q4-slice units) of a base+enhancement layout: an LRU of 2C half-size entries; every chosen expert needs its base, the experts selected by `enh` also their enhancement."""
    sel = {"first4": lambda p: p < 4, "last4": lambda p: p >= 4, "all8": lambda p: True, "none": lambda p: False}[enh]; s = []; n = 0
    for tok in T:
        for l, ids in enumerate(tok):
            for p, e in enumerate(ids):
                n += 1; s.append((l, e, 0))
                if sel(p): s.append((l, e, 1))
    miss = len(s) - cachesim.lru_hits(s, 2 * C); return miss * 0.5 / n

def fam_s2_9():
    out = {}; res = {}
    for k in (2.2, 4.4, 8.0, 2.1, 7.6, 20.8):
        row = {}
        for enh in ("first4", "last4") if k not in (2.2,) else ("first4", "last4", "all8", "none"):
            cells = {}
            for n in TRACES: bq, bs = replay_bytes(n, k, enh); cells[n[-5:]] = dict(q4=r3(bq), scheme=r3(bs), ratio=r3(bq / bs) if bs else None)
            row[enh] = cells
        res[f"k{k}"] = row
    out["bytes_per_access_replay"] = res
    thr = {}
    for ram, k in ((8, 2.1), (16, 7.6), (32, 20.8)):
        rs = [replay_bytes(n, k, "first4") for n in TRACES]; bq = sum(a for a, _ in rs) / 3; bs = sum(b for _, b in rs) / 3
        sq, oq = tokps(1 - bq, TOK_BYTES, RATE_HI); ss, os_ = tokps(1 - bs, TOK_BYTES, RATE_HI)
        thr[f"{ram}GB"] = dict(bytes_ratio=r3(bq / bs), gain_serial=r3(ss / sq), gain_overlap=r3(os_ / oq), q4_serial=r3(sq), q4_overlap=r3(oq))
    out["throughput_at_real_footprints"] = thr
    pos = collections.defaultdict(list); freq = collections.Counter(); persist = collections.defaultdict(list)
    for n in TRACES:
        T = cs.tr(n)
        for tok in T:
            for l, ids in enumerate(tok): freq.update((n, l, e) for e in ids)
        for t, tok in enumerate(T):
            for l, ids in enumerate(tok):
                for p, e in enumerate(ids):
                    pos[p].append(freq[(n, l, e)])
                    if t: persist[p].append(e in T[t - 1][l])
    out["rank_order_check"] = {"mean_expert_hotness_by_position": [r3(sum(pos[p]) / len(pos[p])) for p in range(TOP)], "persistence_by_position": [r3(sum(persist[p]) / len(persist[p])) for p in range(TOP)]}
    pm = probe_matrix(8, "ffn_down_exps", [2, 3], 64, 512)
    if pm is None: out["weights_probe"] = {"ran": False, "reason": "no PUBTECH_GGUF Q8_0 file configured"}
    else:
        mats, dims = pm; rows = mats[0] + mats[1]; den = sum(v * v for r in rows for v in r); e2 = e4 = eb = en = 0.0
        for r in rows:
            b = quant_q2k_like(r); res_ = [a - c for a, c in zip(r, b)]; enh = quant_q2k_like(res_); tot_ = [x + y for x, y in zip(b, enh)]; q4 = quant_q4_like(r)
            e2 += sum((a - c) ** 2 for a, c in zip(r, b)); eb += sum((a - c) ** 2 for a, c in zip(r, tot_)); e4 += sum((a - c) ** 2 for a, c in zip(r, q4))
        out["weights_probe"] = dict(ran=True, tensor="ffn_down_exps layer 8", dims=list(dims), base_q2k_like_rel_err=math.sqrt(e2 / den), base_plus_q2k_like_residual_rel_err=math.sqrt(eb / den), q4_like_rel_err=math.sqrt(e4 / den),
                                    bpw_base=2.625, bpw_base_plus_residual=5.25, bpw_q4_like=5.0)
    out["unmeasured"] = "perplexity of base-only experts (needs per-expert mixed quantisation and a model run); true gate-rank order of the recorded ids; cost of splitting each expert into two smaller reads"
    return out

# ================================================================== registry
FAMILIES = {
    "s2_2": {"idea": "8-dim vector quantisation (2 codebooks of 256, about 2.4 bpw) with lookup-table dot products: fewer bytes and about 8x fewer operations", "run": fam_s2_2, "needs": "hardware",
             "methods": ["T-MAC (Microsoft, EuroSys 2025), LUT mpGEMM on CPU", "T-MAC Microsoft Research blog (Snapdragon X Elite)", "AQLM with CPU Numba LUT kernels", "AQLM HF docs kernel table", "llama.cpp IQ2_XS/IQ2_XXS codebook-grid quants (2.06-2.5 bpw)", "KBVQ-MoE (ICLR 2026), VQ for MoE LLMs", "Shared-codebook VQ with single table lookup for MoE experts on tensor accelerators"]},
    "s2_3": {"idea": "persist the KV cache of a finished prompt to disk keyed by the token-prefix hash; skip prefill on a repeat or shared prefix", "run": fam_s2_3, "needs": "model",
             "methods": ["llama.cpp discussion #18244, restoring KV for a 950k-token prompt", "hearth-llm (PyPI), a caching proxy for llama-server", "DiskLLM (PyPI), session persistence over the slot API", "llama.cpp commit: save and restore KV cache for a single seq id (#6341)", "Tutti: SSD-backed KV cache for long-context serving, arXiv 2605.03375", "py-kvcache: external KV caching for vLLM on NVMe, arXiv 2609.11744", "SGLang-LSM, arXiv 2511.16138"]},
    "s2_4": {"idea": "hide expert I/O under compute by predicting layer l+1's experts from layer l's choice (claimed precision 0.55, 2.3x)", "run": fam_s2_4, "needs": "local",
             "methods": ["ProMoE: proactive caching with learned per-layer predictors", "HOBBIT: mixed-precision expert offloading, built on llama.cpp", "Budgeting Bytes: storage roofline for storage-bound LLM decoding", "Pre-attention expert prediction (arXiv 2511.10676) and Speculating Experts (arXiv 2603.19289)"]},
    "s2_5": {"idea": "one shared low-rank basis per MoE layer kept resident, each expert stored as a 2-3 bit residual (claimed 3.6x fewer bytes)", "run": fam_s2_5, "needs": "model",
             "methods": ["D2-MoE, Delta Decompression for MoE LLMs (ICML 2025)", "MoBE, Mixture-of-Basis-Experts", "MoE-SVD (ICML 2025)", "DeltaMoE, memory-efficient inference for merged MoE with delta compression (ICLR 2025 workshop)", "Bandwidth-efficient adaptive MoE via low-rank compensation"]},
    "s2_6": {"idea": "run each token through only about 16 of 48 layers chosen by a small distilled router (claimed 3.3x)", "run": fam_s2_6, "needs": "model",
             "methods": ["Dr.LLM: Dynamic Layer Routing in LLMs (Heakl et al., ICLR 2026)", "LayerSkip (Meta, ACL 2024)", "ShortGPT (Men et al. 2024)", "Early-exit analysis for MoE models (secondary overview article that came up in the search)"]},
    "s2_7": {"idea": "keep the experts streamed during expert-major prefill pinned in RAM so decode reads RAM (claimed 3x)", "run": fam_s2_7, "needs": "local",
             "methods": ["MoE-Infinity, arXiv 2401.14361"]},
    "s2_8": {"idea": "vector quantisation of expert weights: one 8-bit index per 8 weights into a 256-entry codebook, claimed 1.3 bpw (3.3x fewer bytes)", "run": fam_s2_8, "needs": "model",
             "methods": ["KBVQ-MoE, vector quantisation for MoE (ICLR 2026)", "QMoE sub-1-bit compression (Frantar and Alistarh)", "MoQE, 2-bit expert-only quantisation (Microsoft)", "mixtral-offloading (Eliseev and Mazur): 2-bit HQQ experts plus LRU", "VPTQ (Microsoft), PyPI page", "GPTVQ", "Mixture Compressor for MoE LLMs"]},
    "s2_9": {"idea": "every expert stored as a 2-bit base plus a 2-bit residual; only the router's top-4 experts per layer load the enhancement (claimed 2.4x fewer bytes)", "run": fam_s2_9, "needs": "model",
             "methods": ["D2MoE: nested MWQ expert weights", "Matryoshka Quantization / MatGPTQ", "DynaExq and GEMQ per-expert mixed precision"]},
}

if __name__ == "__main__":
    import json
    keys = sys.argv[1:] or list(FAMILIES)
    for k in keys: print(k, json.dumps(FAMILIES[k]["run"](), indent=1, default=str))
