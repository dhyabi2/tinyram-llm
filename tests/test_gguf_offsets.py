"""gguf_offsets: the GGUF header parser behind the expert read pool, on synthetic files built here (known layout) and, when TINY is set, on a real model file."""
import os, struct, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "tools")); from harness import T
import gguf_offsets as go
def s(x): b = x.encode(); return struct.pack("<Q", len(b)) + b
def build(path, tensors, align=32, extra_kv=()):
    kvs = [("general.architecture", 8, s("test")), *extra_kv]
    hdr = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs))
    for k, t, v in kvs: hdr += s(k) + struct.pack("<I", t) + v
    off = 0; infos = b""; datas = []
    for name, dims, ty, nbytes in tensors:
        infos += s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", ty, off)
        datas.append((off, nbytes)); off += (nbytes + align - 1) // align * align
    body = hdr + infos; pad = (-len(body)) % align; body += b"\0" * pad; start = len(body)
    with open(path, "wb") as f:
        f.write(body)
        for (o, n), (name, *_ ) in zip(datas, tensors): f.write(b"\xAB" * n + b"\0" * ((-n) % align))
    return start, datas
def run():
    t = T("T13 gguf_offsets"); d = tempfile.mkdtemp()
    for ty, (be, bb) in ((12, (256, 144)), (14, (256, 210)), (8, (32, 34)), (1, (1, 2)), (0, (1, 4))):
        for ne0, ne1, ne2 in ((256, 4, 3), (512, 8, 5), (1024, 2, 7)):
            if ne0 % be: continue
            nb = ne0 // be * bb * ne1 * ne2; p = os.path.join(d, f"s_{ty}_{ne0}.gguf")
            start, datas = build(p, [("blk.0.attn_q.weight", [ne0, ne1], ty, ne0 // be * bb * ne1), ("blk.0.ffn_down_exps.weight", [ne0, ne1, ne2], ty, nb), ("blk.1.ffn_up_exps.weight", [ne0, ne1, ne2], ty, nb)])
            out = os.path.join(d, "o.txt"); n = go.main(p, out); lines = open(out).read().splitlines()
            t.check(f"type {ty} dims {ne0}x{ne1}x{ne2}: two expert tensors listed, the attention tensor skipped", n == 2 and len(lines) == 3 and lines[0].startswith("FILE "))
            row = ne0 // be * bb
            for (name, off_i), ln in zip(((("blk.0.ffn_down_exps.weight"), datas[1][0]), (("blk.1.ffn_up_exps.weight"), datas[2][0])), lines[1:]):
                nm, off, stride = ln.split(); t.check(f"type {ty} {ne0}: {nm} stride = row bytes x rows", nm == name and int(stride) == row * ne1)
                t.check(f"type {ty} {ne0}: {nm} offset = data start + tensor offset", int(off) == start + off_i)
                with open(p, "rb") as f:
                    f.seek(int(off)); chunk = f.read(int(stride) * ne2)
                t.check(f"type {ty} {ne0}: the stride x experts bytes at the offset are the tensor's data (marker bytes)", chunk == b"\xAB" * (int(stride) * ne2))
    p = os.path.join(d, "align64.gguf"); start, datas = build(p, [("blk.0.ffn_down_exps.weight", [256, 4, 3], 12, 144 * 4 * 3)], align=64, extra_kv=[("general.alignment", 4, struct.pack("<I", 64))])
    out = os.path.join(d, "o64.txt"); go.main(p, out); off = int(open(out).read().splitlines()[1].split()[1]); t.check("general.alignment=64 is honoured", off == start and off % 64 == 0)
    open(os.path.join(d, "bad.gguf"), "wb").write(b"NOPE" + b"\0" * 40)
    try: go.main(os.path.join(d, "bad.gguf"), out); ok = False
    except ValueError: ok = True
    t.check("a non-GGUF file raises ValueError", ok)
    real = os.environ.get("TINY", "")
    if real and os.path.exists(real):
        out = os.path.join(d, "real.txt"); n = go.main(real, out); size = os.path.getsize(real); lines = open(out).read().splitlines()[1:]
        t.check("real model: at least one expert tensor found", n >= 1)
        t.check("real model: every expert tensor lies inside the file", all(int(l.split()[1]) < size for l in lines))
        t.check("real model: offsets are multiples of the 32-byte alignment", all(int(l.split()[1]) % 32 == 0 for l in lines))
    return t
if __name__ == "__main__": sys.exit(0 if run().summary() else 1)
