#!/usr/bin/env python3
"""Writes the offsets file for the expert read pool: 'FILE <path>' then '<tensor name> <absolute file offset> <bytes per expert>' for every *_exps.weight tensor of a GGUF.
usage: gguf_offsets.py MODEL.gguf OUT.txt        (pure Python GGUF v2/v3 header parser, no dependencies)"""
import os, struct, sys
TYPES = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}
# (block elements, block bytes) per ggml type id for the types that matter
GGML = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 36), 10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210),
        15: (256, 292), 16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110), 22: (256, 82), 23: (256, 136), 24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8), 29: (256, 56), 30: (1, 2), 34: (256, 54), 35: (256, 66), 39: (32, 17)}
def parse(path):
    f = open(path, "rb"); rd = lambda fmt: struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    if f.read(4) != b"GGUF": raise ValueError("not a GGUF file")
    ver = rd("<I"); nt = rd("<Q"); nk = rd("<Q"); align = 32
    def rstr(): n = rd("<Q"); return f.read(n).decode("utf-8", "replace")
    def rval(t):
        if t == 8: return rstr()
        if t == 9:
            et = rd("<I"); n = rd("<Q"); return [rval(et) for _ in range(n)]
        fmt, _ = TYPES[t]; return rd(fmt)
    for _ in range(nk):
        k = rstr(); t = rd("<I"); v = rval(t)
        if k == "general.alignment": align = int(v)
    tens = []
    for _ in range(nt):
        name = rstr(); nd = rd("<I"); dims = [rd("<Q") for _ in range(nd)]; ty = rd("<I"); off = rd("<Q"); tens.append((name, dims, ty, off))
    data = (f.tell() + align - 1) // align * align
    return tens, data
def expert_total(model):
    """Total bytes of all *_exps.weight tensors (the part streamed per token) and the number of experts of the first one."""
    tens, _ = parse(model); tot = 0; ne = 0
    for name, dims, ty, off in tens:
        if not name.endswith("_exps.weight") or len(dims) != 3: continue
        be, bb = GGML[ty]; tot += dims[0] // be * bb * dims[1] * dims[2]; ne = ne or dims[2]
    return tot, ne
def main(model, out):
    tens, data = parse(model); lines = [f"FILE {os.path.abspath(model)}"]
    for name, dims, ty, off in tens:
        if not name.endswith("_exps.weight") or len(dims) != 3: continue
        be, bb = GGML[ty]; row = dims[0] // be * bb; stride = row * dims[1]
        lines.append(f"{name} {data + off} {stride}")
    open(out, "w").write("\n".join(lines) + "\n"); return len(lines) - 1
if __name__ == "__main__": print(main(sys.argv[1], sys.argv[2]), "tensors")
