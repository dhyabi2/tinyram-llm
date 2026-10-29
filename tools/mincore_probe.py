#!/usr/bin/env python3
"""How much of a file is resident in the page cache right now? Maps the file read-only and calls mincore() on it: no page is touched and nothing is read.
usage: mincore_probe.py FILE [--range-mb START LEN] [--json]      (works on macOS and Linux)
This is the cold-state gate: a run only counts as cold if the resident fraction of the model file is small. Without sudo purge it is also how a memory cap is verified."""
import argparse, ctypes, ctypes.util, json, mmap, os

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
PAGE = os.sysconf("SC_PAGE_SIZE")

class _PyBuffer(ctypes.Structure):
    _fields_ = [("buf", ctypes.c_void_p), ("obj", ctypes.py_object), ("len", ctypes.c_ssize_t), ("itemsize", ctypes.c_ssize_t), ("readonly", ctypes.c_int), ("ndim", ctypes.c_int),
                ("format", ctypes.c_char_p), ("shape", ctypes.c_void_p), ("strides", ctypes.c_void_p), ("suboffsets", ctypes.c_void_p), ("internal", ctypes.c_void_p)]

def _address(mm):
    """Start address of a read-only mmap object, obtained through the buffer protocol without touching any page."""
    b = _PyBuffer(); ctypes.pythonapi.PyObject_GetBuffer(ctypes.py_object(mm), ctypes.byref(b), 0)
    addr = b.buf; ctypes.pythonapi.PyBuffer_Release(ctypes.byref(b)); return addr

def resident(path, start=0, length=None):
    """Resident pages of bytes [start, start+length) of the file. Returns pages, resident_pages, fraction, resident_bytes, bytes."""
    size = os.path.getsize(path); start = max(0, min(start, size)); length = (size - start) if length is None else max(0, min(length, size - start))
    if length == 0: return {"pages": 0, "resident_pages": 0, "fraction": 0.0, "resident_bytes": 0, "bytes": 0}
    fd = os.open(path, os.O_RDONLY)
    try:
        off = (start // mmap.ALLOCATIONGRANULARITY) * mmap.ALLOCATIONGRANULARITY; lead = start - off
        mm = mmap.mmap(fd, lead + length, access=mmap.ACCESS_READ, offset=off)
        try:
            n = (lead + length + PAGE - 1) // PAGE; vec = (ctypes.c_char * n)()
            if _libc.mincore(ctypes.c_void_p(_address(mm)), ctypes.c_size_t(lead + length), vec) != 0: raise OSError(ctypes.get_errno(), "mincore failed")
            first = lead // PAGE; pages = n - first; res = sum(1 for i in range(first, n) if vec[i][0] & 1)
        finally: mm.close()
        return {"pages": pages, "resident_pages": res, "fraction": res / pages, "resident_bytes": res * PAGE, "bytes": length}
    finally: os.close(fd)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("file"); ap.add_argument("--range-mb", nargs=2, type=float); ap.add_argument("--json", action="store_true"); a = ap.parse_args()
    r = resident(a.file, *((int(a.range_mb[0] * 2**20), int(a.range_mb[1] * 2**20)) if a.range_mb else (0, None)))
    print(json.dumps(r) if a.json else f"{r['resident_pages']} of {r['pages']} pages resident = {100*r['fraction']:.1f}% ({r['resident_bytes']/2**30:.2f} of {r['bytes']/2**30:.2f} GiB)")
