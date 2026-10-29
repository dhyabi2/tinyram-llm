"""Parse the PHASE lines printed with GGML_PHASE=1 (see patches/llama-phase-timing.patch in the planning repo) into the phase fields of result schema v3.
A line looks like: PHASE decode n_tokens=<N> start=<seconds> end=<seconds>. Other text on the same line or around it (generated tokens share the stream) is ignored.
Prompt phase = the leading calls with n_tokens > 1; decode phase = the calls with n_tokens == 1 that follow. A warmup call (1 token before the prompt) is reported separately."""
import re
_RX = re.compile(r"PHASE decode n_tokens=(\d+) start=([0-9]*\.?[0-9]+) end=([0-9]*\.?[0-9]+)")

def calls(text):
    out = []
    for m in _RX.finditer(text):
        n, a, b = int(m.group(1)), float(m.group(2)), float(m.group(3))
        if b >= a and n > 0: out.append((n, a, b))
    return out

def summarize(text):
    """Returns None when no valid PHASE line exists, else a dict of seconds and counts."""
    c = calls(text)
    if not c: return None
    # leading single-token calls before any multi-token call are warmup
    i = 0
    while i < len(c) and c[i][0] == 1 and any(x[0] > 1 for x in c[i + 1:]): i += 1
    warm = c[:i]; rest = c[i:]
    j = 0
    while j < len(rest) and rest[j][0] > 1: j += 1
    prompt, dec = rest[:j], rest[j:]
    dec1 = [x for x in dec if x[0] == 1]
    t0 = c[0][1]; t_end = c[-1][2]
    s = lambda L: sum(b - a for _, a, b in L)
    return {"warmup_s": s(warm), "warmup_calls": len(warm), "prompt_s": s(prompt), "prompt_tokens": sum(n for n, _, _ in prompt), "prompt_calls": len(prompt),
            "decode_s": s(dec1), "decode_calls": len(dec1), "ttft_s": (prompt[-1][2] - t0) if prompt else None, "total_s": t_end - t0,
            "gap_s": max(0.0, (t_end - t0) - s(c))}
