#!/usr/bin/env python3
"""make_layer_quant_map.py - per-LAYER expert quantization map for llama-quantize (replaces the per-expert bit-width writer).

Why per layer: in llama.cpp the experts of a layer are ONE 3-D tensor (blk.N.ffn_gate_up_exps.weight, blk.N.ffn_down_exps.weight),
and a tensor has one ggml type. Per-expert types are not expressible; per-layer types are, with stock llama-quantize
(--tensor-type-file). This tool spends a byte budget across layers where it buys the most accuracy.

Model: quantization noise power of a layer at b bits per weight is proportional to 4^-b. Layer i has sensitivity s_i (how much one
unit of noise in its experts moves the output) and P_i expert parameters. Starting every layer at the lowest type, the tool repeatedly
upgrades the layer whose next step removes the most noise per extra byte, until the budget is spent. It is a greedy allocation under
that stated model; it is not a proof of optimality and the model is an approximation.

Sensitivity input, best to worst:
  --sensitivity file.json   measured per-layer score, e.g. the KL increase when ONLY that layer is quantized low (see layer_sweep_cmds)
  --freq expert_freq.json   fallback heuristic from profile_experts.py: effective number of experts used per layer (exp of routing entropy).
                            This is a heuristic, not a measurement, and the report says so.

usage: make_layer_quant_map.py --params params.json --budget-gb 30.5 [--sensitivity s.json | --freq f.json] --out layer_types.txt
           [--ladder IQ1_S:1.5625,IQ2_XXS:2.0625,...] [--report report.json]
params.json : {"0": <expert parameters in layer 0, all expert tensors>, "1": ...}
Pass the output to:  llama-quantize --imatrix imat.dat --tensor-type-file layer_types.txt in.gguf out.gguf <BASE>
BASE must be a QUANTIZED type (llama-quantize ignores --tensor-type when the base type is not quantized). IQ1/IQ2 types need --imatrix.
"""
import argparse, heapq, json, math, sys

DEFAULT_LADDER = [("IQ1_S", 1.5625), ("IQ1_M", 1.75), ("IQ2_XXS", 2.0625), ("Q2_K", 2.625), ("Q3_K_S", 3.4375),
                  ("IQ4_XS", 4.25), ("Q4_K", 4.5), ("Q6_K", 6.5625), ("Q8_0", 8.5)]
EXPERT_RULE = r"^blk\.{layer}\.ffn_(gate_up|gate|up|down)_exps\.weight"
# Written FIRST: llama-quantize applies the first matching rule. qwen4_exp stores per-head norm scales as 2-D tensors (ple_norm_key,
# ple_norm_query, ple_norm_conv, shape [256, 4]) that the quantizer's own "skip norms" rule misses; the graph multiplies by them
# element-wise, and a quantized one aborts at run time ("binary_op: unsupported types: f32, q8_0"). Found by running a quantized model.
DEFAULT_PINS = [("norm", "f32")]


def parse_ladder(text):
    out = []
    for item in text.split(","):
        name, _, bpw = item.partition(":")
        if not name or not bpw:
            sys.exit(f"bad ladder entry '{item}' (want NAME:bits_per_weight)")
        out.append((name.strip(), float(bpw)))
    out.sort(key=lambda t: t[1])
    if len(out) < 2:
        sys.exit("the ladder needs at least two types")
    return out


def effective_experts(counts):
    """exp(entropy) of one layer's routing counts: the number of experts that really share the traffic."""
    total = float(sum(counts.values()))
    if total <= 0:
        return 1.0
    return math.exp(-sum((c / total) * math.log(c / total) for c in counts.values() if c > 0))


def allocate(params, sens, ladder, budget_bytes):
    layers = sorted(params, key=int)
    level = {l: 0 for l in layers}
    spent = sum(params[l] * ladder[0][1] / 8 for l in layers)
    if spent > budget_bytes:
        sys.exit(f"budget {budget_bytes / 1e9:.2f} GB is below the minimum {spent / 1e9:.2f} GB (every layer at {ladder[0][0]})")
    noise = lambda b: 4.0 ** (-b)

    def step(l):
        k = level[l]
        if k + 1 >= len(ladder):
            return None
        extra = params[l] * (ladder[k + 1][1] - ladder[k][1]) / 8
        gain = sens[l] * params[l] * (noise(ladder[k][1]) - noise(ladder[k + 1][1]))
        return gain / extra, extra

    heap = [(-s[0], l) for l in layers if (s := step(l))]
    heapq.heapify(heap)
    while heap:
        _, l = heapq.heappop(heap)
        s = step(l)
        if not s:
            continue
        if spent + s[1] > budget_bytes:
            continue                                   # cannot afford this upgrade; cheaper ones may still fit
        spent += s[1]
        level[l] += 1
        if (nxt := step(l)):
            heapq.heappush(heap, (-nxt[0], l))
    return {l: ladder[level[l]] for l in layers}, spent


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True); ap.add_argument("--budget-gb", type=float, required=True)
    ap.add_argument("--sensitivity"); ap.add_argument("--freq"); ap.add_argument("--ladder")
    ap.add_argument("--out", required=True); ap.add_argument("--report")
    ap.add_argument("--pin", action="append", default=None, help="REGEX=TYPE kept out of quantization (repeatable; default: norm=f32)")
    a = ap.parse_args(argv)

    params = {str(k): float(v) for k, v in json.load(open(a.params)).items()}
    ladder = parse_ladder(a.ladder) if a.ladder else DEFAULT_LADDER
    if a.sensitivity:
        raw = json.load(open(a.sensitivity)); source = "measured sensitivity (--sensitivity)"
        sens = {l: float(raw[l]) for l in params}
    elif a.freq:
        counts = json.load(open(a.freq))["counts"]; source = "HEURISTIC: effective experts used per layer (--freq); measure sensitivity to replace this"
        sens = {l: effective_experts({int(e): c for e, c in counts[l].items()}) for l in params}
    else:
        source = "WARNING: no sensitivity given, all layers weighted equally"; sens = {l: 1.0 for l in params}
        print("WARNING: no --sensitivity or --freq: every layer is weighted equally, so the allocation is uniform.", file=sys.stderr)
    if any(v < 0 for v in sens.values()):
        sys.exit("sensitivities must be non-negative")

    chosen, spent = allocate(params, sens, ladder, a.budget_gb * 1e9)
    pins = [tuple(x.split("=", 1)) for x in a.pin] if a.pin else DEFAULT_PINS
    with open(a.out, "w") as f:
        for rx, ty in pins:
            f.write(f"{rx}={ty}\n")
        for l, (name, _) in chosen.items():
            f.write(f"{EXPERT_RULE.format(layer=l)}={name}\n")
    total_p = sum(params.values())
    report = {"sensitivity_source": source, "budget_gb": a.budget_gb, "predicted_expert_bytes": spent, "predicted_expert_gb": round(spent / 1e9, 6), "pins": [list(p) for p in pins],
              "average_bpw": round(spent * 8 / total_p, 4), "ladder": [list(t) for t in ladder],
              "layers": {l: {"type": n, "bpw": b, "params": params[l], "sensitivity": sens[l], "gb": round(params[l] * b / 8 / 1e9, 5)} for l, (n, b) in chosen.items()}}
    if a.report:
        json.dump(report, open(a.report, "w"), indent=1)
    print(f"{len(chosen)} layers, {total_p / 1e9:.3f}B expert parameters: predicted {spent / 1e9:.3f} GB ({report['average_bpw']} bpw avg) of {a.budget_gb} GB budget")
    print("sensitivity:", source)
    return report


def layer_sweep_cmds(n_layers, low="Q2_K", high="Q8_0", model="bf16.gguf", imat="imat.dat", ref="base.kld", text="wiki.test.raw"):
    """Shell commands that measure each layer's sensitivity: quantize ONLY that layer's experts to `low` (all else `high`), then KL vs BF16."""
    cmds = []
    for i in range(n_layers):
        cmds.append(f"llama-quantize --imatrix {imat} --tensor-type '^blk\\.{i}\\.ffn_(gate_up|gate|up|down)_exps\\.weight={low}' {model} sweep_{i}.gguf {high} "
                    f"&& llama-perplexity -m sweep_{i}.gguf -f {text} --kl-divergence-base {ref} --kl-divergence | tee sweep_{i}.log")
    return cmds


if __name__ == "__main__":
    main()
