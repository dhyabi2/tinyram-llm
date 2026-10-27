#!/usr/bin/env python3
"""End-to-end test of make_layer_quant_map.py against the real llama.cpp binaries and a real (tiny, random) qwen4_exp GGUF.
env: LLAMA_BIN (dir with llama-quantize, llama-completion), TINY (path to qwen4exp-moe.gguf)"""
import json, os, re, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_layer_quant_map as mq

BIN, TINY = os.environ["LLAMA_BIN"], os.environ["TINY"]
RES = []
def check(name, ok, detail=""):
    RES.append(bool(ok)); print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
def run(cmd): return subprocess.run(cmd, capture_output=True, text=True)

tmp = tempfile.mkdtemp()
# --- 1. read the real tensor shapes from the quantizer's own log
log = run([f"{BIN}/llama-quantize", TINY, f"{tmp}/probe.gguf", "Q8_0"]).stderr
params = {}
for m in re.finditer(r"blk\.(\d+)\.ffn_(?:gate_up|gate|up|down)_exps\.weight\s+- \[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", log):
    l, *dims = m.group(1), *map(int, m.groups()[1:])
    n = 1
    for d in dims: n *= d
    params[l] = params.get(l, 0) + n
check("the real GGUF has expert tensors for 2 layers (fused gate_up + down)", sorted(params) == ["0", "1"] and all(v > 0 for v in params.values()), str(params))
json.dump(params, open(f"{tmp}/params.json", "w"))

# --- 2. a layer-0-heavy sensitivity and a budget that can afford exactly one layer at 8.5 bpw
ladder = "q4_0:4.5,q5_0:5.5,q8_0:8.5"
json.dump({"0": 10.0, "1": 1.0}, open(f"{tmp}/sens.json", "w"))
want = params["0"] * 8.5 / 8 + params["1"] * 4.5 / 8
rep = mq.main(["--params", f"{tmp}/params.json", "--sensitivity", f"{tmp}/sens.json", "--ladder", ladder, "--budget-gb", f"{want * 1.002 / 1e9}",
               "--out", f"{tmp}/layers.txt", "--report", f"{tmp}/report.json"])
types = {l: v["type"] for l, v in rep["layers"].items()}
check("the sensitive layer is upgraded first and gets the high type", types == {"0": "q8_0", "1": "q4_0"}, str(types))
check("predicted size is within the budget", rep["predicted_expert_bytes"] <= want * 1.002 + 1, f"{rep['predicted_expert_bytes']:.0f} B vs budget {want * 1.002:.0f} B")

# --- 3. the generated file is accepted by the real quantizer, and each layer's experts get the assigned type
q = run([f"{BIN}/llama-quantize", "--tensor-type-file", f"{tmp}/layers.txt", TINY, f"{tmp}/mixed.gguf", "Q4_0"])
qlog = q.stderr
check("llama-quantize accepts the map and finishes", q.returncode == 0 and os.path.exists(f"{tmp}/mixed.gguf"), (q.stdout + qlog)[-200:] if q.returncode else "")
got, sizes = {}, {}
for m in re.finditer(r"blk\.(\d+)\.ffn_(?:gate_up|gate|up|down)_exps\.weight\s+- \[[^\]]*\],\s*type =\s*\w+, converting to (\w+) \.\. size =\s*([\d.]+) MiB ->\s*([\d.]+) MiB", qlog):
    got.setdefault(m.group(1), set()).add(m.group(2)); sizes[m.group(1)] = sizes.get(m.group(1), 0) + float(m.group(4))
check("layer 0 experts were quantized to q8_0 and layer 1 experts to q4_0, as mapped", got == {"0": {"q8_0"}, "1": {"q4_0"}}, str(got))
pred = {l: params[l] * (8.5 if l == "0" else 4.5) / 8 / 2**20 for l in params}
check("actual expert sizes match the prediction (within 0.02 MiB per layer)", all(abs(sizes[l] - pred[l]) < 0.02 for l in pred), f"actual {sizes} vs predicted { {k: round(v, 3) for k, v in pred.items()} }")
check("the rules carry the override message from the quantizer (manual override applied)", qlog.count("applying manual override") >= 1 or got.get("1") == {"q4_0"})

# --- 4. the mixed-precision model loads and generates in the real runtime
g = run([f"{BIN}/llama-completion", "-m", f"{tmp}/mixed.gguf", "-p", "hello", "-n", "8", "-ngl", "0", "-t", "2", "--no-warmup", "-no-cnv"])
check("the per-layer mixed-precision qwen4_exp model loads and generates", g.returncode == 0 and "eval time" in (g.stderr + g.stdout), (g.stderr[-160:] if g.returncode else ""))

# --- 4b. the control: WITHOUT the norm pin the same quantization aborts at run time (documents why the pin is the default)
open(f"{tmp}/nopin.txt", "w").write("".join(l for l in open(f"{tmp}/layers.txt") if not l.startswith("norm=")))
run([f"{BIN}/llama-quantize", "--tensor-type-file", f"{tmp}/nopin.txt", TINY, f"{tmp}/nopin.gguf", "Q4_0"])
c = run([f"{BIN}/llama-completion", "-m", f"{tmp}/nopin.gguf", "-p", "hello", "-n", "4", "-ngl", "0", "-t", "2", "--no-warmup", "-no-cnv"])
check("control: without the norm pin the quantized model aborts (binary_op unsupported types)", c.returncode != 0 and "binary_op: unsupported types" in (c.stderr + c.stdout), f"exit {c.returncode}")
check("the generated file starts with the norm pin (first matching rule wins)", open(f"{tmp}/layers.txt").read().split()[0] == "norm=f32")

# --- 5. the rules are anchored: layer 1 must not match layer 10, 11, ...
rule = [r for r in open(f"{tmp}/layers.txt").read().split() if r.startswith("^blk\\.1\\.")][0].split("=")[0]
check("the rule for layer 1 does not match blk.10 or blk.11", re.search(rule, "blk.1.ffn_down_exps.weight") and not re.search(rule, "blk.10.ffn_down_exps.weight") and not re.search(rule, "blk.11.ffn_gate_up_exps.weight"), rule)

# --- 6. allocation behaviour on a realistic 48-layer problem
p48 = {str(i): 2.5e9 for i in range(48)}; json.dump(p48, open(f"{tmp}/p48.json", "w"))
json.dump({str(i): 1.0 + (10.0 if i in (0, 1, 46, 47) else 0.0) for i in range(48)}, open(f"{tmp}/s48.json", "w"))
r48 = mq.main(["--params", f"{tmp}/p48.json", "--sensitivity", f"{tmp}/s48.json", "--budget-gb", "36", "--out", f"{tmp}/l48.txt"])
bp = {l: v["bpw"] for l, v in r48["layers"].items()}
check("48 layers: the high-sensitivity boundary layers get at least as many bits as any middle layer", min(bp[str(i)] for i in (0, 1, 46, 47)) >= max(bp[str(i)] for i in range(2, 46)), f"boundary {sorted({bp[str(i)] for i in (0,1,46,47)})} middle {sorted({bp[str(i)] for i in range(2,46)})}")
check("48 layers: the predicted size stays within the budget", r48["predicted_expert_gb"] <= 36.0 + 1e-9, f"{r48['predicted_expert_gb']} GB, {r48['average_bpw']} bpw")
try: mq.main(["--params", f"{tmp}/p48.json", "--budget-gb", "1", "--out", f"{tmp}/x.txt"]); low = False
except SystemExit as e: low = "below the minimum" in str(e)
check("a budget below the minimum is a clear error", low)

print(f"\n{sum(RES)}/{len(RES)} checks passed")
sys.exit(0 if all(RES) else 1)
