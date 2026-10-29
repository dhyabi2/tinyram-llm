"""PHASE parser, schema converter and the whole contract: PHASE text -> summary -> schema-3 record -> checker."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "tools")); sys.path.insert(0, HERE)
import phase_parse as pp, results_check as rc, to_schema3 as t3; from harness import T
def L(n, a, b): return f"PHASE decode n_tokens={n} start={a:.6f} end={b:.6f}"
def run():
    t = T("T08/T05 phase and contract")
    # ---- phase parser: one case each ----
    t.check("no PHASE lines gives None", pp.summarize("hello\nworld") is None)
    t.check("empty text gives None", pp.summarize("") is None)
    s = pp.summarize("\n".join([L(6, 0, .08), L(1, .08, .09), L(1, .09, .10)]))
    t.check("prompt of 6 tokens then 2 decode calls: prompt_s .08", abs(s["prompt_s"] - .08) < 1e-9 and s["prompt_tokens"] == 6 and s["prompt_calls"] == 1)
    t.check("decode_s is the sum of the single-token calls", abs(s["decode_s"] - .02) < 1e-9 and s["decode_calls"] == 2)
    t.check("ttft is the end of the last prompt call", abs(s["ttft_s"] - .08) < 1e-9)
    t.check("total spans first start to last end", abs(s["total_s"] - .10) < 1e-9)
    t.check("junk glued in front of a PHASE line is ignored", pp.summarize("52323996374" + L(2, 0, 1))["prompt_tokens"] == 2)
    t.check("generated text between lines is ignored", pp.summarize(L(3, 0, 1) + "\nsome generated words\n" + L(1, 1, 2))["decode_calls"] == 1)
    s = pp.summarize("\n".join([L(1, 0, .5), L(512, .5, 3.5), L(512, 3.6, 6.6), L(3, 6.7, 7.0), L(1, 7.0, 7.4), L(1, 7.4, 7.8)]))
    t.check("a warmup call before a chunked prompt is separated", s["warmup_calls"] == 1 and abs(s["warmup_s"] - .5) < 1e-9)
    t.check("a chunked prompt (512+512+3) sums its calls", s["prompt_calls"] == 3 and s["prompt_tokens"] == 1027 and abs(s["prompt_s"] - (3.0 + 3.0 + .3)) < 1e-9)
    t.check("gaps between calls are reported, not hidden", s["gap_s"] > 0.15)
    t.check("decode only (no prompt call) works", pp.summarize(L(1, 0, 1) + "\n" + L(1, 1, 2))["decode_calls"] == 2)
    t.check("a line with end before start is dropped", pp.summarize(L(1, 5, 4)) is None)
    t.check("a line with 0 tokens is dropped", pp.summarize("PHASE decode n_tokens=0 start=0.0 end=1.0") is None)
    t.check("a line with a non-numeric time is dropped", pp.summarize("PHASE decode n_tokens=2 start=abc end=1.0") is None)
    t.check("CRLF line endings parse", pp.summarize(L(2, 0, 1) + "\r\n" + L(1, 1, 2) + "\r\n")["decode_calls"] == 1)
    t.check("a truncated last line is dropped, earlier lines kept", pp.summarize(L(2, 0, 1) + "\nPHASE decode n_tokens=1 sta")["prompt_calls"] == 1)
    # ---- the whole chain on a synthetic but complete run ----
    text = "\n".join([L(1, 0, .4), L(33, .4, 120.4), L(1, 120.4, 128.0), L(1, 128.0, 135.4)] + [L(1, 135.4 + i * 7.5, 135.4 + (i + 1) * 7.5) for i in range(30)])
    ph = pp.summarize(text)
    old = {"schema": 2, "os": "Linux", "machine": {"cpu": "x", "physical_cores": 2, "ram_gb": 5.8, "ram_cap_gb": 5.0, "cap_label": "MEASURED-CAP"}, "model": "m.gguf", "model_gb": 18.6, "disk": {"128": {"seq_mib_s": 120, "rand_mib_s": 120}, "1024": {"seq_mib_s": 122, "rand_mib_s": 120}},
           "cases": [{"case": "mmap", "tok_s_median": 0.133, "tokens_counted": 33, "read_mib_per_token_median": 900.0}]}
    v3 = t3.convert(old, phase=ph, patch_hash="d56b678f5044")
    t.check("integration: PHASE text -> summary -> schema 3 -> checker: no flags", rc.check(v3) == [], str(rc.check(v3)))
    t.check("integration: the record carries the measured phases", abs(v3["phase"]["prefill_s"] - 120.0) < 1e-6 and v3["wall_s"] == ph["total_s"])
    t.check("integration: disk rate is the max over the read sizes", v3["disk"] == {"seq_mib_s": 122, "rand_mib_s": 120})
    t.check("a missing patch hash is reported (as a missing field)", any(x[0] in ("no_patch_hash", "missing_field") and "patch_hash" in x[1] for x in rc.check(t3.convert(old, phase=ph, patch_hash=None))))
    bad = json.loads(json.dumps(v3)); bad["phase"]["decode_s"] = 1000.0
    t.check("phases that add up to more than the wall time are flagged", any(x[0] == "phase_exceeds_wall" for x in rc.check(bad)))
    # ---- legacy conversions: kept, but never a clean pass ----
    leg = t3.convert(old, phase=None, patch_hash=None)
    t.check("without phase timing the record is marked legacy", leg.get("legacy") is True)
    t.check("a legacy record always reports legacy_no_phase", any(x[0] == "legacy_no_phase" for x in rc.check(leg)))
    t.check("a legacy record is never reported as clean", rc.check(leg) != [])
    root = os.path.join(HERE, "..", "results")
    for name, ngen, gb in (("vps-qwen3-30b-a3b", 32, 18.6), ("colab-gpt-oss-20b", 32, 12.1)):
        old = json.load(open(os.path.join(root, name, "results.json"))); v = t3.convert(old, n_gen=ngen, model_gb=gb); fl = [x[0] for x in rc.check(v)]
        t.check(f"real result {name}: converts with 33 counted tokens", v["tokens_counted"] == 33)
        t.check(f"real result {name}: read per token = total read / 33", abs(v["read_mib_per_token_median"] - old["cases"][0]["read_mib_median"] / 33) < 1e-6)
        t.check(f"real result {name}: flagged as legacy, not as cached or bad number", "legacy_no_phase" in fl and "cached_run" not in fl and "bad_number" not in fl, str(fl))
    mac = json.load(open(os.path.join(root, "mac-qwen3.8-27b-dense", "results.json")))
    v = t3.convert(mac, n_gen=16, model_gb=14.25); fl = [x[0] for x in rc.check(v)]
    t.check("real result mac 27B dense: legacy flag present", "legacy_no_phase" in fl or any(x[0] == "missing_field" for x in rc.check(v)), str(fl))
    return t

if __name__ == "__main__":
    sys.exit(0 if run().summary() else 1)
