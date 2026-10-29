"""T05: result schema v3, reconciliation rules and the incident corpus (each past mistake: a defective case that must be flagged and a fixed twin that must pass)."""
import copy, json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "tools")); sys.path.insert(0, HERE)
import results_check as rc; from harness import T

def clean():
    return {"schema": 3, "os": "Linux", "machine": {"cpu": "x86", "physical_cores": 2, "ram_gb": 5.8, "ram_cap_gb": 5.0, "cap_label": "MEASURED-CAP"}, "model": "m.gguf", "model_gb": 18.6, "case": "mmap",
            "tok_s_median": 0.12, "tokens_counted": 33, "read_mib_per_token_median": 900.0, "disk": {"seq_mib_s": 120, "rand_mib_s": 120}, "phase": {"load_s": 10.0, "prefill_s": 140.0, "decode_s": 260.0}, "wall_s": 410.0, "patch_hash": "d56b678f5044"}

def setp(r, path, v):
    d = r; ps = path.split(".")
    for p in ps[:-1]: d = d[p]
    d[ps[-1]] = v
    return r
def drop(r, path):
    d = r; ps = path.split(".")
    for p in ps[:-1]: d = d[p]
    del d[ps[-1]]; return r

# (name, rule, edits that create the defect, edits that fix it)
INC = [
 ("run that stayed cached (read per token ~0)", "cached_run", [("read_mib_per_token_median", 0.0)], [("read_mib_per_token_median", 900.0)]),
 ("Qwen3-30B run that stayed cached (0.1 MiB per token at 8 tok/s)", "cached_run", [("read_mib_per_token_median", 0.1), ("tok_s_median", 8.0)], [("read_mib_per_token_median", 900.0)]),
 ("read per token x tok/s above the disk limit (the madvise file: 372 MiB/s on 120)", "implied_rate_exceeds_disk", [("tok_s_median", 0.1464), ("read_mib_per_token_median", 2543.0)], [("read_mib_per_token_median", 300.0)]),
 ("speedup quoted from the decode window, decode is 19% of wall", "window_mismatch", [("speedup_basis", "decode"), ("phase.decode_s", 77.0)], [("speedup_basis", "wall"), ("phase.decode_s", 260.0)]),
 ("soft RAM cap requested but not achieved", "soft_cap", [("machine.cap_label", "SOFT-CAP")], [("machine.cap_label", "MEASURED-CAP")]),
 ("contended run", "contended", [("contended", True)], [("contended", False)]),
 ("pin list scored on the trace it was built from", "in_sample", [("pin", {"label": "in_sample", "learn_source": "story", "scored_on": "story"})], [("pin", {"label": "cross_prompt", "learn_source": "db", "scored_on": "story"})]),
 ("pin list from another prompt labelled same-prompt", "mislabelled_pin", [("pin", {"label": "same_prompt", "learn_source": "db", "scored_on": "story"})], [("pin", {"label": "cross_prompt", "learn_source": "db", "scored_on": "story"})]),
 ("significance claimed with 3 runs per arm", "too_few_runs", [("claims_significant", True), ("n_per_arm", 3)], [("claims_significant", True), ("n_per_arm", 6)]),
 ("run order unbalanced", "unbalanced_order", [("order_balanced", False)], [("order_balanced", True)]),
 ("unexplained outlier row (267 s among 375-410 s)", "outlier", [("outlier_rows", [1])], [("outlier_rows", [])]),
 ("tokens counted ignores the 1-token warmup", "denominator", [("tokens_counted", 32)], [("tokens_counted", 33)]),
 ("cache expressed in percent of experts", "percent_units", [("cache_unit", "percent")], [("cache_unit", "footprints")]),
 ("dense weights charged to the disk", "dense_on_disk", [("disk_bytes_include_dense", True)], [("disk_bytes_include_dense", False)]),
 ("projection without serial and overlap bounds", "bound_unstated", [("kind", "projection")], [("kind", "projection"), ("serial_tok_s", 0.5), ("overlap_tok_s", 0.6)]),
 ("proxy-to-target mapping not stated", "transfer_unstated", [("kind", "projection"), ("serial_tok_s", 0.5), ("proxy_model", "qwen3-30b")], [("kind", "projection"), ("serial_tok_s", 0.5), ("proxy_model", "qwen3-30b"), ("transfer_mapping", "by footprint")]),
 ("prefill not measured", "no_prefill", [("phase.prefill_s", None)], [("phase.prefill_s", 140.0)]),
 ("cost counter far from the balance delta (0.20 vs 4.6)", "counter_vs_balance", [("cost_counter_usd", 0.2), ("balance_delta_usd", 4.6)], [("cost_counter_usd", 4.5), ("balance_delta_usd", 4.6)]),
 ("balance ended below the spend floor", "below_floor", [("balance_after", 2.33), ("floor", 5.0)], [("balance_after", 6.0), ("floor", 5.0)]),
 ("no patch hash recorded", "no_patch_hash", [("patch_hash", "none")], [("patch_hash", "d56b678f5044")]),
 ("1,300 tok/s for an 18.6 GB model (tiny-model number on the wrong file)", "implausible_speed", [("tok_s_median", 1300.0)], [("tok_s_median", 0.12)]),
]

def run():
    t = T("T05 result checks")
    t.check("the clean result passes with no flags", rc.check(clean()) == [], str(rc.check(clean())))
    for i, (name, rule, bad, good) in enumerate(INC, 1):
        r = clean(); [setp(r, k, v) for k, v in bad]; got = [x[0] for x in rc.check(r, n_gen=32 if rule == "denominator" else None)]
        t.check(f"incident {i:02d} is flagged by rule '{rule}': {name}", rule in got, f"fired: {got}")
        r = clean(); [setp(r, k, v) for k, v in good]; got = rc.check(r, n_gen=32 if rule == "denominator" else None)
        t.check(f"incident {i:02d} fixed twin passes: {name}", got == [], str(got))
    for f in rc.REQUIRED:
        r = drop(clean(), f); got = rc.check(r)
        t.check(f"missing field '{f}' is rejected and named", any(x[0] == "missing_field" and f in x[1] for x in got), str(got))
    for f in rc.NUMERIC:
        for label, v in (("NaN", float("nan")), ("negative", -1.0), ("infinity", float("inf")), ("a string", "12")):
            r = setp(clean(), f, v); got = rc.check(r)
            t.check(f"{label} in '{f}' is rejected", any(x[0] == "bad_number" and f in x[1] for x in got), str(got))
    text = json.dumps(clean())
    for pos in (1, 10, 50, 200, len(text) - 1):
        try: rc.load_result(text[:pos]); ok = False
        except rc.ResultError: ok = True
        t.check(f"JSON truncated at byte {pos} raises ResultError, nothing half-read", ok)
    t.check("a complete JSON text round-trips", rc.load_result(text) == clean())
    # unit swaps are caught by reconciliation, not by the schema
    r = setp(clean(), "read_mib_per_token_median", 900.0 * 1024); t.check("unit swap: read per token x1024 is flagged", any(x[0] == "implied_rate_exceeds_disk" for x in rc.check(r)))
    r = setp(clean(), "tok_s_median", 0.12 * 1024); t.check("unit swap: tok/s x1024 is flagged as implausible or implied-rate", any(x[0] in ("implausible_speed", "implied_rate_exceeds_disk") for x in rc.check(r)))
    r = clean(); r["disk"] = {"seq_mib_s": 120 / 1024, "rand_mib_s": 120 / 1024}; t.check("unit swap: disk rates /1024 is flagged", any(x[0] == "implied_rate_exceeds_disk" for x in rc.check(r)))
    # several defects at once are all reported
    r = clean(); setp(r, "contended", True); setp(r, "patch_hash", "none"); setp(r, "machine.cap_label", "SOFT-CAP"); got = {x[0] for x in rc.check(r)}
    t.check("three defects in one result are all reported", {"contended", "no_patch_hash", "soft_cap"} <= got, str(got))
    return t

if __name__ == "__main__":
    sys.exit(0 if run().summary() else 1)
