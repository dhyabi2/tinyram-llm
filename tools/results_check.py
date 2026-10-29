"""Checker for probe result files (schema 3) and the rules that catch the mistakes made so far. A result is a dict (or JSON text).
check(result) returns a list of (rule, message). An empty list means the result passes. Rules are named so tests can assert which one fired."""
import json, math

REQUIRED = ["schema", "os", "machine.cpu", "machine.physical_cores", "machine.ram_gb", "machine.cap_label", "model", "case", "tok_s_median",
            "tokens_counted", "read_mib_per_token_median", "disk", "phase.load_s", "phase.prefill_s", "phase.decode_s", "wall_s", "patch_hash"]
NUMERIC = ["tok_s_median", "read_mib_per_token_median", "wall_s", "disk.seq_mib_s", "disk.rand_mib_s", "machine.ram_gb", "tokens_counted", "phase.decode_s"]

class ResultError(Exception): pass

def load_result(text):
    try: return json.loads(text)
    except Exception as e: raise ResultError(f"not valid JSON: {e}")

def get(d, path):
    for p in path.split("."):
        if not isinstance(d, dict) or p not in d: return None
        d = d[p]
    return d

LEGACY_OPTIONAL = ("phase.load_s", "phase.prefill_s", "phase.decode_s", "wall_s", "patch_hash")

def check(r, n_gen=None):
    out = []
    legacy = bool(r.get("legacy"))
    for f in REQUIRED:
        if legacy and f in LEGACY_OPTIONAL: continue
        if get(r, f) is None and not (f == "phase.prefill_s" and isinstance(r.get("phase"), dict) and "prefill_s" in r["phase"]):   # an explicit null prefill means "not measured" and is judged below
            out.append(("missing_field", f"required field '{f}' is missing"))
    for f in NUMERIC:
        v = get(r, f)
        if v is not None and (not isinstance(v, (int, float)) or isinstance(v, bool) or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) or v < 0):
            out.append(("bad_number", f"'{f}' = {v!r} is not a finite non-negative number"))
    if out: return out                                           # the semantic rules need sane numbers
    if legacy: out.append(("legacy_no_phase", "converted from a schema 1/2 result: no phase timing and no patch hash, so window and denominator rules cannot be checked"))
    rpt, tps = r["read_mib_per_token_median"], r["tok_s_median"]; disk = max(get(r, "disk.seq_mib_s") or 0, get(r, "disk.rand_mib_s") or 0)
    if rpt < 20 and r.get("model_gb", 0) > 1: out.append(("cached_run", f"read per token {rpt} MiB: the model stayed in the page cache, this is not a streaming result"))
    if disk and tps * rpt > 1.25 * disk: out.append(("implied_rate_exceeds_disk", f"tok/s x read per token = {tps*rpt:.0f} MiB/s exceeds the disk's {disk} MiB/s"))
    if r.get("speedup_basis") == "decode" and r["wall_s"] > 0 and r["phase"]["decode_s"] / r["wall_s"] < 0.5: out.append(("window_mismatch", "speedup quoted from the decode window but decode is under half of the wall time"))
    if get(r, "machine.ram_cap_gb") and r["machine"]["cap_label"] in ("SOFT-CAP", "UNCAPPED"): out.append(("soft_cap", f"a RAM cap was requested but the cap label is {r['machine']['cap_label']}"))
    if n_gen is not None and r["tokens_counted"] == n_gen: out.append(("denominator", "tokens_counted equals the generated tokens: the warmup token is not counted"))
    if r.get("contended"): out.append(("contended", "another test was running"))
    pin = r.get("pin")
    if pin and pin.get("learn_source") == pin.get("scored_on"): out.append(("in_sample", "the pinned set was learned on the data it is scored on"))
    if pin and pin.get("label") == "same_prompt" and pin.get("learn_source") != pin.get("scored_on"): out.append(("mislabelled_pin", "labelled same-prompt but learned on a different source"))
    if r.get("claims_significant") and r.get("n_per_arm", 99) < 5: out.append(("too_few_runs", "significance claimed with fewer than 5 runs per arm"))
    if r.get("order_balanced") is False: out.append(("unbalanced_order", "run order is not balanced"))
    if r.get("outlier_rows"): out.append(("outlier", f"unexplained outlier rows: {r['outlier_rows']}"))
    if r.get("cache_unit") == "percent": out.append(("percent_units", "cache expressed in percent of experts instead of token-footprints"))
    if r.get("disk_bytes_include_dense"): out.append(("dense_on_disk", "resident dense weights are charged to the disk"))
    if "serial_tok_s" not in r and "overlap_tok_s" not in r and r.get("kind") == "projection": out.append(("bound_unstated", "a projection without a serial and an overlap bound"))
    if r.get("kind") == "projection" and r.get("proxy_model") and not r.get("transfer_mapping"): out.append(("transfer_unstated", "a proxy-to-target mapping is not stated"))
    ph = r.get("phase") or {}
    if not legacy and r.get("wall_s") and sum(v for k, v in ph.items() if k.endswith("_s") and isinstance(v, (int, float))) > 1.02 * r["wall_s"]: out.append(("phase_exceeds_wall", "the phases add up to more than the wall time"))
    if legacy: return out
    if r["phase"].get("prefill_s") is None: out.append(("no_prefill", "prefill was not measured"))
    cb, bd = r.get("cost_counter_usd"), r.get("balance_delta_usd")
    if cb is not None and bd is not None and bd > 0 and abs(cb - bd) / bd > 0.2: out.append(("counter_vs_balance", f"cost counter {cb} vs balance delta {bd}"))
    if r.get("balance_after") is not None and r.get("floor") is not None and r["balance_after"] < r["floor"]: out.append(("below_floor", "balance ended below the spend floor"))
    if not r.get("patch_hash") or r["patch_hash"] in ("", "none"): out.append(("no_patch_hash", "no patch hash recorded"))
    if r.get("model_gb", 0) >= 1 and tps > 200: out.append(("implausible_speed", f"{tps} tok/s for a {r['model_gb']} GB model on CPU"))
    return out
