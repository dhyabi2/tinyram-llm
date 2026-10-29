"""Convert a probe result (schema 1/2) plus parsed phase timing into a schema-3 record that tools/results_check.py can judge.
Without phase timing the record is marked legacy: it is kept, but check() always reports 'legacy_no_phase' so it can never pass as a clean v3 result."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def _disk_max(disk):
    if not disk: return None, None
    if "seq_mib_s" in disk: return disk.get("seq_mib_s"), disk.get("rand1m_mib_s", disk.get("rand_mib_s"))
    seq = [v.get("seq_mib_s") for v in disk.values() if isinstance(v, dict) and v.get("seq_mib_s")]
    rnd = [v.get("rand_mib_s") for v in disk.values() if isinstance(v, dict) and v.get("rand_mib_s")]
    return (max(seq) if seq else None), (max(rnd) if rnd else None)

def convert(old, phase=None, patch_hash=None, n_gen=None, model_gb=None, case_index=0):
    case = (old.get("cases") or [old])[case_index]; m = old.get("machine", {})
    tokens = case.get("tokens_counted") or (n_gen + 1 if n_gen else None)
    rpt = case.get("read_mib_per_token_median")
    if rpt is None and case.get("read_mib_median") is not None and tokens: rpt = case["read_mib_median"] / tokens
    seq, rnd = _disk_max(old.get("disk"))
    r = {"schema": 3, "os": old.get("os") or ("Linux" if m.get("arch") else None), "model": old.get("model"), "model_gb": model_gb or old.get("model_gb"), "case": case.get("case"),
         "machine": {"cpu": m.get("cpu"), "physical_cores": m.get("physical_cores"), "ram_gb": m.get("ram_gb"), "ram_cap_gb": m.get("ram_cap_gb"), "cap_label": m.get("cap_label") or ("UNKNOWN-legacy" if m.get("ram_cap_gb") else "UNCAPPED")},
         "tok_s_median": case.get("tok_s_median"), "tokens_counted": tokens, "read_mib_per_token_median": rpt, "disk": {"seq_mib_s": seq, "rand_mib_s": rnd}, "patch_hash": patch_hash}
    if phase:
        r["phase"] = {"load_s": phase.get("warmup_s", 0.0), "prefill_s": phase.get("prompt_s"), "decode_s": phase.get("decode_s")}; r["wall_s"] = phase.get("total_s")
    else:
        r["legacy"] = True; r["phase"] = {"load_s": None, "prefill_s": None, "decode_s": None}; r["wall_s"] = None
    return {k: v for k, v in r.items() if v is not None or k in ("patch_hash", "wall_s")}
