#!/usr/bin/env python3
"""Runs every test suite and prints one summary. Exit code 0 only if every check passed. No network, no models.
usage: python3 tests/run_all.py          (set TEST_CATALOG=file.jsonl to record every check by name)"""
import importlib, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
SUITES = ["test_cachesim", "test_trace", "test_abstats", "test_results_check", "test_to_schema3", "test_pin", "test_mincore", "test_models", "test_disk_bench", "test_probe_parsers", "test_shell_preflight", "test_hygiene_data", "test_combo_trace", "test_native_and_tools", "test_pubtech_A", "test_pubtech_B", "test_pubtech_C", "test_pubtech_D", "test_pubtech_E", "test_pubtech_F", "test_gguf_offsets", "test_moe_run", "test_v41_model"]
tot = bad = 0; t0 = time.time(); rows = []
for name in SUITES:
    s0 = time.time(); r = importlib.import_module(name).run(); tot += r.n; bad += len(r.bad); rows.append((r.suite, r.n, len(r.bad), time.time() - s0))
print(f"\n{'suite':<28}{'checks':>8}{'failed':>8}{'secs':>8}")
for s, n, b, sec in rows: print(f"{s:<28}{n:>8}{b:>8}{sec:>8.1f}")
print(f"{'TOTAL':<28}{tot:>8}{bad:>8}{time.time()-t0:>8.1f}\n{tot - bad}/{tot} checks passed")
sys.exit(0 if bad == 0 else 1)
