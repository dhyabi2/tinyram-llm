# tinyram-llm

Tools and measurements for running large sparse mixture-of-experts (MoE) models from disk on machines with little RAM, using [llama.cpp](https://github.com/ggml-org/llama.cpp).

Everything here is something you can run and check yourself. Measured results come with their conditions; projections are labelled as projections.

## Quick start (any MoE GGUF)

```bash
bash tools/setup_llama.sh ./llama.cpp                 # clone llama.cpp at the pinned commit, apply patches/, build (needs git, cmake, a compiler)
python3 tools/moe_run.py --model MODEL.gguf --bin ./llama.cpp/build/bin -- -p "Hello" -n 64     # picks threads/pool size for this machine, then runs
python3 tools/compare_run.py --model MODEL.gguf --bin ./llama.cpp/build/bin --tokens 32 --reps 3   # plain mmap vs read pool vs overlapped pool, interleaved
python3 tests/run_all.py                              # about 17,000 checks, no model needed for most
```

To drive this on another computer with Claude Code, paste the prompt in `docs/RUN_WITH_CLAUDE.md`; for Windows (where the read pool still has to be ported) use `docs/WINDOWS_PORT_PROMPT.md`.

## Measured with the explicit expert read pool

The pool (patch `patches/0001`) reads the experts the router picks with explicit aligned reads into an LRU cache instead of letting mmap page them in; with `GGML_MOE_POOL_ASYNC=n`, n background threads read the missing experts while the cached ones are computed.

| Machine | Model | Memory | Plain mmap | Pool | Pool + overlap |
|---|---|---|---|---|---|
| 2020 MacBook Pro, 4 cores, 16 GB, internal NVMe | Qwen3-30B-A3B Q4_K_M (18.6 GB) | 9 GB wired ballast, file about 8x the cache | 0.513 tok/s | 2.28 tok/s (4.4x) | 2.645 tok/s (5.2x) |
| Linux VPS, 5.8 GB, throttled disk | same model | cold, 40 tokens | 0.177 tok/s | (madvise read-ahead on the chosen experts: 0.293 tok/s, 1.65x; wall time 702 -> 572 s) | - |

Fair-baseline check on the same Mac (48 tokens, 3 rounds): plain mmap 0.599 tok/s; mmap with madvise read-ahead on the chosen experts 0.648 (1.08x, within noise); mmap with a parallel page touch 0.315 (worse); pool 2.22 (3.7x); pool + overlap 2.79 (4.7x). Read-ahead hints on the mmap path do not close the gap.

Conditions: clean interleaved A/B (fully wired ballast, swap nearly empty), medians of 3 rounds of 16 tokens (Mac) and 5 rounds of 40 tokens with a null pair (VPS). Short runs start with a cold cache, so steady-state speed is probably higher for the pool. Raw rows: `results/mac-pool-vs-mmap/`, `results/vps-madvise-ab/`. Output is identical to the stock path (greedy text and perplexity) in the tests. Things measured that did not help: an AVX512-VNNI Q4_K kernel, locking the dense weights, lossless compression of expert bytes, cross-layer prefetch with simple predictors. Quality of using fewer experts per token (Qwen3-30B-A3B): 6 of 8 experts +4.7% perplexity, 4 of 8 +30.9%.

The 125B-scale numbers in `tools/combo_model.py`, `ram_sweep.py`, `hundredx_map.py` are model projections (set `COMBO_HIT=real` for a lower, more pessimistic hit-rate curve taken from public benchmarks), not measurements of a 125B model.

## Measured results (earlier baselines, plain mmap)

| Machine | Model | RAM cap | tok/s | Page faults per token | Disk (seq / rand 1 MiB) |
|---|---|---|---|---|---|
| 2-core EPYC VPS | Qwen3-30B-A3B Q4_K_M (18.6 GB) | 5 GB | 0.118 | ~3,700 | 120 / 120 MiB/s |
| 1-core Colab VM | gpt-oss-20b MXFP4 (12.1 GB) | 4 GB (see note) | 0.094 | ~456 | 209 / 647 MiB/s |
| 4-core Intel MacBook Pro, 16 GB | Qwen3.8-27B UD-IQ4_XS (14.25 GB, **dense**) | none applied | 0.027 | n/a (~13.6 GB paged in per token, the whole file) | 1.5-2.4 GB/s with 1-4 MiB reads, but mmap paging got ~340 MB/s |

All runs are disk-bound. The Colab run read about 180 GB for 32 tokens (about 5.6 GB per token) although the active weights are about 2 GB per token, so the page cache thrashes. Raw files are in `results/`.

Caveats: the RAM cap is held with ballast memory and the page cache counts as available, so the effective cache was larger than the target. The VPS was a shared machine. Only the mmap case was run. Both models are smaller than the targets this was designed for.

## Read size decides disk speed (measured on the Mac)

| Read size | Random read |
|---|---|
| 16 KiB | 101 MiB/s |
| 128 KiB | 465 MiB/s |
| 1 MiB | 1,562 MiB/s |
| 4 MiB | 2,380 MiB/s |

mmap paging of the dense 27B file ran at about 340 MB/s, which matches 64-128 KiB reads. Reading in multi-MiB pieces (an MoE expert is a few MiB) can use 4.5-7x more of the same disk. That is the measured disk-side headroom on this Mac, and it is far from 100x. Raw numbers: `results/mac-disk-read-size-sweep/`. A VPS disk showed the opposite: about 120 MiB/s at every read size, because the disk is throttled.

## What is in the repo

- `probe/probe.sh`: builds llama.cpp (CPU), downloads a GGUF model and checks its exact size, measures cache-bypassing disk speed at three read sizes (no fio needed), caps usable RAM with a ballast sized from what is available and VERIFIED (`MEASURED-CAP` or `SOFT-CAP`), and runs `llama-bench -lm mmap --repack 0`, recording tok/s, major page faults per token (from `/proc/vmstat`) and data read. `--swap-test` also runs the swap case and is off by default because it changes swap on the machine.
- `probe/probe.ipynb`: the same script as a Colab/Kaggle notebook (CPU runtime).
- `tools/make_layer_quant_map.py`: per-layer expert quantization map for `llama-quantize --tensor-type-file`, spending a byte budget where it buys the most accuracy under a stated noise model. Pins norm tensors to f32 (see Notes).
- `probe/probe_mac.sh`: macOS version of the probe. Counts page-ins from `vm_stat` (system-wide, so close other apps). `--cap-gb N` emulates a smaller RAM with a wired ballast and verifies it; the result file has the same fields as `probe.sh` (schema 2, with `tokens_counted`, `cap_label`, `os`).
- `mac/ballast.c`, `mac/wired_test.sh`: a wired (`mlock`) memory ballast for emulating a smaller RAM on macOS. Checked on a 16 GB Intel Mac with no sudo: 2 GiB wired, wired pages rose by 99.9% of the target and the compressor did not grow.
- `tools/disk_bench.py`: cache-bypassing read benchmark (macOS and Linux) at a chosen read size.
- `tools/preflight.sh`: 2-second checks before a long run (OS and CPU flags, RAM, decimal size parsing, disk headroom, write test, download tool, HTTP range support, exact remote file size, llama-bench flags). One JSON line per check. A range check that does not follow redirects fails on Hugging Face; this one follows them.
- `tools/hit_rate_sim.py`, `tools/throughput_model.py`: expert-cache hit-rate simulation on a synthetic trace and a simple throughput model. Synthetic inputs, not results.
- `tools/build_ngram.py`, `tools/reference_impl.py` (+ tests): reference implementations used as correctness oracles.
- `c/expert_pool.c`: a small standalone byte-capacity LRU pool with direct I/O, with tests (ASan/UBSan, 150 fuzz scenarios). The pool inside llama.cpp is in `patches/0001`.
- `patches/`: llama.cpp patches (read pool with overlap, prefetch, routing trace, pin, phase timing) for commit eec18f5; `tools/setup_llama.sh` applies and builds them.
- `tools/moe_run.py`, `tools/compare_run.py`, `tools/gguf_offsets.py`: launcher with machine-adaptive settings, interleaved comparison, GGUF expert offsets.
- `tools/combo_trace_sim.py`, `combo_model.py`, `ram_sweep.py`, `hundredx_map.py`, `cachesim.py`, `trace_tools.py`: trace-driven cache and combination simulators (real routing traces in `data/traces/`); `tools/pubtech/`: simulations of published speed-up ideas, one function each.
- `tests/`: 22 suites, run all with `python3 tests/run_all.py`.

## Usage

```bash
# Linux box (root recommended for installing build tools)
bash probe/probe.sh --ram-gb 8 --model-repo ggml-org/gpt-oss-20b-GGUF \
  --model-file gpt-oss-20b-MXFP4.gguf --model-gb 12.2 --active-b 3.6
# results.json and probe.log are written to ~/moe-probe (change with --workdir)
```

Colab: upload `probe/probe.ipynb`, choose a CPU runtime, run the cells in order. Check free disk first; the model plus about 8 GB for the build must fit.

Python tools need `numpy` (`pip install numpy`). C test: `cc -O1 -o t c/test_expert_pool.c c/expert_pool.c && ./t`.

## Notes found while testing

- llama.cpp's CPU backend repacks some quantized weights into one large allocated buffer, which defeats mmap on small RAM. Pass `--repack 0`.
- The stock quantizer can quantize small 2-D norm tensors in some architectures, and the model then aborts at run time. Pin them to f32.
- A dense model (every weight touched on every token) gets nothing from the page cache once the file is near the size of RAM: the 14.25 GB dense run paged in about the whole file per token.
- `llama-bench` runs a 1-token warmup, so divide read volume by tokens + 1; the result files record `tokens_counted`.
- A RAM cap sized from total RAM leaves the page cache larger than intended; size it from available memory and check afterwards.
- `tools/disk_bench.py` replaces fio, which printed a note before its JSON and broke parsing.

## License

Business Source License 1.1, see `LICENSE`. Free to use, copy, modify and distribute for any purpose, except that a company (with affiliates) over US$50 million annual gross revenue needs a separate commercial license. It converts to the MIT license on 2031-10-28.
