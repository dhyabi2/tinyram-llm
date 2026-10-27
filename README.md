# tinyram-llm

Tools and measurements for running large sparse mixture-of-experts (MoE) models from disk on machines with little RAM, using [llama.cpp](https://github.com/ggml-org/llama.cpp).

Everything here is something you can run and check yourself. **No speedup is claimed.** The measured results are baselines only (plain `-lm mmap`).

## Measured results (baseline, plain mmap)

| Machine | Model | RAM cap | tok/s | Page faults per token | Disk (seq / rand 1 MiB) |
|---|---|---|---|---|---|
| 2-core EPYC VPS | Qwen3-30B-A3B Q4_K_M (18.6 GB) | 5 GB | 0.118 | ~3,700 | 120 / 120 MiB/s |
| 1-core Colab VM | gpt-oss-20b MXFP4 (12.1 GB) | 4 GB (see note) | 0.094 | ~456 | 209 / 647 MiB/s |

Both runs are disk-bound. The Colab run read about 180 GB for 32 tokens (about 5.6 GB per token) although the active weights are about 2 GB per token, so the page cache thrashes. Raw files are in `results/`.

Caveats: the RAM cap is held with ballast memory and the page cache counts as available, so the effective cache was larger than the target. The VPS was a shared machine. Only the mmap case was run. Both models are smaller than the targets this was designed for.

## What is in the repo

- `probe/probe.sh`: builds llama.cpp (CPU), downloads a GGUF model, measures disk speed with fio, and runs `llama-bench -lm mmap --repack 0`, recording tok/s, major page faults per token (from `/proc/vmstat`) and data read. `--swap-test` also runs the swap case and is off by default because it changes swap on the machine.
- `probe/probe.ipynb`: the same script as a Colab/Kaggle notebook (CPU runtime).
- `tools/make_layer_quant_map.py`: per-layer expert quantization map for `llama-quantize --tensor-type-file`, spending a byte budget where it buys the most accuracy under a stated noise model. Pins norm tensors to f32 (see Notes).
- `tools/hit_rate_sim.py`, `tools/throughput_model.py`: expert-cache hit-rate simulation on a synthetic trace and a simple throughput model. Synthetic inputs, not results.
- `tools/build_ngram.py`, `tools/reference_impl.py` (+ tests): reference implementations used as correctness oracles.
- `c/expert_pool.c`: a small byte-capacity LRU pool reading extents with direct I/O, with a test (passes under ASan/UBSan). It is not wired into ggml.

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
- `fio` can print a note before its JSON output; strip it before parsing.

## License

MIT, see `LICENSE`.
