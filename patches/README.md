# llama.cpp patches (base commit eec18f5)

Apply with `git apply patches/0001-...patch patches/0002-...patch` inside a llama.cpp checkout at commit `eec18f5`, or run `tools/setup_llama.sh` which does clone, apply and build.

| patch | what it does | switch |
|---|---|---|
| 0001-expert-trace-prefetch-pin-pool | MoE expert reads for single-token decode steps: **read pool** (explicit pread into an LRU pool, cache-bypassing; optional overlap with compute), madvise prefetch, routing trace (decode and prefill), hot-set pin | see below |
| 0003-routing-experiments | cache-aware routing (bias toward experts resident in the pool), anchored use of only K of the selected experts, training-free layer skipping (qwen3moe only; refuted: large quality loss) | `GGML_MOE_CACHE_BIAS=lambda`, `GGML_MOE_USE_K=k`, `GGML_SKIP_LAYERS`, `GGML_SKIP_MOE` |
| 0002-phase-timing | prints one `PHASE decode n_tokens=N start= end=` line per decode call | `GGML_PHASE=1` |
| experiments/ | AVX512-VNNI Q4_K dot (bit-identical, no speed-up) and mlock of non-expert tensors (no effect on a healthy machine); kept for the record, not needed | `GGML_Q4K_VNNI=1`, `LLAMA_MLOCK_DENSE=1` |

All switches are off by default and change no output (checked bit-for-bit on a tiny model across thousands of runs).

## Switches in 0001
- `GGML_MOE_POOL=<offsets file>` `GGML_MOE_POOL_MB=<MiB>`: read pool. The offsets file comes from `tools/gguf_offsets.py MODEL.gguf OUT.txt`; `tools/moe_run.py` makes and caches it for you.
  - `GGML_MOE_POOL_ASYNC=<n>`: n blocking I/O threads; reads overlap with the compute of resident experts (2 is a good start).
  - `GGML_MOE_POOL_DIRECT=1`: O_DIRECT on Linux (macOS always uses F_NOCACHE). `GGML_MOE_POOL_LOG=1`: print hits/misses/bytes at exit.
  - `GGML_MOE_POOL_ADAPT=1`: shrink the pool when free memory runs short, grow back later (avoids the collapse of an oversized pool). `GGML_MOE_POOL_WARM=<file>`: save hot experts at exit, preload at start (no gain across different prompts).
- `GGML_MOE_PREFETCH=1|2`: madvise(WILLNEED) / parallel page touch on the chosen experts (mmap path).
- `GGML_MOE_TRACE=<file>` decode routing trace; `GGML_MOE_TRACE_PREFILL=<file>` one line per layer per prefill batch; `GGML_MOE_PIN=<file>` `GGML_MOE_PIN_LOG=1` hot-set mlock.

Prefill batches and models/tensors not listed in the offsets file use the normal mmap path. Not supported on Windows (the pool compiles out).
