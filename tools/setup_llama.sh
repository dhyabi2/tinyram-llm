#!/usr/bin/env bash
# setup_llama.sh [DIR] - get llama.cpp at the pinned commit, apply the patches, build the tools. Needs git, cmake and a C/C++ compiler; downloads the llama.cpp sources from GitHub (run it yourself on the target PC).
# usage: bash tools/setup_llama.sh [target dir, default ./llama.cpp]    env: JOBS=4 (build jobs), LLAMA_COMMIT (default eec18f5), LLAMA_REPO, CMAKE_EXTRA (extra cmake flags, e.g. -DGGML_CUDA=ON)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); PATCHES="$HERE/../patches"
DIR=${1:-./llama.cpp}; COMMIT=${LLAMA_COMMIT:-eec18f5}; REPO=${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp}; JOBS=${JOBS:-$( (nproc 2>/dev/null || sysctl -n hw.physicalcpu 2>/dev/null || echo 4) )}
for t in git cmake; do command -v "$t" >/dev/null || { echo "missing $t: install it first (macOS: xcode-select --install and brew install cmake; Debian/Ubuntu: apt install git cmake build-essential)" >&2; exit 2; }; done
if [ ! -d "$DIR/.git" ]; then git clone "$REPO" "$DIR"; fi
cd "$DIR"
git fetch -q origin 2>/dev/null || true
git checkout -q "$COMMIT" 2>/dev/null || { echo "cannot check out $COMMIT (the patches are written for it)" >&2; exit 3; }
for p in "$PATCHES"/0*.patch; do
  if git apply --check "$p" 2>/dev/null; then git apply "$p"; echo "applied $(basename "$p")"
  elif git apply --check -R "$p" 2>/dev/null; then echo "already applied: $(basename "$p")"
  else echo "patch does not apply cleanly: $(basename "$p") (is the tree at $COMMIT and unmodified?)" >&2; exit 4; fi
done
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON ${CMAKE_EXTRA:-} >/dev/null   # GPU build: CMAKE_EXTRA="-DGGML_CUDA=ON" (NVIDIA) or "-DGGML_VULKAN=ON"
for tgt in llama-cli llama-completion llama-bench llama-perplexity llama-quantize; do cmake --build build --target "$tgt" -j "$JOBS" 2>&1 | tail -1 || true; done
echo "built: $(ls build/bin | grep -E '^llama-(cli|completion|bench|perplexity|quantize)$' | tr '\n' ' ')"
echo "next: python3 $HERE/moe_run.py --bin $(pwd)/build/bin --model YOUR_MODEL.gguf -- -p 'Hello' -n 64"
