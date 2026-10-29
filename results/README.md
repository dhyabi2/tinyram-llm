# results/

Raw result files. Each folder is one experiment; the JSON files carry the machine, the model and the conditions.

- `colab-gpt-oss-20b`, `vps-qwen3-30b-a3b`, `mac-qwen3.8-27b-dense`, `mac-disk-read-size-sweep`: early plain-mmap baselines and a disk read-size sweep.
- `mac-pool-vs-mmap`: Mac, memory squeezed with a wired ballast: plain mmap vs explicit read pool vs overlapped pool (and a dense-weight-lock check).
- `vps-madvise-ab`: Linux VPS, cold start, madvise read-ahead on the chosen experts vs off, with a null pair.
- Folders named `<host>-<date>` (from `docs/RUN_WITH_CLAUDE.md` / `docs/WINDOWS_PORT_PROMPT.md`) come from other machines.
