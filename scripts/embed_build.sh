#!/bin/bash
# Writes the model file: Qwen3.8-35B-A3B Q8 unchanged, with the RISC-V machine and the NPU as its own tensors, the 2B's weights and program as
# their state, and SYSTEM_FILE as the system prompt, with its context. On the host, once for a system prompt. See docs/embedding.md.
#
#   scripts/embed_build.sh [SYSTEM_FILE]        default: ~/CLAUDE.md
set -euo pipefail
cd "$(dirname "$0")/.."
SYS=${1:-$HOME/CLAUDE.md}
M2=${MODEL_2B:-$HOME/models/Qwen3.8-2B-Distill/Qwen3.8-2B-Q8_0.gguf}
M35=${MODEL_35B:-$HOME/models/Qwen3.8-35B-A3B/Qwen3.8-35B-A3B-Q8_0.gguf}
mkdir -p out/merged logs/t10
scripts/rv64_build.sh
make out/host/q2b_pack out/host/emit64 out/host/chip_run out/host/q2b64_check 2>&1 | /usr/bin/grep -aE 'error|\*\*\*' || true
[ -s out/q2b/q2b.pack ] || out/host/q2b_pack "$M2" out/q2b/q2b.pack
if ! cmp -s "$SYS" out/merged/system-prompt.txt; then
    cp "$SYS" out/merged/system-prompt.txt
    rm -f out/merged/context.bin
fi
OMP_NUM_THREADS=14 out/host/emit64 out/merged/q35-with-2b.gguf "$M35" "$M2" out/q2b/q2b.pack out/guest64/q2b64.elf out/merged/system-prompt.txt out/merged/context.bin
