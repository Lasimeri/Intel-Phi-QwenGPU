#!/bin/bash
# Runs the 2B inside the 35B model file, and beside it the checker's hash for the same job: the reference's tokens per 4,096 against the model's.
# The check takes nothing from the run: chip_run links neither the reference nor the circuit. Host for now; the card is next. See docs/embedding.md.
#
#   scripts/embed_run.sh "PROMPT" [GEN] [LOGDIR]
set -uo pipefail
cd "$(dirname "$0")/.."
PROMPT=$1; GEN=${2:-64}; OUT=${3:-logs/t10}
M2=${MODEL_2B:-$HOME/models/Qwen3.8-2B-Distill/Qwen3.8-2B-Q8_0.gguf}
mkdir -p "$OUT"
OMP_NUM_THREADS=2 out/host/q2b64_check "$M2" out/q2b/q2b.pack out/guest64/q2b64.elf out/merged/system-prompt.txt out/merged/context.bin "$PROMPT" "$GEN" --nofix --positions 1 --hashlog "$OUT/ref-hash.txt" > "$OUT/ref.txt" 2>&1
tail -2 "$OUT/ref.txt" | cut -c1-200
exec out/host/chip_run out/merged/q35-with-2b.gguf "$PROMPT" --gen "$GEN" --hashlog "$OUT/run-hash.txt"
