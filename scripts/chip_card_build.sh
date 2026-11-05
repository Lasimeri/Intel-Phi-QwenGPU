#!/bin/bash
# Builds the card's program for the merged model: the VPU forward with the KNC clang, main with the stack's knc-cc, then the ISA audit, and the audit of the
# form: no interpreter and no NPU symbol may be in it. The stack is a tool only; nothing is written into it.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PHI_TREE:=/media/1TB-NVMe/Xeon-Phi/Intel Phi 7120A}"
: "${PHI_LLVM_BIN:=$HOME/.cache/intel-phi-llvm-build/llvm/bin}"
: "${PHI_SYSROOT_7120A:=$HOME/.cache/intel-phi-7120a-build/toolchain/sysroot}"
OUT=out/card
mkdir -p "$OUT"
vpu=(--target=x86_64-unknown-linux-musl -march=knc "--sysroot=$PHI_SYSROOT_7120A" -std=gnu11 -O2 -Wall -Wextra -Werror
     -ffp-contract=off -fno-strict-aliasing -fno-jump-tables -falign-loops=1 -fPIE)
. scripts/par.sh
par "$PHI_LLVM_BIN/clang" "${vpu[@]}" -c -o "$OUT/chip_vpu.o" src/card/chip_vpu.c
par "$PHI_LLVM_BIN/clang" --target=x86_64-unknown-linux-musl -march=knc -c -o "$OUT/chip_sell.o" src/card/sell.S
par_wait
(
    # shellcheck disable=SC1091
    . "$PHI_TREE/toolchain/env.sh"
    knc-cc -std=gnu11 -O2 -Wall -Wextra -Werror -ffp-contract=off -static -pthread -o "$OUT/chip_card" \
        src/card/chip_main.c src/card/q35_dma.c "$OUT/chip_vpu.o" "$OUT/chip_sell.o"
)
"$PHI_TREE/host/target/debug/phi-isa-audit" "$OUT/chip_card" | /usr/bin/grep -a '^result:'
if nm "$OUT/chip_card" 2>/dev/null | /usr/bin/grep -aE ' (m64_|npu_|Evaluator|RvCore|build_core)'; then
    echo "card: refused: an interpreter or an NPU is linked" >&2
    exit 3
fi
ls -la "$OUT/chip_card" | cut -c25-
