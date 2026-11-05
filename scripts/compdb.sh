#!/bin/bash
# Writes compile_commands.json for the whole tree and checks every source with clangd: a build is not
# clean while the editor shows a diagnostic. The card's VPU code really compiles for -march=knc
# (scripts/card_build.sh, scripts/q35_build.sh), which only the Intel-Phi-LLVM clang knows and the system
# clangd rejects; its entries here give clangd the AVX-512F view, with src/card/clangd_knc.h for the rest.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
: "${PHI_LLVM_BIN:=$HOME/.cache/intel-phi-llvm-build/llvm/bin}"
: "${PHI_SYSROOT_7120A:=$HOME/.cache/intel-phi-7120a-build/toolchain/sysroot}"
: "${PHI_TREE:=/media/1TB-NVMe/Xeon-Phi/Intel Phi 7120A}"
out=compile_commands.json
first=1
entry() { # file, command
    [ "$first" = 1 ] || printf ',\n' >> "$out"
    first=0
    printf '{"directory":"%s","file":"%s","command":"%s"}' "$ROOT" "$1" "$2" >> "$out"
}
printf '[\n' > "$out"
CFLAGS="-std=c11 -O2 -Wall -Wextra -Werror -D_GNU_SOURCE -ffp-contract=off"
CXXFLAGS="-std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off"
for f in src/rv/*.c src/exec/ref.c src/tools/rvrun.c src/tools/rvteam.c src/tools/export_vx.c tests/rv/*.c; do entry "$f" "cc $CFLAGS -c $f"; done
for f in src/circuit/*.cc src/exec/*.cc src/tools/*.cc; do entry "$f" "g++ $CXXFLAGS -c $f"; done
entry src/card/exec_vpu.c "clang -mavx512f -std=gnu11 -O2 -Wall -Wextra -Werror -ffp-contract=off -c src/card/exec_vpu.c"
entry src/card/run_main.c "cc -std=gnu11 -O2 -Wall -Wextra -Werror -ffp-contract=off -c src/card/run_main.c"
# the text path: the host's programs, and the card's through the AVX-512F view plus clangd_knc.h
REF="-std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off -mfma -mavx2 -mf16c -fopenmp"
KNCVIEW="clang -mavx512f -include src/card/clangd_knc.h -std=gnu11 -O2 -Wall -Wextra -Werror -ffp-contract=off"
for f in src/text/*.cc tests/text/tok_gate.cc tests/text/ref_check.cc tests/text/q35_verify.cc tests/text/elem_gate_host.cc tests/text/align_check.cc tests/text/q2b_check.cc tests/text/q2b_sweep.cc tests/text/q2b_fix_check.cc tests/text/q2b_cardtool.cc; do entry "$f" "g++ $REF -c $f"; done
entry tests/text/llama_dump.cpp "g++ -std=c++17 -O2 -Wall -Wextra -I$HOME/llama.cpp/include -I$HOME/llama.cpp/ggml/include -c tests/text/llama_dump.cpp"
entry tests/text/route_sim.c "cc -std=gnu11 -O2 -Wall -Wextra -Werror -c tests/text/route_sim.c"
entry src/card/q35_vpu.c "$KNCVIEW -c src/card/q35_vpu.c"
entry tests/card/elem_gate_vpu.c "$KNCVIEW -c tests/card/elem_gate_vpu.c"
entry src/card/q35_elem.h "$KNCVIEW -x c-header -c src/card/q35_elem.h"
entry src/card/q35_main.c "cc -std=gnu11 -O2 -Wall -Wextra -Werror -ffp-contract=off -c src/card/q35_main.c"
entry src/card/q2b_vpu.c "cc -std=gnu11 -O2 -Wall -Wextra -Werror -ffp-contract=off -c src/card/q2b_vpu.c"
entry src/card/q2b_main.c "cc -std=gnu11 -O2 -Wall -Wextra -Werror -ffp-contract=off -c src/card/q2b_main.c"
# the embedded engine on the card, and the windows that watch it
entry src/card/chip_vpu.c "$KNCVIEW -fno-strict-aliasing -c src/card/chip_vpu.c"
entry src/card/chip_main.c "cc -std=gnu11 -O2 -Wall -Wextra -Werror -ffp-contract=off -c src/card/chip_main.c"
entry src/card/q35_dma.c "cc -std=gnu11 -O2 -Wall -Wextra -Werror -ffp-contract=off -c src/card/q35_dma.c"
entry src/tools/chip_watch.c "cc -std=c11 -O2 -Wall -Wextra -Werror -D_GNU_SOURCE -c src/tools/chip_watch.c"
entry src/tools/native_watch.c "cc -std=c11 -O2 -Wall -Wextra -Werror -D_GNU_SOURCE -c src/tools/native_watch.c"
entry tests/card/elem_gate_main.c "cc -std=gnu11 -O2 -Wall -Wextra -Werror -c tests/card/elem_gate_main.c"
entry src/guest/stories_prompt.c "clang --target=riscv32-unknown-elf -march=rv32im_zba -mabi=ilp32 -O2 -Iinput/upstream/QwenPU/chipformer/rv/env -Iout/guest -Iinput/upstream/QwenPU/guests/stories260k -ffreestanding -c src/guest/stories_prompt.c"
entry src/guest/rt_min.c "clang --target=riscv32-unknown-elf -march=rv32im_zba -mabi=ilp32 -O2 -Iinput/upstream/QwenPU/chipformer/rv/env -ffreestanding -fno-builtin -c src/guest/rt_min.c"
# the RV64 machine and the NPU on the host, and the guests as RV64
HOST64="cc -std=gnu11 -O2 -Wall -Wextra -Werror -D_GNU_SOURCE -ffp-contract=off"
for f in src/rv64/*.c src/npu/*.c src/tools/rv64run.c src/tools/rv64npu.c tests/rv64/chiptest.c; do entry "$f" "$HOST64 -c $f"; done
for f in src/guest64/*.c tests/rv64/vxtest.c; do entry "$f" "clang --target=riscv64-unknown-elf -march=rv64i_zmmul_zba -mabi=lp64 -mcmodel=medany -O2 -ffreestanding -nostdlib -fno-builtin -Wall -Wextra -Werror -c $f"; done
printf '\n]\n' >> "$out"

# One file: prints its diagnostics, nothing when clean. A clangd that fails without a diagnostic is reported too, never read as
# clean. --check-locations=0 drops clangd's hover and tweak pass at every token, which is no diagnostic: core64.cc 10.45 s ->
# 0.54 s, and the same 2 diagnostics on a file that had 2 when this was measured.
check_one() {
    local log rc=0 found
    log=$(clangd --check="$1" --check-locations=0 --log=error 2>&1) || rc=$?
    found=$(printf '%s\n' "$log" | /usr/bin/grep -aE '^(E|V)\[|error:|warning:' | /usr/bin/grep -avE 'tweak|Ignored' || true)
    [ "$rc" = 0 ] && [ -z "$found" ] && return 0
    printf 'DIAGNOSTICS %s: exit %d\n%s\n' "$1" "$rc" "$(printf '%s\n' "$found" | head -5 | cut -c1-170)"
}
export -f check_one
files=(src/rv/*.c src/exec/ref.c src/exec/*.cc src/circuit/*.cc src/text/*.cc src/tools/*.c src/tools/*.cc tests/rv/*.c tests/text/*.cc tests/text/*.cpp tests/text/*.c tests/card/*.c
    src/card/run_main.c src/card/exec_vpu.c src/card/q35_vpu.c src/card/q35_main.c src/card/q35_elem.h src/card/q2b_vpu.c src/card/q2b_main.c src/guest/*.c
    src/card/chip_vpu.c src/card/chip_main.c src/card/q35_dma.c
    src/rv64/*.c src/npu/*.c tests/rv64/*.c src/guest64/*.c)
report=$(printf '%s\0' "${files[@]}" | xargs -0 -P "${JOBS:-14}" -I{} bash -c 'check_one "$1"' _ {})
[ -z "$report" ] || printf '%s\n' "$report"
bad=$(printf '%s\n' "$report" | /usr/bin/grep -c '^DIAGNOSTICS' || true)
echo "clangd over the tree: ${#files[@]} files checked, $bad with diagnostics"
exit "$bad"
