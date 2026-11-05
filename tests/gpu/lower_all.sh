#!/bin/bash
# Lower every kernel of the 2B with stand-in parameters (pointers to small buffers, constants, one runtime arg) and print
# its block count, registers and instructions, or the driver's reason for refusing it. Host.
set -u
cd "$(dirname "$0")/../.."
P=${1:-out/guest/q2b_kernels.ptx}  # a PTX of the 29 kernels: the CUDA C or the OpenCL C build
R=./out/host/ptxrun
fails=0
run() {  # NAME GX BX PARAMS...
    local name=$1 gx=$2 bx=$3
    shift 3
    local out
    out=$($R $P "$name" "$gx" 1 "$bx" 1 "$@" -v 2>&1 | /usr/bin/grep -E "^$name:|^unsupported" | head -1)
    case "$out" in unsupported*|"") fails=$((fails + 1));; esac
    printf '%-20s %s\n' "$name" "$out"
}
run embed              64 32  p:2048 c:0 p:64
run norm_a             64 32  p:2048
run norm_b              1 32
run norm_c             64 32  p:2048
run norm_d              1 32  c:2048
run norm_e             64 32  p:2048 p:2048 p:2048
run quant_blocks       64 32  p:2048 c:0
run gemv_h64           64 32  c:0 c:0 p:2048
run gemv_ffx192        64 32  c:0 c:0 p:2048
run gemv_ring64       192 32  c:0 c:0 p:64
run gemv_logits       256 32  c:0 c:0 a:0 -a 0
run silu_mul          192 32  p:64 p:64 p:64
run add_into           64 32  p:64 p:64
run conv_silu         192 32  p:64 p:64 p:64
run gdn_gates           1 32  p:64 p:64 p:64 p:64 p:64 p:64
run quant_vec128_unit  16 128 p:2048 p:2048 p:32 c:1
run quant_vec128_scale 16 128 p:2048 p:2048 p:32
run gdn_head           16 128 c:0 p:32 p:32 p:2048 p:2048 p:2048 p:32 p:32 p:32 p:32 p:32 p:2048 p:128 p:2048 p:2048 p:2048 p:128 p:2048 p:2048
run norm256             8 256 p:4096 c:512 p:256 p:2048
run rotate              1 256 p:2048 p:64
run quant_vec256        8 256 p:2048 c:0 p:32
run quant_kc            2 256 p:512 c:0
run quant_vt            2 256 p:512 c:0 p:64 c:64
run attn_scores         1 256 c:0 c:0 c:0 p:256
run attn_softmax        1 256 p:256 p:256
run attn_gq             1 256 p:256 c:0 c:0 c:0 c:64 c:70
run attn_pv             1 256 c:0 c:0 c:0 c:64 p:256 p:256 c:70
run step_pre            1 32
run step_end            1 32  p:64
echo "host: lower_all: $fails of 29 kernels refused"
exit $((fails > 0))
