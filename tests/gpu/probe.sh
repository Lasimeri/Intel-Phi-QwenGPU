#!/bin/bash
# The driver's known answers: upstream kernels and the probes, every output line compared. Host. Exit 1 on any difference.
set -u
cd "$(dirname "$0")/../.."
K=input/upstream/qwenGPU/kernels
P=out/guest/probe.ptx
R=./out/host/ptxrun
fails=0
EXTRA=${EXTRA:-}
check() {  # NAME EXPECTED COMMAND...
    local name=$1 want=$2
    shift 2
    local got
    got=$("$@" $EXTRA 2>&1 | /usr/bin/grep -E '^buffer|^unsupported' | tail -1)
    if [ "$got" != "$want" ]; then
        echo "FAIL $name: got '$got', want '$want'"
        fails=$((fails + 1))
    else
        echo "ok   $name"
    fi
}
run_suite() {
check vadd    "buffer 2: 7 9 11"                 $R $K/vadd.ptx vadd 1 1 32 1 p:3,2,3,4 p:3,5,6,7 p:3 c:3
check vadd-a  "buffer 2: 7 9 11"                 $R $K/vadd.ptx vadd 1 1 32 1 p:3,2,3,4 p:3,5,6,7 p:3 a:0 -a 3
check saxpy   "buffer 1: 16 -4 21"               $R $K/saxpy.ptx saxpy 1 1 32 1 c:5 p:3,3,-1,4 p:3,1,1,1 c:3
check matmul  "buffer 2: 19 22 43 50"            $R $K/matmul.ptx matmul 1 1 4 8 p:4,1,2,3,4 p:4,5,6,7,8 p:4 c:2
check colsum  "buffer 1: 11 22 33 44 55 66 77 88" $R $K/colsum.ptx colsum 1 1 32 1 p:16,1,2,3,4,5,6,7,8,10,20,30,40,50,60,70,80 p:8 c:8 c:2
check divk    "buffer 2: 2 -2 0 -1"              $R $K/divk.ptx divk 1 1 32 1 p:4,100,-100,7,-1 p:4 p:4 c:7 c:4
check udivk   "buffer 2: 0 5 7"                  $R $K/udivk.ptx udivk 1 1 32 1 p:3,100,4294967295,7 p:3 p:3 c:10 c:3
check fdivk   "buffer 2: -5 -2 0 -1"             $R $K/fdivk.ptx fdivk 1 1 32 1 p:4,100,-100,7,-1 p:4 p:4 c:-7 c:4
check divdec  "buffer 2: 33 -33 33"              $R $K/divdec.ptx divdec 1 1 32 1 p:3,10,-10,7 p:3 p:3 c:3 c:3
check bitops  "buffer 1: 1 -1640531513 77265352 -2049215672" $R $K/bitops.ptx bitops 1 1 32 1 p:4,1,-1,123456789,-987654321 p:4 c:4
check mulk    "buffer 1: -3 -6 -9$(printf ' 0%.0s' $(seq 1 37))" $R $K/mulk.ptx mulk 2 1 32 1 p:40,1,2,3 p:40 c:-3 c:40
check affine  "buffer 1: 10 13 16 19"            $R $P affine 1 1 32 1 p:4,1,2,3,4 p:4 c:4
check barsum  "buffer 1:$(printf ' 127%.0s' $(seq 1 64))" $R $P barsum 1 1 64 1 p:64,$(seq -s, 0 63) p:64 c:64
check reduce  "buffer 1: 820 40"                 $R $P reduce 2 1 32 1 p:40,$(seq -s, 1 40) p:2 c:40
check scalars "buffer 1: 142857 1000 983040"     $R $P scalars 1 1 32 1 p:5,1000000,0,7,196608,327680 p:3
check loopsum "buffer 1: 11 22 33 44 55 66 77 88" $R $P loopsum 1 1 32 1 p:16,1,2,3,4,5,6,7,8,10,20,30,40,50,60,70,80 p:8 a:0 a:1 -a 8,2
banks_want="buffer 0:"
for i in $(seq 0 31); do banks_want="$banks_want $((1000 * i + 98 + i + ((32 * (i + 1)) << 16)))"; done
for i in $(seq 0 31); do banks_want="$banks_want 16777216"; done
check banks   "$banks_want"                      $R $P banks 1 1 32 1 p:64 c:32
check shifts  "buffer 1: 3 -3 2 20 -6 -4"        $R $P shifts 1 1 32 1 p:3,100,-100,131072 p:6
check ushifts "buffer 1: 20 1073741820 -4 75"    $R $P ushifts 1 1 32 1 p:4,5,-16,-16,100 p:4 a:0 -a 2
check gdnunit  "buffer 0:$(printf ' 129258%.0s' $(seq 1 32))" $R $P gdnunit 1 1 32 1 p:32
check gdnbound1 "buffer 1: 18 0 384472656 0"      $R $P gdnbound 1 1 32 1 p:5,0,32768,15,24800,8000000 p:4
check gdnbound2 "buffer 1: 17 0 776422656 391950000" $R $P gdnbound 1 1 32 1 p:5,26130,30000,18,24800,8000000 p:4
check blockmax "buffer 1:$(printf ' -1%.0s' $(seq 1 128))$(printf ' -129%.0s' $(seq 1 128))" $R $P blockmax 2 1 128 1 p:256,$(seq -s, -1 -1 -256) p:256
check smaxblock "buffer 0:$(printf ' 4%.0s' $(seq 1 128))" $R $P smaxblock 1 1 128 1 p:128
}
run_suite
EXTRA="-base 9" run_suite  # the kernel behind 9 filler blocks: every thread block must start at the entry
echo "host: probe.sh: $fails failures"
exit $((fails > 0))
