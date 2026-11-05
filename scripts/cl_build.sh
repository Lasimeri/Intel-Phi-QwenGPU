#!/bin/bash
# OpenCL C -> PTX for the embedded GPU: clang's OpenCL front-end, the device header src/cl/qgpucl.h instead of the
# default OpenCL headers (-cl-no-stdinc), NVPTX, -O2. Every function must inline: the driver lowers no calls (the
# kernel stubs clang emits are folded by the PTX parser). Host.
#   scripts/cl_build.sh FILE.cl...      -> out/cl/NAME.ptx
set -eu
cd "$(dirname "$0")/.."
mkdir -p out/cl
for f in "$@"; do
    name=$(basename "$f" .cl)
    clang -x cl -cl-std=CL2.0 -cl-no-stdinc -target nvptx64-nvidia-cuda -O2 -Werror -Wall -Wextra -I src/cl -I src/guest -S -o "out/cl/$name.ptx" "$f"
    echo "host: $f -> out/cl/$name.ptx ($(/usr/bin/grep -c '^.visible .entry' "out/cl/$name.ptx") kernels)"
done
