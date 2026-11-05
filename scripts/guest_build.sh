#!/bin/bash
# Compile the guest's CUDA kernels to PTX with clang's NVPTX backend (no CUDA SDK): scripts/guest_build.sh [FILE.cu...]
# Output: out/guest/NAME.ptx for each. Host.
set -eu
cd "$(dirname "$0")/.."
mkdir -p out/guest
files=("$@")
if [ ${#files[@]} -eq 0 ]; then files=(src/guest/*.cu); fi
for f in "${files[@]}"; do
    name=$(basename "$f" .cu)
    clang -x cuda --cuda-device-only -nocudainc -nocudalib --cuda-gpu-arch=sm_70 -Xclang -target-feature -Xclang +ptx70 \
        -O2 -S -Wno-unknown-cuda-version -I src/guest "$f" -o "out/guest/$name.ptx"
    echo "host: $f -> out/guest/$name.ptx ($(/usr/bin/grep -c '^\s*\.visible \.entry' "out/guest/$name.ptx") kernels)"
done
