# Intel-Phi-QwenGPU

A SIMT GPU compiled into the weights of Qwen3.8-35B, with Qwen3.8-2B running on it. The machine is a circuit of
SwiGLU units, DeltaNet heads and banks (memories written compactly); one forward pass of the added layers is one step
of the machine; the model file carries the 35B byte for byte, then the GPU as `chip.*` tensors. Host work so far; the
target card is a Xeon Phi 7120A.

| stage | state (2026-11-05) |
|---|---|
| G1 reference machine (`src/gpu/sm_ref.c`) | `tests/gpu/smtest.c` 0 failures |
| G2 driver: PTX to machine blocks (`src/gpu/lower.cc`) | `tests/gpu/probe.sh` 48 cases 0 failures |
| G3 the 2B as kernels of the machine | 8 of 8 tokens = the integer reference (`logs/g4/run6.txt`) |
| G8a-c OpenCL C front-end, libqgpucl, the 2B through the OpenCL API | `tests/cl/probe_cl.sh` 30 cases 0 failures; `tests/cl/q2b_cl.c` 8 of 8 tokens |
| G4 the machine as a circuit (`src/circuit/sm_circ.cc`) | 206,699 units, 67 banks, 141 layers; `circ_gate` predicts the reference's stream, 2,999,999 of 2,999,999 steps (`logs/g4/gate7.txt`) |
| G5 the model file (`src/circuit/emit.cc`) | `q35-with-gpu.gguf`: the 35B's 753 tensors byte-identical + 2,904 chip tensors; `emit_gate` 2,999 of 2,999 steps and 94,791,000 dim values equal the exact evaluator; `gpu_run` reproduces the reference's stream (`logs/g5/merge-verify.txt`) |
| G6 card executor, G7 bridge, G8d OpenGL subset | not started |

Not yet produced: a reply token out of the model file (65 M steps per 2B position at about 100 steps/s on the host).

## Read
- `docs/INDEX.md`: the tree, the build, the tests, every log.
- `docs/handoff.md`: state, proofs, root causes, how to continue.
- `docs/machine.md`: the machine's contract (execution model, instructions, blocks, tokens, launch queue).
- `docs/circuit.md`: the circuit's design, the commit rule its gates enforced, numbers.
- `docs/opencl.md`: the OpenCL layer.

## Build and gate (host)
```
/usr/bin/make all text gpu cl circuit model
./out/host/q2b_build MODEL_2B.gguf out/cl/q2b_kernels.ptx out/gpucl
./out/host/q2b_gpu_check out/gpucl MODEL_2B.gguf 8 "PROMPT"
./out/host/circ_gate out/gpucl MODEL_2B.gguf 8 "PROMPT" --max 3000000
./out/host/gpu_emit out/model/q35-with-gpu.gguf out/gpucl --base MODEL_35B.gguf
./out/host/emit_gate out/model/q35-with-gpu.gguf out/gpucl MODEL_2B.gguf 8 "PROMPT" --max 3000 --dims
./out/host/gpu_run out/model/q35-with-gpu.gguf "PROMPT" --gen 8 --max 20000 --hashlog run-hash.txt
```
Models are not in this repository: Qwen3.8-2B Q8_0 and Qwen3.8-35B-A3B Q8_0 GGUF files. The image (`out/gpucl`) is
built from the 2B; everything under `out/` is build output.

## Provenance
The gate language, the graph, the evaluator and the embedding form are ports of QwenPU and chipformer (MIT): see
`NOTICE.qwenpu`. The OpenCL kernels of the 2B were written for this machine.
