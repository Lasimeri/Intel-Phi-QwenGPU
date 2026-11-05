# Intel-Phi-QwenGPU: index

A SIMT GPU (one streaming multiprocessor: 32-lane warps, thread blocks with barriers, a launch queue, memory banks)
meant to live inside the Qwen3.8-35B Q8_0 GGUF as added tensors, with the Qwen3.8-2B Q8_0 running on it through
OpenCL. Everything here runs on the host today, on a C reference of the machine; nothing is in the 35B's file yet and
nothing has run on the Xeon Phi 7120A yet. Read `docs/handoff.md` first for the state, then `docs/machine.md` for the
contract, `docs/opencl.md` for the API layer.

## Stages (status 2026-11-04 20:41, host; every proof rerun on the current image after the 20:38 compaction)

| stage | what | status |
|---|---|---|
| G1 | the reference machine (`src/gpu/sm_ref.c`) | DONE: `out/host/smtest` 0 failures |
| G2 | the driver: PTX -> machine blocks (`src/gpu/lower.cc`) | DONE: `tests/gpu/probe.sh` 48 cases (24 x 2 ROM bases) 0 failures |
| G3 | the 2B as kernels of the machine + its program | DONE on the reference: 8 of 8 tokens = the integer reference (`logs/g3/run3.txt`; rerun `run4.txt` identical) |
| G8a | OpenCL C front-end for the kernels | DONE: `tests/cl/probe_cl.sh` 30 cases 0 failures; the 2B's 29 kernels are OpenCL C |
| G8b | libqgpucl: OpenCL 1.2 subset over the machine | DONE: `out/host/clhello` |
| G8c | the 2B through the OpenCL API | DONE: `out/host/q2b_cl` 8 of 8 tokens |
| G8d | OpenGL subset (rasterization as kernels) | NOT STARTED |
| G4 | the machine as a circuit of tensors (`src/circuit/sm_circ.cc`, `docs/circuit.md`) | BUILT: 206,702 units, 141 layers; `circ_gate` predicts the reference's stream on the real job, 853,914 tokens so far (`logs/g4/gate5.txt`), each stop a found defect, fixed |
| G5 | emit the GGUF: 35B unchanged + chip.* tensors + banks | DONE 11-05 06:48: `out/model/q35-with-gpu.gguf` (41 GB real): the 35B's 753 tensors byte-identical (`cmp`), 2,904 chip tensors; `emit_gate` on it 2,999/2,999 steps and 94,791,000 dim values equal the evaluator; `gpu_run` reproduces the reference's stream (`logs/g5/merge-verify.txt`). A reply token out of the file: not yet (65 M steps a position, 89 to 140 steps/s on the host) |
| G6 | the card executor on the 7120A, live window | NOT STARTED |
| G7 | the bridge: the 35B calls the 2B | NOT STARTED |

Accuracy of the arithmetic itself (`tests/text/q2b_fix_check`): the GPU-exact integer 2B agrees with the exact 2B on
87.50 % of top-1 tokens over 288 positions (6 prompts); the unmodified integer pass had 93.40 %. Open.

## Tree

```
docs/machine.md            the contract: execution model, instructions, blocks, tokens, launch queue
docs/handoff.md            state, proofs, the seven root causes, how to continue every stage
docs/opencl.md             the OpenCL layer: device header, build, runtime, program replay
docs/qwenpu-*.md           inherited from Intel-Phi-QwenPU (the RISC-V project): reference only
src/gpu/sm.h               encodings (80-bit instruction rows, 64-bit LIVE word, queue words)
src/gpu/sm_ref.{h,c}       the reference machine; probe hooks (stop_qi, watch_lo/hi, trace_desc)
src/gpu/ptx.{h,cc}         PTX parser (clang's OpenCL kernel stubs folded into their bodies)
src/gpu/kinstr.h           the lowered instruction and its read/write sets
src/gpu/blocks.{h,cc}      placement into 6-stage blocks, liveness, live-out planning
src/gpu/lower.{h,cc}       PTX -> machine blocks: values, uniformity, registers, folding, encode, listing
src/gpu/image.{h,cc}       an image on disk (meta.txt rom.bin m.bin w.q8) or a bare machine, bound to SmImage
src/cl/qgpucl.h            the OpenCL C device header (work items, barrier, the machine's intrinsics)
src/cl/qgpucl.cc           libqgpucl: the OpenCL runtime
src/cl/qgpucl_ext.h        qgpuCreateBufferAt, qgpuStats
src/guest/q2b_map.h        the 2B's layout of M, A, S (word addresses the kernels and the builder share)
src/guest/q2b_kernels.cl   THE 2B kernels (29, OpenCL C)
src/guest/q2b_kernels.cu   the same kernels in CUDA C: frozen, kept as the second front-end's proof
src/guest/qgpu.h           the CUDA shim (intrinsics as inline PTX asm)
src/text/                  GGUF reader, tokenizer, q2b_fix (THE integer reference the kernels reproduce), q2b_ref /
                           q35_ref (the exact models), elem/vocab headers
src/tools/q2b_build.cc     builds the image: ROM instances, M constants and tables, W, the program (queue.txt,
                           program.txt, listing.txt)
src/tools/q2b_gpu_check.cc runs the image on the reference for a prompt, tokens vs q2b_fix
src/tools/q2b_gpu_probe.cc stage-by-stage comparison of M / A / S against q2b_fix's taps; traces
src/tools/ptxrun.cc        one kernel on the reference with small banks (the probe suites); -base N
src/tools/ptxdump.cc       parsed PTX
src/circuit/gate.{h,cc}    the gate language (features, adders, products, tables, registers); graph.{h,cc} the layer graph with
                           banks, the scheduler, the slot allocator with handovers; eval.{h,cc} the exact evaluator (threads);
                           sm_circ.{h,cc} THE MACHINE IN GATES; emit.cc the emitter (gpu_emit)
src/exec/membits.h         how an address selects bytes of each memory kind: shared by eval.cc and tforward.cc
src/exec/tforward.{h,cc}   the forward over the model file alone (norms, DeltaNet heads, shared expert, banks)
src/tools/circ_gate.cc     the circuit against the reference's stream (the evaluator must predict every token)
src/tools/emit_gate.cc     the model file against the reference's stream; --dims compares every dim with the evaluator
src/tools/gpu_run.cc       the runner: the file, a prompt, nothing else linked (nm: 0 machine symbols)
src/exec/chip_image.*, src/card/, src/tools/chip_*  copied from QwenPU for G6; not adapted yet
tests/gpu/smtest.c         the machine's own test
tests/gpu/probe.cu + probe.sh      the driver's known answers (CUDA C)
tests/cl/probe_cl.cl + probe_cl.sh  the same in OpenCL C
tests/gpu/lower_all.sh     every 2B kernel lowers (takes a PTX path)
tests/cl/clhello.c         a standard OpenCL host program on the embedded GPU
tests/cl/q2b_cl.c          the 2B through the OpenCL API (replays out/gpucl/program.txt)
tests/text/q2b_fix_check.cc  the arithmetic gate vs the exact 2B
scripts/cl_build.sh        OpenCL C -> out/cl/NAME.ptx
scripts/guest_build.sh     CUDA C -> out/guest/NAME.ptx
scripts/compdb.sh          compile_commands.json (QwenPU's; to adapt)
out/gpucl/                 THE image (OpenCL kernels): 532 ROM blocks, 982 descriptors, M 6,314,946 words, w.q8 2.0 GB
out/gpu/                   the CUDA-kernel image: same results, redundant (1.9 GB)
logs/g3/                   every measured run (fix-*.txt gates, probe*.txt stages, run*.txt tokens)
logs/g4/                   the circuit: classes.txt (stage layouts), gate*.txt (the circuit gate runs), circ-stats1.txt
logs/g5/                   the model file: emit*.txt, egate*.txt (file gate), run1.txt (gpu_run), *-hash.txt
out/model/q35-with-gpu.gguf  THE MODEL: the real 35B + the GPU (143 chip layers, hidden 16384, 67 banks, 209,157-token vocabulary)
out/model/chip-2b-v2.gguf  the GPU alone, same chip tensors (5.6 GB); chip-2b.gguf the one before the 06:07 fix: both redundant now
```

## Build and test (host; `/usr/bin/make`, never the shell's `make` alias)

```
/usr/bin/make all text gpu cl circuit model    # smtest ptxdump ptxrun | q2b_fix_check gguf_meta | q2b_build q2b_gpu_check q2b_gpu_probe | libqgpucl clhello q2b_cl | circ_gate | gpu_emit gpu_run emit_gate
./out/host/smtest                              # host: smtest: 1095 tokens, hash 260d9612934e2b8f, ... 0 failures
scripts/guest_build.sh tests/gpu/probe.cu && tests/gpu/probe.sh      # host: probe.sh: 0 failures (48 cases)
scripts/cl_build.sh tests/cl/probe_cl.cl && tests/cl/probe_cl.sh     # host: probe_cl.sh: 0 failures (30 cases)
scripts/cl_build.sh src/guest/q2b_kernels.cl && tests/gpu/lower_all.sh out/cl/q2b_kernels.ptx   # 0 of 29 refused
./out/host/q2b_build MODEL.gguf out/cl/q2b_kernels.ptx out/gpucl    # the image (writes w.q8 2.0 GB)
./out/host/q2b_gpu_probe out/gpucl MODEL.gguf "PROMPT" --positions 3 # 2005 ok, 2 skipped, 0 DIFFER (logs/g3/probe-cl12.txt)
./out/host/q2b_gpu_check out/gpucl MODEL.gguf 8 "PROMPT"             # tokens vs the reference
./out/host/clhello                                                   # y: 10 13 16 19
QGPU_IMAGE=out/gpucl ./out/host/q2b_cl out/gpucl/program.txt out/cl/q2b_kernels.ptx 8 TOKEN... -- EXPECTED...
```
MODEL.gguf = ~/models/Qwen3.8-2B-Distill/Qwen3.8-2B-Q8_0.gguf. The arithmetic gate (6 min, single thread):
`./out/host/q2b_fix_check MODEL.gguf 48 tests/text/prompts_ab.txt`.
