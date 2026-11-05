# Hand-off: Intel-Phi-QwenGPU, 2026-11-05 04:4x

## READ THIS FIRST: the state at 2026-11-05 05:5x (host; the card untouched; the real 35B untouched)
The order of work set on 11-04 20:49: get everything in place before the model with the GPU embedded is created; the GL
subset waits. In place now:
- **Contract v2 (lane-serial)**, `docs/machine.md`, memory `qwengpu-lane-serial-decision`: the reference runs the 2B's
  job with 8/8 tokens, 1,688,967,096 trace tokens, hash ee9fcde2143e34ff (`logs/g4/run6.txt`); smtest, probe.sh,
  probe_cl.sh 0 failures; stage classes {49, 63, 49, 63, 49, 127} (`logs/g4/classes.txt`).
- **The circuit (G4)**: `src/circuit/sm_circ.cc`, described in `docs/circuit.md`: 206,702 units, 23,449 nodes, 67
  banks, 141 layers. `circ_gate` makes the exact evaluator predict every token of the reference's stream on the real
  job: 2,999,999 of 2,999,999 tokens agreed to the 3 M limit (`logs/g4/gate7.txt`, 1,373 passes/s on 13 threads,
  2,184 s): the first run to its limit. Every earlier stop (tokens 81,975; 704,436; 689,072; 828,728; 853,915) was a
  defect of one class, fixed: state a part commits early is seen by the part's own recomputation passes (memory
  `qwengpu-recomputation-rule`): stores, S writes, reduction words, register commits, scfg all wait for the part's end
  now, each pending word filled in one pass, fired in one later pass, cleared there. A ONE-POSITION run (66 M tokens,
  the first EMIT of a real 2B token out of the circuit, about 13 h) started 06:45: `logs/g4/gate8-position.txt`.
- **The model file (G5 prerequisites)**: `make model` builds `gpu_emit` (`src/circuit/emit.cc`: the circuit as chip.*
  tensors, slots shared by lifetime with handover units, banks with any number of ports, memories chip.mem.K, hidden
  16384 with 16 ballast dims so c = 2^-35, 143 layers = 141 + 2 exit clear layers, the 209,157-token vocabulary;
  `--base MODEL_35B` puts the 35B's metadata and tensors first, byte for byte), `tforward` (the forward over the file
  alone; `src/exec/membits.h` is the one place that says how an address selects bytes, shared with the evaluator),
  `emit_gate` (the file must predict the reference's stream; `--dims` compares every dim at the sublayer after its
  producer with the evaluator and auto-probes the lowest differing one), `gpu_run` (the file and a prompt, nothing of
  the machine linked: nm 0 symbols). RESULT: the chip-only file `out/model/chip-2b.gguf` (5.6 GB real, 401.6 GB
  logical, sparse) predicted 2,999 of 2,999 tokens and 94,800,000 dim values equal the evaluator's
  (`logs/g5/egate4.txt`); the file predicted 199,999 of 199,999 tokens over 200,000 (`logs/g5/egate5.txt`, 103 passes/s single-threaded
  with three other runs on the host); `gpu_run` from the file alone produced the same 200,000-token stream: 48 of 48
  hash points equal (`logs/g5/ref-hash.txt` vs `run-hash.txt`, FNV-1a every 4,096 tokens), 103 tok/s. After the
  06:07 circuit fix the file is `out/model/chip-2b-v2.gguf` (its gate: 2,999/2,999 tokens, 94,791,000 dim values
  equal, `egate6.txt`).
- **THE MERGED MODEL (G5), 06:48**: `out/model/q35-with-gpu.gguf` = `gpu_emit OUT out/gpucl --base
  ~/models/Qwen3.8-35B-A3B/Qwen3.8-35B-A3B-Q8_0.gguf` (`logs/g5/merge1.txt`): the 35B's 753 tensors and 26 int keys
  first, then 2,904 chip tensors and 27 chip keys (3,657 tensors, 53 int keys); 41 GB real, 439.43 GB logical (sparse).
  Verified (`logs/g5/merge-verify.txt`): `cmp` of the 35B's 37,791,158,784 data bytes against the original: identical;
  `emit_gate` on the merged file: 2,999/2,999 steps, 94,791,000 dim values equal the evaluator (`merge-egate.txt`);
  `gpu_run` with the file's OWN tokenizer (the 35B's): 19 prompt tokens, 20,000 steps, 4 of 4 hash points equal the
  reference's stream (`merge-hash.txt` vs `ref-hash.txt`). Not verified: llama.cpp loading it as a 35B (no local binary). The NVMe is at 97 % (29 GB free); the chip-only files `chip-2b.gguf` and `chip-2b-v2.gguf` are now
  redundant (11 GB).
- **Not done**: a reply token through the file (a position is about 65 M
  passes: 5 days at 140 passes/s), depth reduction (141 layers), the card (G6), the bridge (G7), the GL subset (G8d),
  the ICD, the arithmetic gap (87.50 % vs 93.40 %).
- Build: `/usr/bin/make all text gpu cl circuit model`. Gates: `circ_gate out/gpucl MODEL_2B 8 "PROMPT" [--max N]
  [--watch ADDR]`, `emit_gate out/model/chip-2b.gguf out/gpucl MODEL_2B 8 "PROMPT" [--max N] [--dims] [--probe DIM]
  [--hashlog F]`, `gpu_run out/model/chip-2b.gguf "PROMPT" --tok MODEL_2B --gen 8 [--max N] [--hashlog F]`.
  QGPU_CLASSES must agree between the image and every circuit tool (the default does).

## The hand-off as of 2026-11-04 20:41 (written 20:34; reviewed after the 20:38 compaction with every proof rerun on the image of that time)

## 21:48 addendum: the contract changed for the circuit (G4a), gate green again
User 20:49: "Get what's needed in place before the model is created with the GPU embedded inside it": G4 (the circuit)
and the emitter's prerequisites come now; the GL subset waits. Sizing showed a lane-parallel record (6 stages x 32
lanes in one forward pass) would be about 32x the RV64 chip in gates, beyond this disk; the contract is now
**lane-serial** (docs/machine.md, memory `qwengpu-lane-serial-decision`): one lane's stages per pass, 32 lane parts a
record, LSTEP for lanes without live-outs, reductions' running partials in the lane parts (at most 2 a block), WARP(w)
tokens after END/BAR, `sup` carrying max |S| in rd (the `smax` tracker is gone), late results (reductions, udiv, usqrt)
readable in a block only by uniform instructions, no sdot/ld S after a sup in a block, no double write of an S element
by a lane in a block, no load of what an earlier lane of the block stored (the reference faults on each). Measured on
the real job after the change (image build4, ROM 541 blocks): `q2b_gpu_check` 8/8 tokens, 17,541,588 records,
1,242,184,604 trace tokens (1.1565x the lane-parallel 1,074,098,578; 47,776,331 per position), hash 42bf841d7d10bb6d,
40.9 s (`logs/g4/run4.txt`); probe 2005 ok, 0 DIFFER over 3 positions (`probe4.txt`); `q2b_cl` every token as expected,
1,299,046,431 trace tokens (`cl-run4.txt`); smtest 0 failures (1674 tokens); probe.sh and probe_cl.sh 0 failures.
Bug found on the way (4/8 tokens, constant 0.1856 ratio at norm256's output): the fold macros' `WARP_IN_BLOCK == 0`
test reached the lowerer as a per-lane compare, so the read-modify-write of the fold word ran per lane and lane L + 1
re-read lane L's store; fix: a setp on a thread-index field is a uniform predicate (lower.cc Val.puni), so the fold's
selp and stv are uniform and written once after all lanes. Circuit work in progress: `src/circuit/gate.{h,cc}` (the
gate language, ported), banks and the SIMT step next (G4b), then the exact evaluator gate against sm_ref's stream.

## 22:46 addendum: the circuit exists and predicts the real stream (G4b, G4c partly)
`src/circuit/sm_circ.cc` is the SIMT machine in gates (one lane's stages per pass; scratch words read low and written
high carry the computed state; token-written registers carry what the stream tells; a lane's commits come from its own
VAL tokens); `src/circuit/eval.{h,cc}` the exact evaluator with banks (multi-port, lagged write channels, a thread pool
over each level's units); `src/circuit/graph.{h,cc}` gained bank nodes; `src/tools/circ_gate.cc` runs the reference on
the real job and makes the circuit predict every token. Contract changes on the way (docs/machine.md): three chunk
families VLO/VHI/VEND, WARP tokens, reductions' partials in the lane parts, `sup` carrying max |S| in rd (smax gone),
stage classes {49, 63, 49, 63, 49, 127} by measurement (`logs/g4/classes.txt`). Results: 205,877 units, 23,019 nodes,
66 banks, 141 layers; 299,999 of 299,999 tokens predicted on the job (`--max 300000`, 1,485 passes/s); a 10 M token run
is in `logs/g4/gate2.txt`. One bug found by the gate at token 81,975: the A/S/C store banks defaulted to memory M (the
quantizer's bytes overwrote the queue words); fixed. Shape (`logs/g4/circ-stats1.txt`): the 30,020 dims need 8,592
residual slots with sharing, so the emitter must use hidden 16384 or the circuit must shrink; 141 layers is deep (the
six stages' forwarding chains): both to be attacked before G5. Not started: the emitter (port QwenPU's emit64.cc to
Graph + banks), tforward with the new bank semantics, the full-position gate (about 65 M tokens, 12 h at this rate).

## What this is
A SIMT GPU for the inside of the Qwen3.8-35B Q8_0 GGUF, and the Qwen3.8-2B Q8_0 as its first program, driven through
OpenCL. The ordering set on 11-04 19:44: OpenGL + OpenCL support on the embedded GPU come FIRST; the 2B then
runs on it through them; the embedding into the 35B (G4 circuit, G5 GGUF emit), the card (G6) and the bridge (G7)
follow. Nothing replaces those: "inside the 35B on the embedded GPU" still needs them.

Where it stands: the GPU exists as a C reference machine on the host (`src/gpu/sm_ref.c`), with a PTX lowerer, an
OpenCL C front-end, an OpenCL 1.2-subset runtime, and the 2B running bit-exactly against its integer reference both as
a self-contained program (the launch queue in the image) and through the OpenCL API. OpenGL has not been started.
Nothing has run on the Xeon Phi 7120A; nothing is in the 35B's file.

## Proven (host, every number from a logged run)
- Machine: `out/host/smtest`: 1095 tokens, hash 260d9612934e2b8f, 18 records, 69 instructions, 4 launches, 0 failures.
- Driver: `tests/gpu/probe.sh` 0 failures (24 known-answer kernels, each at ROM base 0 and 9); `tests/cl/probe_cl.sh`
  0 failures (15 cases x 2 bases, OpenCL C).
- The 2B's program on the reference (`logs/g3/run3.txt` 20:18 and `run4.txt` 20:40 on the current image, identical to
  the record; image out/gpucl, prompt "Write a haiku about rain.", 19 prompt
  tokens, 8 generated): 8 of 8 tokens identical to q2b_fix teacher-forced on the machine's tokens
  ("Rain falls softly,\nGreen leaves turn" = 57044 16563 54166 11 198 18978 10583 2404); 1,074,098,578 trace tokens
  over 26 positions = 41,311,484 per position (46,361,227 for a generated position with the 248,320-row logits,
  39,142,027 for a prompt position); 17,155,540 records; 24,818 launches; hash 31b74e47292e17d3; 29.6 s wall
  including the reference, 36.2 M trace tok/s.
- Stage probe (`logs/g3/probe-cl12.txt`, 20:40, current image, `--positions 3`): 669 checks a position (every layer's
  norm, projections, GDN inputs, state, head output, attention q/k/v blocks, scores, softmax, probability blocks,
  FFN, residuals, output norm, argmax); 2005 ok, 2 skipped (result_output at the two prompt positions before the
  last: the machine computes no logits there), 0 DIFFER. The per-head attention taps (attn_p, attn_e, f_q, f_xs,
  f2_q, f2_xs) hold the positions attended so far, so at position p they compare p + 1 words (1 at position 0); the
  log's "(reference has 1 values, the check 65536 words)" lines say that and nothing more. `probe-cl11.txt` (20:17)
  shows 2 DIFFER lines: the probe then still compared result_output at prompt positions ("machine best 0 at 0"), a
  probe defect fixed afterwards, not a machine defect; the 20:34 hand-off cited that log for 0, which it does not show.
- The 2B through OpenCL (`tests/cl/q2b_cl.c`, QGPU_IMAGE=out/gpucl; `logs/g3/cl-run1.txt`, 20:40): 8 of 8 tokens as
  expected; 24,818 launches, 24,905 program steps, 17,612,035 records, 1,119,514,453 trace tokens (4.2 % above the
  image's own program: runtime pointer args reload), 22.4 s wall at 20:31 and 23 s at 20:40.
- Arithmetic vs the exact 2B (`tests/text/q2b_fix_check`, 6 prompts, 288 positions): GPU-exact integer pass 87.50 %
  top-1, 99.65 % top-5, mean rank 0.170 (`logs/g3/fix-gpu6.txt`); the unmodified integer pass 93.40 % / 100 %
  (`fix-base.txt`). 32-bit wraps counted: 0 of gemv acc, kv24, v24, rms24. The 5.9-point loss is arithmetic: the
  candidates are the attention probability quantization (g per 32 positions + residual block), scores per block then
  >> 4, pr by reciprocal, RoPE as two mulsh, the norm's rs via div48, the embedding's mulsh by 8. Not bisected.

## The machine as it is now (docs/machine.md; the circuit G4 must implement exactly this)
32 lanes; r0..r15 (r15 = 0); u0..u15 uniform PER WARP SLOT (ureg[8][16], slot = warp mod 8; u15 = 0; u14 = the
launch's args block from descriptor word 1, set for every slot); 6 stages a block, LMAX 4 live-outs, 4096 ROM blocks,
up to 256 warps a launch (the 8-bit field writes 256 as 0), wpb <= 8. Order: thread blocks one after another, each
from the entry block; within a barrier segment the warps run one after another through the segment's blocks; a
warp's live-out store is in memory before the next warp's loads. Banks M (int32), A (Q8 blocks, 36 B), S (int16), W
(GGUF Q8_0 blocks, 34 B, f16 scale -> Q.24 table), C (int8). q8dot's immediate form holds the whole A index and adds
nothing to the W address; the register form advances both by imm. Tokens VAL/EXEC/NEXT/EMIT/HALT/GO as before.

## The seven root causes of this debugging (keep them: each is a class)
1. sm_ref: every 2nd+ thread block of a launch started at ROM block 0, not the entry (smtest/probe never had a
   multi-block launch at entry > 0). Fix: SmMachine.entry. Regression: ptxrun -base N, suites at two bases.
2. lower.cc encode: a register shift (sh = 63) had use_rc false: the amount came from lane r0. Found by the stage
   probe (first DIFFER = the norm) and the listing (`shl u2 u3 <<63` without rc). Regression: probe `ushifts`.
3. One global uniform register file, but the warps of a thread block run a segment one after another: warp 0's later
   writes clobbered the segment's live-in registers for warps 1-3. Fix: a uniform file per warp slot (CUDA semantics).
   Found with --trace-store (store + uniform file) and --trace-desc (uniform dump after every block of warp 0).
4. q2b_map.h sized M_KREAL at 128 words while the kernel writes 16 heads x 128: heads 1-15 overwrote M_MIX. Found by
   instrumenting the last two lines (`on`, `silu(z)`, both exact) against M_MIX's garbage. Lesson: every per-head region
   indexed by hd * ST is VHEADS x ST.
5. The immediate form of q8dot added the immediate (the A index) to the W address as well: block 0 of every gemv with a
   nonzero A base read row r+4's block 0 (+-1 rows in (r, r+4) pairs). Found by recomputing one row from the image's
   bytes (`recompute_row` in the probe). Fix in the machine + contract.
6. attn_scores / attn_softmax folded with `base == 0 && group == 0` in a launch of one work-group of 8 warps: every
   warp restarted the fold. Fix: the first WARP starts it. Found by the SC_SUM store trace (warps 1-7 storing 0).
7. The descriptor's 8-bit warps field: a 256-warp launch (each full 8192-row logits chunk) was written as 0 and ran
   nothing; 30 of 31 chunks never ran. Fix: 0 means 256. Found by the SC_BEST store trace (stores only in the last
   chunk's descriptor).
Method that found all seven: bit-exact stage comparison (q2b_fix taps <-> M / A / S regions after each descriptor),
then the listing, then store/uniform traces, then a recomputation from the image's own bytes.

## Tools (all host)
- `q2b_build MODEL.gguf KERNELS.ptx OUTDIR`: lowers each (kernel, baked params, grid) once (39 instances, 532 blocks,
  2929 instructions), writes rom.bin, m.bin (tables, constants, RoPE 65536 x 64, the queue, the args blocks), w.q8 (the
  GGUF's Q8_0 blocks in Q2B_MAT order, 58,789,888 blocks), meta.txt, listing.txt, queue.txt (descriptor names + the
  probe's check lines), program.txt (every descriptor with its grid and parameters: c:V p:WORD a:K q:K | args).
- `q2b_gpu_probe IMAGE MODEL "PROMPT" [--positions P] [--show N] [--watch W:N] [--trace-store W:N] [--trace-desc D]`:
  runs descriptor by descriptor (sm_run returns 3 at stop_qi), compares each check's region with q2b_fix's tap of the
  same name and position; kinds by tap name: plain M words; `*_q` A8 lanes from an A block; `*_xs` A32 scales; `vt*_q`
  the transposed value cache (dim i at block + i * CTX/32, lane pos % 32); `kc*` and `attn_vs` move with the position;
  `gdn_state` the S bank vs fix.gdn_state(); `result_output` the argmax (skipped at prompt positions). On a DIFFER of
  linear_attn_out / result_output it recomputes the row from the image's bytes.
- `q2b_gpu_check IMAGE MODEL GEN "PROMPT" [--hashlog F]`: the program to HALT, each emitted token vs the reference;
  prints the prompt's token ids (the replayer takes them).
- `ptxrun K.ptx NAME GX GY BX BY params [-v] [-a w,..] [-base N]`; `tests/gpu/lower_all.sh [PTX]`.
- OpenCL: `scripts/cl_build.sh`, `libqgpucl.a` (+ `qgpucl_ext.h`), `clhello`, `q2b_cl` (docs/opencl.md).

## How to continue, stage by stage
- G8d OpenGL subset: a rasterization pipeline as OpenCL C kernels on the machine (vertex transform -> bin/raster ->
  fragment -> framebuffer in M as 32-bit words), a C API named like GL for the subset (context, buffers, draw
  triangles, clear, read pixels), the host reading the framebuffer through clEnqueueReadBuffer; first proof: one
  triangle rendered by the embedded GPU on the reference machine, checked against a host rasterization of the same
  triangle; then run it through the same path the 2B uses. Say plainly in the docs that it is a subset, not a
  conformant GL.
- Accuracy: bisect the 87.50 % vs 93.40 % gap in q2b_fix.cc with compile-time switches per change (attention g blocks,
  scores >> 4, pr reciprocal, RoPE double rounding, div48, embedding shift), 6 gates in parallel (one process is 4.0 GB
  RSS; mind oomd above 90 % RAM + swap).
- G4 circuit: port kernel.py's step onto src/circuit/graph.h with the contract above (per-warp uniform file 8 x 16
  words, q8dot both forms, warps 0 = 256, entry restart, segment order); decouple src/circuit/eval.cc from the RV64
  headers; gate: the circuit's records bit-exact with sm_ref on the 2B's program for 3 positions.
- G5: emit the GGUF (35B tensors untouched + gpu.* tensors + ROM/M/A/S/W banks as tensors); tforward on the host from
  the file alone; hash = sm_ref's run3 hash for the same prompt. NVMe is at 92 % (81 GB free): the 35B merged file
  needs ~42 GB; out/gpu (1.9 GB) is redundant with out/gpucl and may be removed.
- G6 card: port src/card/chip_vpu.c to this machine's record step; 60 cores x 2 = 120 threads, card memory at most
  12 GB; the live window on the desktop; trace tok/s on the 7120A is the metric (QwenPU VX2 did 178-409 trace tok/s
  with 216 M trace tokens per inner token; this machine needs 41.3 M per position, each token heavier). The card was
  released by Blackstar; ask before taking it.
- Runtime items: an ICD (`clIcdGetPlatformIDsKHR` + dispatch table; opencl-headers 2026.05.29 and ocl-icd 2.3.5 are
  installed) so `clinfo` and any OpenCL program see the device; 2D/3D NDRanges; launches above 8192 work-items split
  with group-id offsets; a real allocator (the bump allocator never frees); ptx.cc's die() -> exceptions so a bad
  program returns CL_BUILD_PROGRAM_FAILURE instead of exiting the host process; clEnqueueCopyBuffer/FillBuffer;
  8-byte scalars (the machine is 32-bit: reject with a clear log).

## Conventions and traps that bit here
- Tool shell: `$var` unquoted is ONE argument (zsh, no word split): use `eval` or `${=var}`; `make` is an alias for
  -j16: always `/usr/bin/make` (14 jobs max on the host); `grep` is ugrep: `/usr/bin/grep`.
- A `sed ... && grep ... && sed ...` chain stops at the first grep that matches nothing: verify each edit separately.
- My own probe rule: a tap name ending in `_q` means an A8 compare: never name an M-word tap `*_q` (attn_q was renamed
  attn_query).
- clang 23's OpenCL front-end emits every kernel as a stub calling a noinline `__clang_ocl_kern_imp_NAME`; the parser
  folds it. `__nvvm_bar_sync(0)` is the barrier builtin that compiles; `-cl-no-stdinc` + the device header.
- The reference machine's uniform registers persist across launches per slot; a loop counter left in u6 runs once per
  later launch (smtest's lesson).

## Chronology 11-04 21:00 to 11-05 04:44 (the stretch that built the circuit; every number from the named log)
1. Contract v2: smtest's token expectation corrected (1329 -> 1518 -> 1674 tokens, 0 failures); the gdnunit probe
   wrote one S element twice in a block (a fault under the new rules): a barrier between its two updates; the CUDA copy
   of the kernels frozen, `src/guest/q2b_kernels.cl` is the source.
2. Chunk families split into VLO / VHI / VEND (VEND = a lane's last high chunk) so the circuit's counters depend on token
   kinds only; applied to sm_ref, lower, the gate language, the tokens table in docs/machine.md (CHIP_VOCAB 209,157).
3. Design findings before the circuit: six chained multipliers would be about 100 layers deep, so the stage layout was
   chosen by measuring token cost (`logs/g4/classes.txt`: all-deep 1,242,184,604 tokens; deep at 1,3,5: 1,679,437,946;
   at 1,4: 2,296,120,462; at 2: 3,980,235,950); udiv / usqrt as 2 / 1 restoring steps per lane pass with the state in
   scratch words. Decision: three deep stages, classes {49, 63, 49, 63, 49, 127} (1.35x the tokens of all-deep for about
   60 % of its depth and half its gates). The register gadget became token-written (`Circ::reg`).
4. The exact evaluator with banks (`eval.cc`: lagged ports, memories loaded from the 2B image, MEM_SCR added) and the
   circuit (`sm_circ.cc`): datapath, high-side writes, branch predicate, exit. First build 205,877 units, 141 layers.
5. Gate on the real job: 20,000 / 20,000 tokens at 858 passes/s; smtest's hand-made blocks given their own all-units
   classes. 2 M run: first disagreement at token 81,975 (machine EXEC(4124), circuit NEXT(0)); token history added to the
   report; the failing pass was a queue read after a launch (descriptor 7 read as kind 1, target 0: a stored 290969 at
   word 14); memory comparison added; root cause: the A/S/C store banks were built from `std::vector<Bank>(MEM_COUNT)`
   whose default memory is M (`bks[k].mem = k` fixed it); sm_ref gained `stop_req` so the sink stops the reference.
6. 300,000 / 300,000 tokens with the 13-thread evaluator (1,485 passes/s, 202 s); depth and residual-slot stats added to
   the tool (`logs/g4/circ-stats1.txt`: 282 sublayers, 8,592 slots for 30,020 dims).
7. 10 M run (`logs/g4/gate2.txt`): 704,435 / 704,435, then token 704,436 (machine VEND(0), circuit VEND(1); u8, u10, u11
   of slot 4 differed): a lane's recomputation passes saw side effects of its computing pass (the `sup` S write and the
   reduction's running word landed too early). Fix 04:41: pending state writes (SCR_SUP0, 6 x 3 words, fire at the
   lane's end) and before/after word pairs per reduction (SCR_RED0A/B, RED1A/B); the evaluator's scratch is 128 words;
   206,651 units, 23,409 nodes, 31,050 dims, 141 layers. Rerun `--max 3000000 > logs/g4/gate3.txt` (04:41).
