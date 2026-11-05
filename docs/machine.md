# The machine: a SIMT streaming multiprocessor in the 35B's weights

This is the contract. `src/gpu/sm.h` holds the encodings, `src/gpu/sm_ref.c` the reference in C (host, a test
reference: never the result), and the circuit (`src/circuit/sm*.cc`) is the same machine in gates, written as tensors
of the 35B (`docs/embedding.md`). A forward pass of those layers on one chip token is one step of this machine. The
token stream is the machine's execution log, as in qwenGPU (`input/upstream/qwenGPU/qgpu/circuit/kernel.py`), widened
for the 2B.

## Execution model

Single instruction, multiple threads (Flynn's array processor): one control unit, 32 processing elements (lanes) with
their own registers, no program counter of their own. A **warp** is the 32 lanes. A **block** is up to S warp
instructions; a block runs **lane by lane**: one forward pass computes one lane's S stages (the lane index is machine
state, a counter), so the record of a block in the token stream is

    EXEC(b, phase)
    lane 0's part   lane 1's part   ...   lane 31's part
    VAL x 2 for each uniform live-out that is not a reduction, in LIVE order
    [WARP(w') when the warp changes: END, BAR]

where a lane's part is VAL x 2 (low chunk, high chunk) for each lane live-out in LIVE order, then VAL x 2 for each
reduction (`wsum`, `wmax`) in stage order holding its running partial after this lane (lane 31's is the result), or
one LSTEP token when the block has neither; and then the next EXEC. Only values that a later block reads cost tokens
(the live-outs); everything inside a block is free: a block reads committed registers or an earlier stage's result of
the same block (forwarding). Warps run one after another; the warp index w is machine state, told by a WARP token
when it changes (a launch starts at w = 0). A launch runs `warps` warps of a kernel from its entry block; a thread
block of `wpb` warps takes turns at a barrier, phase by phase (upstream's `bar`). Divergent control flow is
predicated: the driver (`src/gpu/lower.cc`) gives every PTX block a per-lane mask register when a branch is not
uniform, and loops branch on "any lane still on".

Why the partials and the WARP token are in the stream: in the layers a head writes before it reads within a pass, so
a value that changes by computation from one pass to the next (a running sum, the next warp) must travel through the
tokens or be a decay counter; the circuit reads the previous lane's partial from the last tokens (at most 4 back,
hence at most two reductions a block) and the next warp from WARP's payload.

Lane order (the circuit's cost is why: a lane-parallel block would be 32 copies of every datapath, about 32x the
RV64 chip, more than this disk holds; lane-serial costs 1.106x the tokens on the 2B's job, measured): lane L's stages
run, its part of the record is written, then its commits happen (its register live-outs, its stores, its `sup`
writes), then lane L + 1 starts. A reduction (`wsum`, `wmax`) and a scalar (`udiv`, `usqrt`: one combinational unit
each, evaluated in the final pass) are final only after the last lane: such a value is **late**, and in its block
only a uniform instruction may read it (uniform instructions are evaluated in the final pass, after lane 31, in
stage order, with the lane 0 guard for a uniform `stv`); a scalar's own operands must not be late; at most two
reductions a block; no `sdot` or `ld` of the S bank may follow a `sup` in a block (a sup's write lands after its
lane's pass). The driver's placer (`src/gpu/blocks.cc`) enforces all of it; the reference faults on a violation, and
on a lane loading what an earlier lane of the same block stored (lane-serial semantics no kernel may lean on). Late
uniform live-outs commit after the uniform VALs; a reduction's result commits from lane 31's partial.

Order (the stores through `stv` depend on it): the warps of a thread block run one after another through the blocks
of a barrier segment (from the entry, or the block after a `bar`, to the next `bar` or END), each warp the whole
segment before the next warp starts; after the last warp the segment advances; thread blocks run one after another,
each from the entry block. A warp's live-out store is in memory before the next warp's loads of the same segment.

Beyond upstream, the machine has:

- **Uniform registers** u0..u15: warp-wide scalars (Turing's uniform datapath). A uniform live-out costs 2 tokens
  instead of 64. They are global machine state shared by every warp of a launch (so a reduction over warps is a
  uniform add across warps). u15 reads 0. Lane register r15 reads 0.
  Each warp slot (w mod 8, the warp's place in its thread block) has its own uniform file, as CUDA's warp-uniform values
  are per warp: what a warp computes in one barrier segment is still its own in the next, whatever the other warps of
  the thread block computed in between; a slot's file persists across thread blocks and launches.
- **Memories** as banks, word-addressed by lane or uniform registers, loads free: M (int32 words: activations,
  constants, tables, the tokens, the launch queue), A (quantized blocks: 32 int8 and an int32 Q.24 scale), S (int16:
  the GDN state), W (the 2B's weights, the GGUF's own Q8_0 blocks: f16 scale + 32 int8, read-only), C (int8: the
  attention value cache). A store that a later block reads is a **live-out to memory** (`stv`): its VAL tokens
  carry the value, the bank writes it at the address the block computes. Memory never faults: an address is taken
  modulo the bank's size.
- **Warp instructions** for the 2B: `q8dot` (one Q8 block per lane: 32 int8 x int8 with both scales, exactly
  `q2b_fix`'s gemv), `sdot` and `sup` (the GDN state read and update per element, exactly `q2b_fix`'s), `wsum` and
  `wmax` (a reduction over the lanes into a uniform register), `mulsh` (a signed 64-bit product shifted with
  rounding, the elementwise form the 2B uses everywhere), `udiv` and `usqrt` (uniform only: a 64-bit numerator over a
  32-bit divisor, a 64-bit square root).
- A **launch queue**: descriptors in M run kernels one after another with no host between them; a descriptor can
  branch on a word of M, emit a word as a token record, or halt until the host gives a new turn. The host only
  writes the prompt's tokens into M and resumes the queue: it computes nothing.

## Instructions

Fields: op, rd ra rb rc (4 bits each), ua uc (ra, rc are uniform), ub (0 lane register, 1 uniform register, 2
immediate), ud (the result is uniform: every operand uniform or immediate; lane 0's value), pd cmp signed (setp),
gvalid g gneg (a guard predicate, for selp and stv), bank (ld and stv), sh (6 bits: a shift amount), a 32-bit
immediate: 48 bits of fields and the immediate, 80 bits a ROM row. Predicates p0..p7 are
block-local. Arithmetic is 32-bit two's complement, wrapping.

| op | effect |
|---|---|
| `add sub max min and or xor` rd, ra, rb/imm | the usual; max and min signed or unsigned |
| `shl shr` rd, ra, sh [, rc] | shift by the sh field (0..62; 32 or more gives 0, or the sign for shr), or by rc (unsigned, sh = 63); shr arithmetic when signed, else logical |
| `mad` rd, ra, rb/imm, rc | rd = low 32 of ra * rb + rc |
| `mulhi` rd, ra, rb/imm | the high 32 of the unsigned product |
| `mulsh` rd, ra, rb/imm, sh [, rc] | rd = (ra * rb + 2^(sh-1)) >> sh, signed 64-bit product, sh the field (0..62; 0 is the low word); sh = 63 takes the shift from rc as a signed number (negative: a left shift) |
| `setp` pd, ra, rb/imm | pd = ra cmp rb (eq ne lt le gt ge, signed or unsigned) |
| `selp` rd, ra, rb/imm, @p | rd = p ? ra : rb (gneg flips p) |
| `tid` rd | rd = 32 w + lane |
| `sreg` rd, imm | rd = (t >> imm[0:5]) & ((1 << imm[8:13]) - 1), t = 32 w + lane: a field of the thread index |
| `ld` rd, [ra + imm], bank | M: the word; A8: lane byte (block * 64 + j, j < 32) sign-extended; A32: the block's scale (j = 32); S: the int16, sign-extended; C: the byte, sign-extended; W8: an int8 of a weight block (block * 64 + j); W32: its f16 scale as Q.24 |
| `stv` [ra + imm], rb, bank | a live-out to memory: the bank writes rb (its low byte or half for A8, C, S) when the record's VALs for it arrive; inactive lanes (guard off) write nothing |
| `q8dot` rd, [ra + imm], rb, rc, bank | rd = rc + u, where inner = sum of X[ra + imm].q[i] * A[rb].q[i] over 32 i, t = (inner * mw + 2^15) >> 16 with mw the X block's scale as Q.24 (an f16 for W), u = (t * xs + 2^15) >> 16 with xs = A[rb].scale. X = W (bank 0) or A (bank 1: the KV cache). ra lane (a block index); rb a uniform register, to which imm is added as well (both blocks advance together), or an immediate holding the whole A block index (then nothing is added to ra). Modulo 2^32 |
| `sdot` rd, [ra + imm], rb, rc | rd = rc + S[ra + imm] * rb (rb as a signed 8-bit value) |
| `scfg` ra | the state update's constants: ra packs e (bits 0..15, Q.15), sh1 + 32 (bits 16..22), sh2 + 32 (bits 23..29); uniform, so it runs in the final pass: a `sup` in the same block still uses the constants from before the block |
| `sup` rd, [ra + imm], rb, rc | S[a] = clamp(shr_round(shr_round(S[a] * e, sh1) + shr_round(rc * rb, 16), sh2), -32767, 32767), rb the lane's delta (Q.24), rc = kreal (Q.24, the row's key times its scale); rd = max(rd, abs(the new S[a])): rd is read and written, so a run of sups carries the largest magnitude a lane wrote, and a `wmax` of it replaces a tracker; a lane whose guard is off writes nothing and keeps rd |
| `clz` rd, ra | the number of leading zero bits of ra (32 for 0) |
| `wsum wmax` rd, ra | rd (uniform) = the sum (mod 2^32) or the signed maximum over the 32 lanes of ra; inactive lanes (guard off) contribute 0 or are skipped |
| `udiv` rd, ra, rb, rc | uniform: rd = (ra:rc as 64 bits, rc high) / rb, unsigned; rb = 0 gives 0; the quotient's low 32 bits |
| `usqrt` rd, ra, rc | uniform: rd = isqrt(ra:rc as 64 bits), the guest's restoring root |
| `nop` | padding, so that an instruction lands in a stage that has its unit |

`shr_round(v, s)` is `q2b_fix`'s: `(v + 2^(s-1)) >> s` for s > 0, `v << -s` for s <= 0.

Stage classes: mul (mad mulhi mulsh sup), dot (q8dot), state (sdot), logic (and or xor shl shr clz), red (wsum wmax),
scalar (udiv usqrt), any (the rest). Which stages hold which units is the chip's choice (`SM_STAGE_CLASSES` in
`sm.h`); the driver pads with nops to reach a stage that has the unit, as upstream's placer does.

## Blocks and live-outs

A block's LIVE entry: the number of live-outs (0..4), the stage of each in emission order, the branch: `next`,
`jump t`, `cond s neg t [any]` (the predicate written by stage s's setp: lane 0's, or any lane's), `end` (the warp
is done: the next warp starts at block 0 of the launch, or the launch ends), `bar` (the next warp of the thread block
resumes at the phase's first block; after the thread block's last warp every warp continues at b + 1), and log2 wpb.
A live-out of a register-writing stage commits rd (lane or uniform); a live-out of an `stv` stage writes memory.
A live-out must be emitted before any live-out that overwrites a block input it reads (`blocks.plan`).

## Tokens

| token | id | meaning |
|---|---|---|
| VLO(v) | v | the low 16-bit chunk of a value |
| VHI(v) | 65536 + v | its high chunk |
| VEND(v) | 131072 + v | its high chunk when the value is the last item of a lane's part, of the uniform part, or of an emit |
| EXEC(b, phase) | 196608 + b + phase << 12 | a block runs for the current warp; phase: this record starts a phase (R = b) |
| NEXT(k) | 204800 + k | launch descriptor k begins (k < 4096) |
| EMIT | 208896 | then VLO VEND: a word the queue emits (the 2B's next token) |
| HALT | 208897 | the queue waits for the host |
| GO | 208898 | the host resumes the queue (after writing M) |
| LSTEP | 208899 | a lane's part of a record when its block has neither lane live-outs nor reductions |
| TEXT | 208900 | what every Qwen text token looks like to the circuit |
| WARP(w) | 208901 + w | the next record runs for warp w (after END or BAR; a launch starts at warp 0 untold) |

CHIP_VOCAB = 209157. Three chunk families so that the circuit's two counters (the lane, the item within a part) step
on token kinds alone: VEND and LSTEP end a lane's part, VHI ends an item. The program (ROM), the queue and every memory
start as the banks' contents in the model file: no host records. Measured counts of the 2B's job under each contract
are in docs/handoff.md.

## The launch queue

Two words per descriptor in M at `SM_QUEUE`: word 0 = kind (bits 0..1: 0 launch, 1 branch if M[word 1] != 0 to
descriptor word0 >> 16, 2 emit M[word 1], 3 halt) | entry block << 2 | warps << 14 (8 bits: 256 warps are written as 0) | log2 wpb << 22; word 1 = the
address, for branch and emit. A launch's warps count from 0; its lane registers start as the previous launch left
them (the driver never reads a register before writing it). The uniform registers persist across launches (that is
how a scalar passes from one kernel to the next without memory). After `halt`, GO restarts at the descriptor after
it, with the host's writes in M.

For a launch, word 1 is the kernel's **args word**: the sequencer loads it into u14 (`SM_UARGS`, never allocated
by the driver) when the launch starts, and the kernel reads its runtime parameters as `ld u, [u14 + k]` from M. So
one copy of a kernel in the ROM serves every call: the queue gives each call its own args block.
