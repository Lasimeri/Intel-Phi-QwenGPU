# The circuit: the SIMT machine as layers of the 35B (G4)

`src/circuit/sm_circ.cc` builds the machine of `docs/machine.md` as a graph of the 35B's layer form: SwiGLU units
(`c * a * relu(b)`, every weight dyadic, every activation an exact bf16 value), Gated DeltaNet heads (one-hot keyed
rows), and banks (the memories, written compactly). `src/circuit/eval.cc` is its exact evaluator; `src/tools/circ_gate`
is its gate: the reference machine (`src/gpu/sm_ref.c`) runs the real 2B job and feeds every token of its stream to the
evaluator, which must have predicted it. The gate language (`gate.{h,cc}`: nibble adders, products as nibble columns,
bit selects, counters, tables, registers) is the port of QwenPU's RV64 chip code, proven exact there. Nothing of this
is in the model file yet: the emitter (G5) comes next.

## One pass

A forward pass over one token is one step of the machine: it computes ONE lane's six stages of the current block
(lane-serial records, machine.md) and predicts the next token. Levels within the pass, bottom to top:

1. the token's features: its family (VLO VHI VEND EXEC NEXT WARP EMIT HALT GO LSTEP TEXT) and payload bits;
2. token-written registers (theads written only when their token arrives): b from EXEC, R at an EXEC with phase, q from
   NEXT, w from WARP (cleared at NEXT), the emit mode, the descriptor fields (kind, entry, warps, lwpb, address) from
   the two queue words read at the NEXT pass; u14 (the args word) written into every slot's uniform file at NEXT;
3. the ROM block at b (a 544-bit bank read) and its decode: per stage the fields, a one-hot op (only ops of the classes
   the stage holds), active / writes / uniform / stv / sup / reduction flags; the LIVE word: which live-outs are lane
   items, uniform items, reductions; the item -> stage tables;
4. the scratch words read low (`MEM_SCR`, sm_circ.h): LN (the lane this pass computes; 32 = the final pass), LI (the
   item within the part), ANY and P0 (the branch predicate over the lanes), the reductions' running values (a
   before/after pair each), the udiv/usqrt state, the scfg constants, the pending stores (8 slots) and pending state
   writes (6 stages). LN steps on VEND|LSTEP, resets at EXEC; LI steps on VHI, resets at EXEC|VEND|LSTEP: token kinds
   only, never the counter's own value;
5. the pending stores of the lane that just ended land (below the datapath), the pending state (`sup`) writes too; the
   uniform items' stores land at the uniform VEND or at the next record's first token;
6. the register files read low: 24 lane ports (a, b, c, and rd for sup, per stage) at (slot, LN, field), 18 uniform
   ports at (slot, field);
7. the datapath: per stage, operands with forwarding (the latest earlier stage writing the register, kinds agreeing),
   the guard (the latest earlier setp on the predicate), the address A + imm, then the units the stage's class holds:
   ANY (add sub max min setp selp tid sreg ld stv: one A + B and one A - B adder, loads from every bank by one address),
   LOGIC (and or xor, two-level barrel shifts, clz by a thermometer), MUL (one 32x32 multiplier with a signed
   correction: mad, mulhi, mulsh with a rounding 64-bit barrel; sup: old * e, the delta product, two rounding shifts,
   clamp, |new S|, max with rd), DOT (q8dot: the X block from W or A, the A block, the f16 scale through the F16
   table bank, bit planes against nibbles, two rounded products, + rc), STATE (sdot), RED (wsum / wmax of the lane
   against the running value), SCALAR (udiv: two restoring steps a pass over 32 lane passes; usqrt: one step a pass);
8. the branch predicate of the lane, the item values;
9. the scratch words written high (the new LN, LI, ANY, P0 at EXEC, the reductions' after/before pair at a new-lane pass,
   the scalar state, scfg at the final pass, the pending stores and state writes refilled at the computing pass);
10. the register files written from the payloads: at a lane item's high chunk, register rd of the item's stage, row
    (slot, LN before the step), data = the VLO payload one pass back and this pass's VHI/VEND payload; uniform items
    likewise into the uniform file; a reduction's result from its after-word at the last lane's end;
11. the exit: the next token as a case list (`SelCase`) over the token kind and the state: EXEC -> lane 0's first item
    or LSTEP; VLO -> VHI or VEND of the same item; VHI -> the next item's VLO; VEND|LSTEP -> the next lane, or the
    uniform items, or the record's end (NEXT -> EXEC(b + 1), JUMP / COND -> EXEC(target), END -> WARP(w + 1) or
    NEXT(q + 1) when the launch is over, BAR -> WARP); WARP -> EXEC(entry or R, phase); NEXT -> EXEC(entry, 1) /
    NEXT(target or k + 1) / EMIT / HALT; EMIT -> VLO VEND of M[addr] -> NEXT(q + 1). The code dims carry the choice
    at 2^-20 as in the RV64 chip.

## Why the state is shaped like this

A head writes before it reads within a pass, and a lagged channel can only sample a form computed below its head. So
nothing in the layers can update itself from its own previous value in one pass. Three mechanisms carry state:
- the token stream: b, q, w, the descriptor, and every value a lane commits (its own VAL tokens); the reductions'
  running partials are in the lane parts too (the contract);
- scratch words read low and written high: the same memory, two bank nodes; the write lands at the end of the pass and
  is read at the start of the next;
- a lane's recomputation passes (its VLO/VHI tokens after the computing pass) must see the machine as the computing pass
  saw it: stores and state writes wait in pending words until the lane ends; a reduction's before-value is kept beside
  its after-value. The two defects found by the gate so far were exactly this (store banks defaulting to M: token
  81,975; S writes and the reduction word landing before the lane ended: token 704,436).

## Numbers (host, 11-05 05:5x)

| | |
|---|---|
| units / nodes / banks / dims | 206,702 / 23,449 / 67 / 31,601 (`circ_gate --stats`, after the deferred register commits) |
| layers | 141 (282 sublayers) |
| residual slots the dims need with lifetime sharing | 9,020: hidden 8192 does not fit; the emitter uses 16384 with 16 ballast dims (c = 2^-35) |
| evaluator | 1,000 to 1,800 passes/s with 13 worker threads (the per-level sync of 141 levels is the ceiling); the file's forward 133 to 142 passes/s single-threaded |
| circuit gate | 2,999,999 of 2,999,999 tokens of the real job to the 3 M limit (`logs/g4/gate7.txt`, 1,373 passes/s); every earlier stop (81,975; 704,436; 689,072; 828,728; 853,915) was a defect of the recomputation class (memory `qwengpu-recomputation-rule`), fixed; a one-position run (66 M) is `gate8-position.txt` |
| the model file | `out/model/q35-with-gpu.gguf` (the real 35B byte-identical + the GPU, 41 GB real): 2,999 of 2,999 steps and 94,791,000 dim values equal the evaluator (`logs/g5/merge-verify.txt`); the chip-only files: 199,999 of 199,999 steps (`egate5.txt`), `gpu_run` 48 of 48 hash points equal the reference over 200,000 steps, 89 to 103 passes/s |
| stage classes | {49, 63, 49, 63, 49, 127}: deep units in stages 1, 3, 5, scalars in 5 (`logs/g4/classes.txt`) |

## Open

- Depth: 141 layers come from six stages of forwarding + units in sequence; the shallow stages still carry two adders,
  a shifter and the operand muxes each. Flattening (one adder shared by add/sub/compare, forwarding folded into the
  unit's first level, fewer levels in sum_columns) is the next optimization; it also lifts the evaluator's ceiling.
- The emitter, the file's forward, its gate and the runner exist (`src/circuit/emit.cc`, `src/exec/tforward.cc`,
  `src/tools/emit_gate.cc`, `src/tools/gpu_run.cc`; `make model`). The merge into the real 35B is `gpu_emit OUT --base
  MODEL_35B`; run 06:48: `out/model/q35-with-gpu.gguf`, see the hand-off.
- A full position of the job is about 65 M passes: 5 days through the file at 140 passes/s, 10 h on the evaluator; the
  circuit gate covers the first 0.85 M, the file gate the first 0.2 M (running).
- Stale uniform pending stores fire again at the WARP, NEXT and EXEC passes after the record (idempotent: same word,
  same value, nothing else writes M in between); clearing them at the fire would be cleaner.
