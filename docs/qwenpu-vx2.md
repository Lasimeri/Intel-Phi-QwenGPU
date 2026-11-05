# The machine's contract: RV64 + VX2

`src/rv64/m64.{h,c}` is the contract in C, a host test reference. `src/circuit/core64.cc` and `src/circuit/vx64.cc` are the same machine in gates, written out as tensors of the 35B (`docs/embedding.md`). Every instruction is a **record of tokens**, each token one forward pass; state changes at the record's last token. Both must give exactly the same tokens.

Status of the gates: RV64 core and VPUT, VGET, VMAT8 passed teacher-forced with exactness asserts (`logs/t9/circ64-1632.txt`, `vx2-seed*.txt`, `vx2-controls.txt`), before the memories became banks. **The other 12 instructions have never run in gates.**

## ISA

RV64I + Zba + MUL + MULW + VX2. **No DIV, REM or MULH\*:** they would be large and the guest needs none; `scripts/rv64_build.sh` fails if a guest contains one (clang once wrote a `mulhu` for a constant division). ROM at `0x1000000`, 8,192 pre-decoded words (`m64_rom_word`: byte 0 is the micro-op + a spare bit, bytes 1-3 spare bits + a 5-bit field). RAM from 0, 32 KB. `ecall` a7 = 1: a byte out, 9 bits; a7 = 93: HALT; a7 = 2: a debug note, the reference's only, never in the chip.

## Token families

| Family | Payload | Meaning |
|---|---|---|
| `V` | 16 | a chunk of a value or a stored word, not final |
| `F0` `F1` `F2` `F3` | 16 | final after 0, 1, 2, 3 `V`: value record of 16-bit chunks, sign-extending, shortest form. A head's write reaches back 3 tokens, so at most 4 tokens |
| `J` | 16 | final, jump: anchor = payload, n = 0 |
| `R` | 16 | final, anchor refresh after 63 instructions, nothing executes |
| `BYTE` | 9 | final, a byte out |
| `FLT`, `HEX` | 3, 4 | a fault: cause, 8 hex digits of the pc, HALT |
| `VC` | 15 | an NPU op's configuration: slot (3 bits) << 12, 12 bits |
| `VE`, `VH` | 16 | stream tokens; `VH` completes an element |
| `SS0`, `SS1` | 16 | final, a store: after V V V, bits 48..63 of the merged word; which one is the RAM value head |
| `NT` `HALT` `GO` | 0 | final, nothing to report; halt; start |

Stores of any width are V V V SS over the merged 64-bit word.

## VX2 (opcode 0x5B)

`A = x[rs1]`, `B = x[rs2]`, `C = x[rd]` read as an operand. Streams hold at most 64 `VH`: a thermometer counts them, the guest loops.

| funct3, funct7 | Op | Operands | Effect | Record |
|---|---|---|---|---|
| 1 | VPUT | C addr, A value | `VM[C] = A` | VC(0, 1), VE VH, NT |
| 2 | VGET | A addr | `x[rd] = VM[A]`, sign-extended | VC(0, 1), value |
| 3 | VMAT8 | C dst, A block, B = va, (n-1) << 8, carry << 14, store << 15 | a matrix row over n <= 64 blocks, next block A + 16; `carry` continues the last sum; `store`: `VM[C] = acc` | VC(0..5), VE VH per block, NT |
| 4, 0 | VADD | C = dst, (n-1) << 24; A, B addrs | `dst[e] = a[e] + b[e]`, n <= 32 | VC(0..7), VE VH (a), VE VH (result) per element, NT |
| 4, 1 | VMUL | also s << 30, s = 16 or 24 | `dst[e] = (a[e] * b[e] + round) >> s` | the same |
| 4, 2 | VMULS | B a scalar, n <= 64 | `dst[e] = (a[e] * B + round) >> s` | VC(0..7), VE VH per element, NT |
| 5 | VQ | C block, A addr, B r | lane i of VA block C = `(VM[A + i] * r + 2^23) >> 24`, low 8 bits | VC(0, 2..6), VH x 32, NT |
| 6 | VXS | C block, A value | `xs[C] = A` | VC(0), VE VH, NT |
| 7, 0 | VSDOT | C acc, A row, B = k8, first << 8, half << 9 | for j in the half row: `acc[j] = (first ? 0 : acc[j]) + S[row][j] * k8` | VC(0..3), VE VH per j, NT |
| 7, 1 | VSUP | C = delta, half << 22; A row; B kreal, 28 bits | for j in the half row: `S[row][j] = clamp(round((S * e >> sh1) + (kreal * delta[j] >> 16)) >> sh2)`, +-32767, 48-bit then 52-bit datapath | VC(0..6), per j **VE if its magnitude is a new largest, else VH**, payload the new S; NT |
| 7, 2 | VSCFG | A = e; B = (sh1 + 32) or (sh2 + 32) << 6 | sets them; the largest restarts | VC(0..2), NT |
| 7, 3 | VSMAX | | `x[rd] =` the largest magnitude since VSCFG | value |
| 7, 4 | VWB | A block, B lane, 32 = the scale | `x[rd] =` the lane, or the scale | VC(2..4), value |
| 7, 5 | VCPUT | C addr, A byte | a byte of the KV cache | VC(0..2), VH, NT |
| 7, 6 | VCGET | A addr | `x[rd] =` the byte, sign-extended | VC(0..2), F0 |

VC slots are positional: slot 0, 1 = C or the VM address (bits 22, 23 of a state op's slots are its half and first); 2, 3 = A; 4, 5, 6 = B; 7 = rd's top (n - 1, s). VMAT8 and VWB put the block's top bits and the va or lane in slot 4. Exact fields: `vc_case` in `src/circuit/vx64.cc`, `vc()` in `src/rv64/m64.c`.

**VMAT8's arithmetic, a block:** as `Fix::gemv` of `src/text/q2b_fix.cc`: `inner = sum w[i] * q[i]` over 32 lanes, `t = (inner * mw + 2^15) >> 16`, `u = (t * xs + 2^15) >> 16`, `acc += u`, modulo 2^32. The largest values measured (`logs/t7/q2b-fix-gate.txt`): matrix output 6,508,460 Q.16, block product 35,658, so 32 bits do not wrap.

In the gates: two nibble bit planes for the dot (`w = 16 (hi - 8) + lo`, a lane stored with its sign bit flipped), nibble products reduced by `sum_columns`, one VM read per token, the running sum through the `vx.acc` head. `VSUP`'s family is the "new largest" flag since a head cannot compare and write in one token: upstream's flag trick.

## Memory formats

- **Tiles** (`src/text/q2b_pack.h`): 16 rows x 32 columns, 576 B: 512 B int8 by column (byte `16 i + r` is column i of row r), then 16 int32 Q.24 scales. **Block address = tile index * 16 + row in the tile.** A matrix's rows come in groups of 16, a group's tiles consecutive.
- **VM:** int32 words, Q.16 activations. The map is `src/guest64/q2b64_map.h`: job words, function tables, the matrices' places, activations, the tokens, RoPE, then per layer its constants, windows and caches' scales. **The state instructions carry their VM address in 22 bits** (`VSUP`'s rd bit 22 is its half; `VSDOT`'s slot 1 holds its half and first above the address), so `VM_A`, `VM_DELTA`, `VM_O` must stay below 2^22: a static assert in the map guards it. RoPE at 65,536 positions is 4.19 M words and sits above them for this reason; below them it broke every logit, and the token hash could not see that (`docs/embedding.md`, section 7).
- **VA:** 256 blocks of 32 int8, literal heads, 4-hot read x 4: group `b >> 5`, 2 key heads x 2 value heads x 8 lanes, rows of 2 lanes. **xs:** one word per block.
- **VS:** row = `(gdn layer * 16 + head) * 128 + i`, 128 int16. Written 4 words at a time from the last 4 tokens' payloads.
- **KV cache:** `q64_kc(il, slot)`, `q64_vc(il, slot)`: bytes, slot = position * 2 + head.

## The guest

`src/guest64/q2b64.c`: the whole 2B in RV64 + VX2, every number the integer pass's. Division and square root are guest code. Matrices 57.02 % of its tokens, plain RISC-V 24.82 %, VSDOT 8.21 %, VGET 4.10 %, VSUP 2.31 % (`logs/t9/q2b64-gate.txt`, before the last changes).

The integer pass itself, `src/text/q2b_fix.cc`, formats: activations Q.16 in int32; matrix input int8 with one Q.24 scale per 32 columns; GDN q, k, v int8 per vector; state int16 per head with a power-of-two scale, chosen from a bound (largest |s| x decay + 127 x kscale x largest |delta|), so no pass over the elements; function tables at 2^-8, exp at 2^-11. Against the exact 2B: 94.17 % top-1, 100 % top-5 over 240 positions (`logs/t9/q2b-fix-ab-new.txt`).
