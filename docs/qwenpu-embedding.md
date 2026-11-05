# The embedded engine: a RISC-V machine and an NPU inside the 35B's model file

The goal: a RISC-V machine and an NPU, interpreted as C, running inside the Qwen-35B model file as an internal inference engine for the Qwen-2B. This file says how, what is proven and what is not. State as of 2026-11-02; the handoff (`docs/HANDOFF-2026-11-02b.md`) has the latest.

```
token in ─▶ embedding row ─▶ computer layers (RV64 + NPU as weights) ─▶ output head ─▶ next token ─▶ fed back
```

One token through the layers is one step of the machine, a **trace token**. The 2B's program on that machine emits 3 bytes per 2B position: 216-218 M trace tokens at the start of a context, 712.7 M at position 2,552 (measured on the reference).

## 1. Rules of the form

- **One model file.** `out/merged/q35-with-2b.gguf`: the 35B's metadata, tensor table and tensor bytes copied raw and unchanged, then `qwen35moe.chip.*` keys and `chip.*` tensors.
- **Nothing linked.** The runner, `src/tools/chip_run.cc`, links `src/exec/tforward.cc`, the GGUF reader and the tokenizer. `nm` finds 0 symbols of `m64_`, `npu_`, `Evaluator`, `RvCore`, `build_core`, `q2b::Fix`. Native kernels swapped in for compiled blocks ("macro-fusion", proposed earlier) are out: that is linking under another name. Allowed speedups are generic and derived from the tensors: sparsity, change-driven evaluation.
- **Memory limits do not weaken the form.** Card at most 12 GB, host to `MemAvailable - 4 GB`, then stream from the 1TB NVMe. Streaming fixes capacity, not bandwidth.
- **The matrix operations run on the 7120A's VPU**, one machine over 60 cores x 2 = 120 threads: section 5b.

## 2. The residual and the layers

Port of upstream's `chipformer/q35/emit.py` and `merge.py` to the 35B, in `src/circuit/emit64.cc`.

Residual: `[Qwen's 2048 dims][ballast x 8][tau][IR dims 1 ...]`, hidden 8192. Chip tokens carry `B = 2^40` in every ballast dim. Upstream widens Qwen's matrices with zero columns; here Qwen's tensors stay as they are, 2048 wide, and that padding is a rule. A literal widened copy would be about 37.79 GB x 4, mostly zeros.

Each chip layer has the 35B's shape: 16 key heads, 32 value heads, state 128, conv 4; tensors `chip.blk.L.{attn_norm, attn_qkv, attn_gate, ssm_alpha, ssm_beta, ssm_conv1d, ssm_a, ssm_dt.bias, ssm_norm, ssm_state, ssm_out, post_attention_norm, ffn_gate_inp_shexp, ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp}`. Level `2L + 1` of the circuit is the layer's heads, level `2L + 2` its FFN.

**The FFN's realization.** All units live in the **shared expert**, widened to 8,192 units through its own metadata key (`expert_shared_feed_forward_length`, which is 512 for Qwen's layers). `ffn_gate_inp_shexp` reads 128 from the ballast, so its gate is `sigmoid(128) = 1`. A chip layer's 256 routed experts are absent, meaning zero: whatever the router picks adds exactly 0. A unit is `c * a * relu(b)`: `gate = K b`, `up = a / K`, `down = c`, `x = silu(K b) * (a / K)`.

**Heads.** `row(expr)` writes a linear form as a projection row: `coef / c` on the form's dims, `coef / (B c)` on the first ballast dim for the constant. Query and key channels are `K (2 e - 1)`, so `silu` gives `K` or exactly 0. Address decodes cost no FFN unit: `andpre` is a linear threshold.

**The exit.** A last layer clears every chip dim but the 21 code dims, and the ballast. A counter deep in a long stream can survive it: `j[2]` at 2^-99 times `c` is below what the norm holds in F32 (measured at token 65,536+). The output head is tied to `chip.token_embd`: logit = row . x; dims no row touches add exactly 0 and are skipped. `TForward::next_token` finds the row whose dot reaches the upper bound (the sum of the largest `|row[d]| |x[d]|`), which proves it the argmax without scanning the others, and falls back to a full scan otherwise.

## 3. Exactness conditions

| Condition | Why | Evidence |
|---|---|---|
| hidden a power of 2 | `q_rms_scale` multiplies by `1 / hidden` as a float; 1/5120 would be inexact | by construction |
| `m B^2 / hidden` a power of 4 | every norm on a chip token divides by a power of 2: `c = 2^-35`, `B c = 32` | `q_rsqrt(4^j)` exact for all 81 j in -20..60 |
| Weights exact in F32 | emission dies otherwise | `emit64` |
| **BF16 activations in chip layers** | a head's read is normalized through `1/sqrt(128)`, which leaves F32 noise: 0.49999997 for 0.5, largest error 0.048 on a `2^-20` exit code. Rounding to the top 16 bits at hn, conv input, the normed column, mix, xs, y removes it, as upstream's bf16 checkpoint does. Source of the noise not isolated | `logs/t9/embed-1.txt` against `embed-2.txt` |
| `q_sigmoid(x < -87) = 0` | every closed gate is `silu` of a large negative; `elem.h` returned NaN there, in a range Qwen's text never reaches. Fixed in `elem.h`. | scratchpad test, 05:1x |
| `q_sigmoid(128) = 1`, `q_silu(128 k) = 128 k` for k in 1..255 | open gates | same test |
| `exp(-softplus(a - 110))` exactly 1, 0.5, 0 for a = 0, 110, 238 | a head's decay selector | same test |

| Denormals kept on the VPU | `2^-99 x 2^-35 = 2^-134`, as the host computes it; a flush would change a value | checked at the start of a card run on every thread: `chip_vpu_setup` |

On the card `src/card/chip_vpu.c`'s `sigmoid0` adds `q_sigmoid`'s branch to `q35_elem.h`'s `v_sigmoid` as a blend.

## 4. The memories

Their contents are the state the layers start from: DeltaNet state in upstream's design, `snapshot()` there. The file holds them.

| Memory | Size | Form | Holds |
|---|---|---|---|
| `W`, the weights | 3,674,368 tiles, 2.12 GB | bank `chip.bank.K.mem` | the 2B's Q8 values and Q.24 scales, a block = one row of a tile |
| `M`, VM | 2^23 words, 33.5 MB | bank | the model's constants, activations, GDN windows, ks and vs, RoPE for 65,536 positions, job words, tokens |
| `S`, VS | 2^16 rows of 128 int16, 16.8 MB | bank | GDN state, 16 x 128 x 128 for each of 18 layers |
| `C`, the KV cache | 2^29 bytes | bank, sparse | full attention's k and v as int8: 65,536 positions x 2 x 256 x 2 x 6 layers |
| VA, xs | 256 blocks | literal heads | a matrix's int8 input, with one Q.24 scale per 32 columns |
| ROM, RAM, registers | 8,192 words, 32 KB, x1-x31 | literal heads | the guest |

**Banks are the same heads written compactly.** Real heads for the weights would be about 230,000 key heads (about 27 GB of builder memory, about 352 GB of file pages, arithmetic). A bank keeps the head's semantics: its address, write-enable and data forms are rows (`ra, wa, we, data, taps`), a token's write comes before its read, a read gives bits 0/1, or bytes / 256 for `W`. **Expansion to literal heads has never been checked at full size.**

## 5. The forward, `src/exec/tforward.cc`

It reads the file only. `load_sparse` walks a tensor's data extents with `SEEK_DATA` and keeps nonzeros, which changes no value. Then, per layer, the arithmetic of `src/text/q35_ref.cc` with `src/text/elem.h`'s functions: norm, `qkv`, `gate`, `alpha`, `beta`, the causal conv with `silu`, q and k normalized per key head, the state update, the read, its norm, `mix`, `ssm_out`, banks, residual, norm, the shared expert, residual. It knows nothing of RV64 or the NPU.

## 5b. The forward on the card, `src/card/chip_vpu.c`

The same arithmetic, operation for operation, from `src/exec/chip_image.{h,cc}`'s image of the file's tensors, 2.90 GB with the memories: 205,918 slices, only 6,636 distinct weight lines, 28,639,018 embedding nonzeros. Every float is in a zmm register: KNC's x87 keeps scalars in extended precision, so a float is only loaded, computed on in a vector, stored, or compared as bits. A row's columns stay ascending, so a line of 16 rows computes the same bits as 16 rows on their own.

What makes it faster holds for any tensors, never for a particular program:

- **A line is computed again only if a column it reads moved.** A form's line, a unit, a head's output are pure functions of what they read, and keep their last value. Fan tables go from a changed dim to its readers. Per token, measured on the first 15,000 (`logs/t11/stats.txt`, host probe): 12,005 lines of hn forms, 1,411 with a changed input.
- **`hn` one token ago is one array.** `hn` moves only by logged writes, so replaying them a token later keeps a second array exactly one token behind, across the token's boundary too. It replaced a snapshot per sublayer, 2 MB against a 512 KB L2.
- **A line that does not change is not written**, so other cores keep their copy.
- **The RMS scale is proved, not summed.** A rounded sum of squares is monotonic in every term, so if the sum with every small dim at a bound M equals the sum with them at 0, every x between has that same sum: the ballast swallows the rest whole. Each y then only has to stay within M. Checked against the real sum on every sublayer for the first 4,096 tokens.
- **A closed unit's up form is not read; a zero column adds exactly nothing** (`src/card/sell.S`).

**The team** is a 4-ary heap over thread ids, each line of work always on the same member so what it keeps stays in that core's cache. An empty phase measured 8.2-9.5 us at 120 threads, 3.6-4.1 us at 21, 1.4-2.0 us at 5 (`chip_main`'s start lines). Sublayers with few lines run on thread 0: 6.7-7.6 phases a token, measured.

**A checkpoint** holds the machine's state and nothing the engine only remembers: the conv and memory windows oldest first, the attention states, VM, VS and the KV cache, 593.8 MB. Everything a line keeps starts again from zeros on a resume, which is what it would be had its inputs been zero last token, so any engine can go on from any checkpoint: proved on the real run across an engine change.

**The output head** is integer only: a row with the largest value of the right sign on every dim x is nonzero on reaches the sum of the largest terms, and no row can pass that. Dims no row reads add exactly 0 and are skipped: a counter deep in a long stream can survive the clear layer (`j[2]` at 2^-99).

## 6. System prompt, context, turns

- `qwen35moe.chip.system_prompt` holds the text: `~/CLAUDE.md` frozen at build, 11,880 B, 2,552 tokens of the 2B's tokenizer.
- **The context window is 65,536 positions**: separate from the system prompt, which takes 2,552 of them.
- The file also holds **that text's KV cache**, a prompt cache: `src/tools/q2b64_host.h` runs the integer pass (`src/text/q2b_fix.cc`) over the system prompt once at build and writes state, windows and caches into the memories; `out/merged/context.bin` caches it. Loading the file restores it, and the first prompt continues at position 2,552. Reason: consuming 2,552 tokens through the layers on every start is at least 541 G trace tokens, about 34 years at 500 tok/s (arithmetic). Changing the text means rebuilding.
- **Prompts come at run time.** The guest emits READY (byte 511) and waits on `JOB_SEQ`; the runner writes the turn's tokens at `VM_TOKS + pos`, sets `JOB_N` and `JOB_GEN`, bumps `JOB_SEQ`. Positions go on through the turns: the context persists as a native model's does (`chip_run --chat`).
- Each position emits 3 bytes, the next token's id; the runner decodes reply tokens with the tokenizer in the file.

## 7. How a run is checked

No test programs. Two checks on the real job, and neither replaces the other:

1. **The layers against the reference.** The model's token hash per 4,096 tokens against the C reference's for the same job, live: `chip_host --ref`, a difference stops the run. A mismatch is a bug in an op or a bank: the count names the token, the reference names the instruction.
2. **The guest against the integer pass.** `src/tools/q2b64_check.cc` without `--nofix` compares all 248,320 logits at every position with `src/text/q2b_fix.cc`. `--notes`, with `out/guest64/q2b64_notes.elf` built with `-DQ64_NOTES`, compares each stage of each layer and stops at the first difference. `--selfstart` runs from position 0 with no restored context.

**The hash cannot see a wrong guest.** Check 2 was left for later while check 1 was green on 1.17 M tokens, and the guest was wrong at every position: 248,319 of 248,320 logits, by up to 1.66 M in Q.16, its own argmax junk. Widening the context to 65,536 had grown RoPE to 4.19 M words and pushed `VM_A`, `VM_DELTA`, `VM_O` past 2^22, and the state instructions carry their address in 22 bits: `VSUP` read its delta from the wrong memory. The layers did the same wrong thing as the reference, faithfully. `--notes` named layer 0's mix in one run. RoPE now sits above the activations and a static assert guards it (`src/guest64/q2b64_map.h`). Result: 42 positions at the real context, 0 logits differ, `logs/t13/ref-fix.txt`.

## 8. What is proven and what is not

Proven:

- **The guest computes the 2B, at the real context, on the host:** 42 positions after a 2,552-token system prompt, 0 of 248,320 logits differ from the integer pass at any of them, `logs/t13/ref-fix.txt`.
- **Layers in a file compute the gates, on the host:** a forward from the file alone against the exact evaluator, RV64 core + VPUT VGET VMAT8 as literal heads, 3,000 of 3,000 tokens, 0 of 3,549 dims differ, `logs/t9/embed-2.txt`.
- **The card's forward equals the reference on the real job, so far:** the hash per 4,096 tokens, live, through a resume, `logs/t13/card-snapshot/`. With every line computed and with change-driven evaluation, both.
- **`q2b_fix` itself against the exact 2B:** 94.17 % top-1, 100 % top-5 over 240 positions (`logs/t9/q2b-fix-ab-new.txt`): the integer pass is an approximation of the F32 model, not equal to it. A 2B token from this machine is the integer pass's token.

**Not proven yet:** the 12 NPU instructions beyond VPUT VGET VMAT8 in gates, and the memories' full range: judged only as the running hash passes through them, a small part of a position so far. A reply token through the layers: **none has been produced.** Chip tokens through Qwen's 40 layers: upstream's `mute()` is not ported, they are skipped. The 35B's own text through this file. The bridge, the 35B calling the 2B. The memories' expansion to literal heads at full size.

## 9. Cost

Measured: **about 712.9 M trace tokens per position at 2,552**, 13,509,301,000 to the first reply token of a 19-token prompt, 29,853,376,343 for the whole 42-position job (`logs/t13/ref.txt`, reference); 207.6-211.5 M at the start of a context (`logs/t12/self-haiku.txt`): the guest-level attention dot and p.V scale with the context.

Rates, trace tokens per second:

| Machine | Rate | Source |
|---|---|---|
| C interpreter, one host thread, not the model | 141.6-189.6 M | `logs/t13/ref.txt`, `ref-fix.txt`, `logs/native/run.txt` |
| The layers, one host thread | 513-546 | `logs/t10/run.txt` |
| The layers, card, every line computed | 178 | `logs/t11/m0e/run.txt` |
| The layers, card, change-driven, first engine | 279-289 | `logs/live` at the time, `logs/t11/m1/run.txt` |
| The layers, card, current engine | 335 over its first 20,480 tokens, table builds included; 409 over 30 s | `logs/t13/card-snapshot/run.txt` |

**152 M tok/s through the layers stays unreached.**

Where a token's time goes on the card, current engine, us (`logs/t13/card-snapshot/run.txt`): all 2,967: units 740, heads 526 (conv 256), snap 489, y 448, out 303 (mostly table builds amortized over a short window), hn 168, forms 119, scale 65, apply 61, banks 23, embed 9. 7.6 phases, 1,480 FFN lines, 292 attention lines, 503 y rows. Most of it is thread 0 and the members stalled on memory and on cache lines moving between cores, not arithmetic: a KNC thread also issues at most every other cycle.

Levers that keep the form, none built yet: the sibling thread sharing thread 0's serial work and L2; prefetching the lists being walked; chip ops for the attention dot and p.V (guest code today, about 3 K tokens per head per position, which would take the position cost back toward its start-of-context figure). Native kernels swapped in for compiled blocks are out: that is linking under another name.

`docs/diagram-inner.txt` draws it.
