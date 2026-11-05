/* The SIMT machine's encodings: instructions, blocks, tokens, the launch queue. One definition shared by the reference
 * (sm_ref.c), the driver (lower.cc), the circuit (sm*.cc), the emitter and the tools. docs/machine.md is the contract
 * in words; this file is the contract in bits. Plain C. */
#ifndef GPU_SM_H
#define GPU_SM_H

#ifndef __OPENCL_C_VERSION__
#include <stdint.h>  /* OpenCL C gets these types from src/cl/qgpucl.h */
#endif

enum {
    SM_LANES = 32,
    SM_NREG = 16,          /* lane registers, r15 reads 0 */
    SM_NUREG = 16,         /* uniform registers, u15 reads 0 */
    SM_ZERO = 15,
    SM_UARGS = 14,         /* the uniform register the sequencer loads with a launch's args word (descriptor word 1) */
    SM_NPRED = 8,
    SM_STAGES = 6,         /* instructions per block */
    SM_LMAX = 4,           /* live-outs per block */
    SM_BBITS = 12,         /* blocks in the ROM */
    SM_NBLK = 1 << SM_BBITS,
    SM_WBITS = 8,          /* warps per launch */
    SM_TBITS = SM_WBITS + 5,
    SM_WPB_MAX = 8,        /* warps per thread block with barriers */
    SM_QBITS = 12,         /* launch descriptors */
    SM_NIB = 8
};

/* token ids: a 32-bit value is two chunks, VLO then VHI, or VLO then VEND when it is the last item of a lane's part (or of
 * the uniform part, or of an emit): the circuit's counters step on the kinds */
enum {
    SM_VLO0 = 0,
    SM_VHI0 = 1 << 16,
    SM_VEND0 = 2 << 16,
    SM_VAL_MAX = 3 << 16,
    SM_EXEC0 = SM_VAL_MAX,                    /* EXEC(b, phase) = EXEC0 + b + phase << BBITS */
    SM_NEXT0 = SM_EXEC0 + (1 << (SM_BBITS + 1)),
    SM_EMIT = SM_NEXT0 + (1 << SM_QBITS),
    SM_HALT = SM_EMIT + 1,
    SM_GO = SM_EMIT + 2,
    SM_LSTEP = SM_EMIT + 3,  /* a lane's part of a record when the block has no lane live-outs or reductions */
    SM_TEXT = SM_EMIT + 4,
    SM_WARP0 = SM_EMIT + 5,  /* WARP(w) = WARP0 + w: the warp the next record runs for, when it changes (END, BAR) */
    SM_CHIP_VOCAB = SM_WARP0 + (1 << SM_WBITS)
};
static inline uint32_t sm_warp_tok(uint32_t w) { return SM_WARP0 + w; }

static inline uint32_t sm_exec(uint32_t b, uint32_t phase) { return SM_EXEC0 + b + (phase << SM_BBITS); }
static inline uint32_t sm_next(uint32_t k) { return SM_NEXT0 + k; }

/* ops */
enum {
    SM_NOP, SM_ADD, SM_SUB, SM_MAD, SM_MULHI, SM_MULSH, SM_AND, SM_OR, SM_XOR, SM_SHL, SM_SHR, SM_MAX, SM_MIN,
    SM_SETP, SM_SELP, SM_TID, SM_SREG, SM_LD, SM_STV, SM_Q8DOT, SM_SDOT, SM_SUP, SM_SCFG, SM_WSUM, SM_WMAX,
    SM_UDIV, SM_USQRT, SM_CLZ,
    SM_NOPS
};

/* stage classes: a bit per class */
enum {
    SM_CL_ANY = 1, SM_CL_MUL = 2, SM_CL_DOT = 4, SM_CL_STATE = 8, SM_CL_LOGIC = 16, SM_CL_RED = 32, SM_CL_SCALAR = 64
};

/* the chip's choice of units per stage (measured 11-04, logs/g4/classes.txt): the deep units (mul, dot, state) in stages 1, 3, 5, the
 * scalar unit in stage 5 only; the shallow classes everywhere. Three deep stages cost 1.35x the tokens of six on the 2B's job and
 * about 60 % of their depth in the layers. */
#define SM_STAGE_CLASSES_DEFAULT \
    { 49, 63, 49, 63, 49, 127 }

#ifndef __OPENCL_C_VERSION__
#include <stdio.h>
#include <stdlib.h>
/* QGPU_CLASSES="c0,c1,c2,c3,c4,c5" overrides the stage classes (the driver and the reference must see the same) */
static inline void sm_classes_env(uint8_t classes[SM_STAGES]) {
    const char *e = getenv("QGPU_CLASSES");
    if (!e) return;
    unsigned v[SM_STAGES];
    if (sscanf(e, "%u,%u,%u,%u,%u,%u", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]) != SM_STAGES) {
        fprintf(stderr, "QGPU_CLASSES: six comma-separated class masks expected\n");
        exit(2);
    }
    for (int s = 0; s < SM_STAGES; s++) classes[s] = (uint8_t)v[s];
}
#endif

enum { SM_CMP_EQ, SM_CMP_NE, SM_CMP_LT, SM_CMP_LE, SM_CMP_GT, SM_CMP_GE };
enum { SM_SRC_LANE = 0, SM_SRC_UNIFORM = 1, SM_SRC_IMM = 2 };
enum { SM_BANK_M, SM_BANK_A8, SM_BANK_A32, SM_BANK_S, SM_BANK_C, SM_BANK_W8, SM_BANK_W32, SM_NBANKS_LD };  /* W8: an int8 of a weight block (block * 64 + j), W32: its scale as Q.24 */

/* an instruction, as fields */
typedef struct {
    uint8_t op, rd, ra, rb, rc, valid, pd, cmp, sgn, ub, ua, uc, ud, gvalid, g, gneg, bank, sh;  /* sh: mulsh's and the shifts' amount, 0..63 */
    uint32_t imm;
} SmInstr;

/* the 80-bit ROM row: 3 chunks of fields, 2 of the immediate */
static inline void sm_encode(const SmInstr *i, uint16_t out[5]) {
    uint64_t f = (uint64_t)(i->op & 31) | (uint64_t)(i->rd & 15) << 5 | (uint64_t)(i->ra & 15) << 9 | (uint64_t)(i->rb & 15) << 13
               | (uint64_t)(i->rc & 15) << 17 | (uint64_t)(i->valid & 1) << 21 | (uint64_t)(i->pd & 7) << 22 | (uint64_t)(i->cmp & 7) << 25
               | (uint64_t)(i->sgn & 1) << 28 | (uint64_t)(i->ub & 3) << 29 | (uint64_t)(i->ua & 1) << 31 | (uint64_t)(i->uc & 1) << 32
               | (uint64_t)(i->ud & 1) << 33 | (uint64_t)(i->gvalid & 1) << 34 | (uint64_t)(i->g & 7) << 35 | (uint64_t)(i->gneg & 1) << 38
               | (uint64_t)(i->bank & 7) << 39 | (uint64_t)(i->sh & 63) << 42;
    out[0] = (uint16_t)f;
    out[1] = (uint16_t)(f >> 16);
    out[2] = (uint16_t)(f >> 32);
    out[3] = (uint16_t)i->imm;
    out[4] = (uint16_t)(i->imm >> 16);
}

static inline void sm_decode(const uint16_t in[5], SmInstr *i) {
    uint64_t f = (uint64_t)in[0] | (uint64_t)in[1] << 16 | (uint64_t)in[2] << 32;
    i->op = f & 31;
    i->rd = f >> 5 & 15;
    i->ra = f >> 9 & 15;
    i->rb = f >> 13 & 15;
    i->rc = f >> 17 & 15;
    i->valid = f >> 21 & 1;
    i->pd = f >> 22 & 7;
    i->cmp = f >> 25 & 7;
    i->sgn = f >> 28 & 1;
    i->ub = f >> 29 & 3;
    i->ua = f >> 31 & 1;
    i->uc = f >> 32 & 1;
    i->ud = f >> 33 & 1;
    i->gvalid = f >> 34 & 1;
    i->g = f >> 35 & 7;
    i->gneg = f >> 38 & 1;
    i->bank = f >> 39 & 7;
    i->sh = f >> 42 & 63;
    i->imm = (uint32_t)in[3] | (uint32_t)in[4] << 16;
}

/* branches */
enum { SM_BR_NEXT, SM_BR_JUMP, SM_BR_COND, SM_BR_END, SM_BR_BAR };

/* a block's LIVE entry: 64 bits */
typedef struct {
    uint8_t nlive;
    uint8_t src[SM_LMAX];  /* stage of each live-out, in emission order */
    uint8_t branch, stage, neg, any, lwpb;
    uint16_t target;
} SmLive;

static inline uint64_t sm_live_encode(const SmLive *l) {
    uint64_t v = (uint64_t)(l->nlive & 7);
    for (int i = 0; i < SM_LMAX; i++) v |= (uint64_t)(l->src[i] & 7) << (3 + 3 * i);
    v |= (uint64_t)(l->branch & 7) << 15 | (uint64_t)(l->stage & 7) << 18 | (uint64_t)(l->neg & 1) << 21 | (uint64_t)(l->any & 1) << 22
       | (uint64_t)(l->target & 4095) << 23 | (uint64_t)(l->lwpb & 3) << 35;
    return v;
}

static inline void sm_live_decode(uint64_t v, SmLive *l) {
    l->nlive = v & 7;
    for (int i = 0; i < SM_LMAX; i++) l->src[i] = v >> (3 + 3 * i) & 7;
    l->branch = v >> 15 & 7;
    l->stage = v >> 18 & 7;
    l->neg = v >> 21 & 1;
    l->any = v >> 22 & 1;
    l->target = v >> 23 & 4095;
    l->lwpb = v >> 35 & 3;
}

/* the ROM: per block, STAGES rows of 5 chunks and one LIVE entry */
typedef struct {
    uint16_t rows[SM_STAGES][5];
    uint64_t live;
} SmBlock;

/* the launch queue: two words per descriptor */
enum { SM_Q_LAUNCH, SM_Q_BRANCH, SM_Q_EMIT, SM_Q_HALT };
static inline uint32_t sm_qword(uint32_t kind, uint32_t entry, uint32_t warps, uint32_t lwpb, uint32_t target) {
    return (kind & 3) | (entry & 4095) << 2 | (warps & 255) << 14 | (lwpb & 3) << 22 | (target & 255) << 24;  /* 256 warps are written as 0 */
}
/* target descriptors above 255 go in the high half of word 1 for branches: word1 = address | target_hi << 24 */

/* where the fixed words of M live */
enum {
    SM_M_QUEUE = 0,            /* 4096 descriptors x 2 words */
    SM_M_JOB = 8192,           /* the job words: seq, n, gen, pos, nstops, stops... */
    SM_M_USER = 8192 + 64      /* the guest's map starts here */
};

#endif
