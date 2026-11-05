/* The reference SIMT machine: the token stream the circuit must produce, executed in C on the host. A test reference,
 * never the result. One machine over an image (ROM, the launch queue in M, the banks) and a sink for tokens. */
#ifndef GPU_SM_REF_H
#define GPU_SM_REF_H

#include <stddef.h>
#include <stdint.h>

#include "sm.h"

typedef struct {
    /* program */
    SmBlock *rom;
    uint32_t nblk;
    uint8_t classes[SM_STAGES];  /* SM_CL_* per stage: an instruction outside its stage's units is a fault */
    /* memories */
    int32_t *m;   uint32_t m_words;   /* M */
    uint8_t *a;   uint32_t a_blocks;  /* A: 36 bytes a block: 32 int8, int32 scale */
    int16_t *s;   uint32_t s_elems;   /* S */
    const uint8_t *w; uint32_t w_blocks;  /* W: 34 bytes a block: f16 scale, 32 int8; read-only */
    int8_t *c;    uint32_t c_bytes;   /* C */
    const int64_t *f16_q24;           /* 65536 entries: an f16's value in Q.24 */
} SmImage;

/* token sink: count, FNV-1a 64 hash, and an optional callback */
typedef struct {
    uint64_t n, hash;
    void (*emit)(void *ctx, uint32_t tok);
    void *ctx;
} SmSink;

typedef struct {
    SmImage *img;
    SmSink sink;
    /* machine state */
    uint32_t regs[SM_WPB_MAX][SM_NREG][SM_LANES];
    uint32_t ureg[SM_WPB_MAX][SM_NUREG]; /* one uniform file per warp slot (w mod 8): warp-uniform values, as CUDA has them */
    uint32_t qi;          /* the descriptor to run next */
    uint32_t w, b, R, entry; /* warp, block, the phase's start, the launch's entry block */
    uint32_t stop_qi;        /* 0 = run on; else sm_run returns 3 before running this descriptor (a probe's breakpoint) */
    uint32_t watch_lo, watch_hi; /* a store trace: every M store into [lo, hi) is printed to stderr with its block and warp */
    uint32_t trace_desc;     /* with trace_desc_on: after every block warp 0 commits in this descriptor, the uniform registers go to stderr */
    uint8_t trace_desc_on;
    uint32_t warps, lwpb; /* the launch's */
    uint8_t phase, halted, running, fault;
    uint8_t stop_req;     /* a sink may set it: sm_run returns 4 at the next block boundary */
    char fault_what[128];
    /* the state unit */
    int32_t scfg_e, scfg_sh1, scfg_sh2;
    /* the cross-lane hazard guard: what earlier lanes of this block stored (a lane may not load it: the fault names the kernel) */
    uint64_t hz_key[2048];
    uint32_t hz_gen[2048];
    uint32_t hz_cur, hz_n;
    /* counters */
    uint64_t records, launches, instrs, lsteps;
    uint64_t by_op[SM_NOPS];
} SmMachine;

#ifdef __cplusplus
extern "C" {
#endif
void sm_init(SmMachine *m, SmImage *img);
/* runs until HALT, a fault, or max_records records; returns 0 at HALT, 1 at the limit, 2 at a fault, 3 at stop_qi, 4 when the sink asked */
int sm_run(SmMachine *m, uint64_t max_records);
/* the host's turn: after writing M, GO */
void sm_go(SmMachine *m);
#ifdef __cplusplus
}
#endif
/* the reference's own hash of a token stream */
static inline uint64_t sm_fnv(uint64_t h, uint32_t tok) {
    h ^= tok;
    return h * 0x100000001B3ull;
}
#define SM_FNV0 0xCBF29CE484222325ull

/* the arithmetic the warp instructions use, shared with q2b_fix's successors */
static inline int64_t sm_shr_round(int64_t v, int s) {
    if (s <= 0) return (int64_t)((uint64_t)v << -s);
    return (v + ((int64_t)1 << (s - 1))) >> s;
}

#endif
