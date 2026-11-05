/* The 2B's memories on the machine: where everything lives in M (words), A (Q8 blocks), S (int16) and W (the weights'
 * blocks), shared by the kernels (CUDA), the image builder (src/tools/q2b_build.cc) and the checker. Plain C.
 *
 * W holds the GGUF's Q8_0 blocks of every matrix, concatenated in Q2B_MAT order: block (m, row r, block b) =
 * base[m] + r * nblocks[m] + b; the bases are in M at M_WBASE (so a kernel reads its matrix's base from M, or the
 * queue passes it in the args block). A matrix's input is a run of A blocks; its output a run of M words. */
#ifndef GUEST_Q2B_MAP_H
#define GUEST_Q2B_MAP_H

#include "../gpu/sm.h"

enum {
    Q2B_D = 2048, Q2B_LAYERS = 24, Q2B_VHEADS = 16, Q2B_KHEADS = 16, Q2B_ST = 128,
    Q2B_KEY_DIM = Q2B_KHEADS * Q2B_ST, Q2B_VAL_DIM = Q2B_VHEADS * Q2B_ST, Q2B_QKV = 2 * Q2B_KEY_DIM + Q2B_VAL_DIM,
    Q2B_QHEADS = 8, Q2B_KVHEADS = 2, Q2B_HD = 256, Q2B_FF = 6144, Q2B_ROT = 64, Q2B_VOCAB = 248320,
    Q2B_CTX = 65536, Q2B_GDN_LAYERS = 18, Q2B_ATT_LAYERS = 6,
    Q2B_FN_STEP = 8, Q2B_FN_RANGE = 8, Q2B_EXP_STEP = 11, Q2B_EXP_RANGE = 8, Q2B_SP_RANGE = 16,
    Q2B_SILU_N = 2 * Q2B_FN_RANGE << Q2B_FN_STEP, Q2B_SIG_N = Q2B_SILU_N, Q2B_SP_N = 2 * Q2B_SP_RANGE << Q2B_FN_STEP,
    Q2B_EXP_N = Q2B_EXP_RANGE << Q2B_EXP_STEP
};

/* the matrices, in W order */
enum { MAT_EMB, MAT_QKV, MAT_GATE, MAT_ALPHA, MAT_BETA, MAT_OUT, MAT_Q, MAT_K, MAT_V, MAT_O, MAT_FG, MAT_FU, MAT_FD, MAT_PER_LAYER = 12 };
/* layer il's matrix m (m >= MAT_QKV) has table index 1 + il * MAT_PER_LAYER + (m - 1) */
static inline int q2b_mat_index(int il, int m) { return m == MAT_EMB ? 0 : 1 + il * MAT_PER_LAYER + (m - 1); }
#define Q2B_NMATS (1 + Q2B_LAYERS * MAT_PER_LAYER)

static inline int q2b_is_full(int il) { return (il + 1) % 4 == 0; }
static inline int q2b_gdn_index(int il) { return il - il / 4; }  /* GDN layers before and including il, minus one */
static inline int q2b_att_index(int il) { return il / 4; }

/* ---------------------------------------------------------------- M, in words ----------------------------------- */
enum {
    /* the job (SM_M_JOB = 8192) */
    JOB_SEQ = SM_M_JOB,      /* bumped by the host for a new turn */
    JOB_N,                   /* tokens in the context once this turn's are in */
    JOB_GEN,                 /* the most to generate this turn */
    JOB_POS,                 /* the position being computed */
    JOB_PROMPT_N,            /* positions that are prompt (no reply yet) */
    JOB_MADE,                /* replies made this turn */
    JOB_NEXT,                /* the token just chosen */
    JOB_CONT,                /* 1: another position follows in this turn */
    JOB_NSTOPS,
    JOB_STOPS,               /* 8 words */
    JOB_ONE = JOB_STOPS + 8, /* the constant 1, for unconditional queue jumps */
    JOB_END = JOB_ONE + 1,

    M_ARGS = SM_M_USER,                    /* args blocks: 8 words each, one per descriptor */
    M_ARGS_WORDS = 8,
    M_ARGS_END = M_ARGS + M_ARGS_WORDS * (1 << SM_QBITS),
    M_SC = M_ARGS_END,                     /* scalars kernels pass to each other */
    SC_BIG = M_SC, SC_R, SC_SCALE, SC_S, SC_RS, SC_TOP, SC_SUM, SC_BEST, SC_BEST_IDX, SC_SKIP, SC_NOEMIT, SC_SPARE,
    SC_TMP = M_SC + 32,                    /* per thread block scratch: 16 words */
    SC_TMP2 = M_SC + 48,
    SC_TMP3 = M_SC + 64,
    M_SC_END = M_SC + 128,
    M_WBASE = M_SC_END,                    /* base block and nblocks per matrix: 2 words each */
    M_WBASE_END = M_WBASE + 2 * Q2B_NMATS,
    M_SILU = M_WBASE_END,
    M_SIG = M_SILU + Q2B_SILU_N,
    M_SP = M_SIG + Q2B_SIG_N,
    M_EXP = M_SP + Q2B_SP_N,
    M_OUTNORM = M_EXP + Q2B_EXP_N,
    M_X = M_OUTNORM + Q2B_D,               /* the residual */
    M_H = M_X + Q2B_D,                     /* its norm */
    M_Y = M_H + Q2B_D,                     /* a layer's output */
    M_T1 = M_Y + Q2B_D,                    /* scratch, FF wide */
    M_T2 = M_T1 + Q2B_FF,
    M_G = M_T2 + Q2B_FF,
    M_U = M_G + Q2B_FF,
    M_FFX = M_U + Q2B_FF,
    M_Z = M_FFX + Q2B_FF,                  /* the GDN gate z */
    M_CONV = M_Z + Q2B_VAL_DIM,            /* the conv output, silu'd */
    M_QI = M_CONV + Q2B_QKV,               /* GDN q, k, v in int8 as words */
    M_KI = M_QI + Q2B_KEY_DIM,
    M_VI = M_KI + Q2B_KEY_DIM,
    M_KREAL = M_VI + Q2B_VAL_DIM,          /* every head's k * kscale: VHEADS x ST (the kernel writes all heads) */
    M_MIX = M_KREAL + Q2B_VHEADS * Q2B_ST,
    M_A = M_MIX + Q2B_VAL_DIM,             /* a head's state read with k */
    M_DELTA = M_A + Q2B_ST,
    M_O = M_DELTA + Q2B_ST,                /* every head's state read with q: VHEADS x ST (the probe compares it) */
    M_OQ = M_O + Q2B_VHEADS * Q2B_ST,
    M_GDBG = M_OQ + Q2B_VHEADS * Q2B_ST,   /* per head 8 words the probe compares: obig s2 isq rms24 root24 f8 oscale nse */
    M_SZ = M_GDBG + Q2B_VHEADS * 8,        /* every head's silu(z), VHEADS x ST (the probe compares it; M_OQ holds `on`) */
    /* per-head words come in runs of 32: a warp writes 32 lanes, the 16 past the heads are scratch */
    M_ALPHA = M_SZ + Q2B_VHEADS * Q2B_ST,
    M_BETA = M_ALPHA + 32,
    M_DECAY = M_BETA + 32,                 /* e per head */
    M_BETAS = M_DECAY + 32,                /* sigmoid(beta) per head */
    M_QS = M_BETAS + 32,                   /* per key head */
    M_KS = M_QS + 32,
    M_VS = M_KS + 32,                      /* per value head */
    M_QG = M_VS + 32,                      /* a full-attention layer's q and gate projection */
    M_KK = M_QG + 2 * Q2B_QHEADS * Q2B_HD,
    M_VV = M_KK + Q2B_KVHEADS * Q2B_HD,
    M_QV = M_VV + Q2B_KVHEADS * Q2B_HD,    /* q heads normed and rotated */
    M_OA = M_QV + Q2B_QHEADS * Q2B_HD,     /* the attention output before o_proj */
    M_QSCALE = M_OA + Q2B_QHEADS * Q2B_HD, /* per q head, a run of 32 */
    M_P = M_QSCALE + 32,                   /* a head's scores over the positions */
    M_EXPV = M_P + Q2B_CTX,                /* their exponentials */
    M_TOKS = M_EXPV + Q2B_CTX,
    M_ROPE = M_TOKS + Q2B_CTX,             /* position x (32 cos, 32 sin), Q.15 */
    M_LAYERS = M_ROPE + Q2B_CTX * Q2B_ROT,

    /* a GDN layer's words */
    LG_ATTN = 0, LG_POST = LG_ATTN + Q2B_D, LG_CONV = LG_POST + Q2B_D, /* 4 taps by tap, each QKV: tap t at LG_CONV + t * QKV */
    LG_WIN = LG_CONV + 4 * Q2B_QKV,                                    /* 4 slots of the projection: slot (pos + t) & 3 */
    LG_NORM = LG_WIN + 4 * Q2B_QKV, LG_AV = LG_NORM + Q2B_ST, LG_DT = LG_AV + Q2B_VHEADS, LG_SE = LG_DT + Q2B_VHEADS,
    LG_SMAX = LG_SE + Q2B_VHEADS, LG_SIZE = LG_SMAX + Q2B_VHEADS,
    /* a full-attention layer's words */
    LA_ATTN = 0, LA_POST = LA_ATTN + Q2B_D, LA_QN = LA_POST + Q2B_D, LA_KN = LA_QN + Q2B_HD,
    LA_VSMAX = LA_KN + Q2B_HD, /* the largest vs so far per kv head, a run of 32 */
    LA_VSC = LA_VSMAX + 32,   /* vs per (position, kv head): position * 2 + head */
    LA_SIZE = LA_VSC + Q2B_CTX * Q2B_KVHEADS,

    M_END = M_LAYERS + Q2B_GDN_LAYERS * LG_SIZE + Q2B_ATT_LAYERS * LA_SIZE
};
/* a layer's words: the GDN layers before it, then the full-attention layers before it */
static inline uint32_t q2b_layer_m(int il) {
    const uint32_t full_before = (uint32_t)il / 4, gdn_before = (uint32_t)il - full_before;
    return (uint32_t)M_LAYERS + gdn_before * (uint32_t)LG_SIZE + full_before * (uint32_t)LA_SIZE;
}

/* ---------------------------------------------------------------- A, in blocks ---------------------------------- */
enum {
    A_H = 0,                                /* the normed residual, 64 blocks */
    A_FFX = A_H + Q2B_D / 32,               /* 192 */
    A_MIX = A_FFX + Q2B_FF / 32,            /* 64 */
    A_OA = A_MIX + Q2B_VAL_DIM / 32,        /* 64 */
    A_Q = A_OA + Q2B_QHEADS * Q2B_HD / 32,  /* q heads, 8 blocks each */
    A_F = A_Q + Q2B_QHEADS * Q2B_HD / 32,   /* the quantized probability weights of one head, CTX / 32 blocks */
    A_F2 = A_F + Q2B_CTX / 32,              /* their residual */
    A_KC = A_F2 + Q2B_CTX / 32,             /* the key cache: (att layer, kv head, position) x 8 blocks */
    A_VT = A_KC + Q2B_ATT_LAYERS * Q2B_KVHEADS * Q2B_CTX * (Q2B_HD / 32),  /* V transposed: (att layer, kv head, dim) x CTX / 32 */
    A_END = A_VT + Q2B_ATT_LAYERS * Q2B_KVHEADS * Q2B_HD * (Q2B_CTX / 32)
};
static inline uint32_t q2b_kc_block(int al, int kvh, uint32_t pos) {
    return (uint32_t)A_KC + (((uint32_t)al * Q2B_KVHEADS + (uint32_t)kvh) * Q2B_CTX + pos) * (Q2B_HD / 32);
}
static inline uint32_t q2b_vt_block(int al, int kvh, uint32_t dim, uint32_t pos) {
    return (uint32_t)A_VT + (((uint32_t)al * Q2B_KVHEADS + (uint32_t)kvh) * Q2B_HD + dim) * (Q2B_CTX / 32) + (pos >> 5);
}

/* ---------------------------------------------------------------- S, in int16 ----------------------------------- */
static inline uint32_t q2b_s_row(int gl, int hd, uint32_t i) { return (((uint32_t)gl * Q2B_VHEADS + (uint32_t)hd) * Q2B_ST + i) * Q2B_ST; }
#define Q2B_S_ELEMS ((uint32_t)Q2B_GDN_LAYERS * Q2B_VHEADS * Q2B_ST * Q2B_ST)

#endif
