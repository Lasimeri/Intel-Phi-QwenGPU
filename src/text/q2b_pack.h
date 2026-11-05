/* The 2B's pack for the 7120A: one blob the card maps into 2 MiB pages, every array an offset from its start. Shared by the
 * host's packer (C++) and the card's engine (C). The weights are the GGUF's Q8_0 values, unchanged, moved into tiles; the
 * block scales become Q.24 integers; constants and function tables are made once on the host, so the card and the host's
 * integer pass (src/text/q2b_fix.cc) hold the same bits.
 *
 * A tile: 16 rows x 32 columns, 576 B, 64 B aligned: 512 B of int8 by column (byte 16 i + r is column i of row r), then
 * 16 int32 scales, one per row. A matrix's rows come in groups of 16, and a group's tiles are consecutive. */
#ifndef TEXT_Q2B_PACK_H
#define TEXT_Q2B_PACK_H

#include <stdint.h>

#define Q2B_MAGIC 0x314B5032u
#define Q2B_D 2048
#define Q2B_LAYERS 24
#define Q2B_VHEADS 16
#define Q2B_KHEADS 16
#define Q2B_ST 128
#define Q2B_KEY_DIM (Q2B_KHEADS * Q2B_ST)
#define Q2B_VAL_DIM (Q2B_VHEADS * Q2B_ST)
#define Q2B_QKV (2 * Q2B_KEY_DIM + Q2B_VAL_DIM)
#define Q2B_CONV 4
#define Q2B_QHEADS 8
#define Q2B_KVHEADS 2
#define Q2B_HD 256
#define Q2B_FF 6144
#define Q2B_ROT 64
#define Q2B_VOCAB 248320
#define Q2B_TILE 576
#define Q2B_ROPE_POS 4096

/* function tables: the same grids as q2b_fix.cc */
#define Q2B_FN_STEP 8
#define Q2B_FN_RANGE 8
#define Q2B_EXP_STEP 11
#define Q2B_EXP_RANGE 8
#define Q2B_SP_RANGE 16

typedef struct {
    uint64_t off;
    uint32_t rows, cols;
} Q2bMat;

typedef struct {
    Q2bMat qkv, gate, alpha, beta, out;   /* GDN */
    Q2bMat q, k, v, o;                    /* full attention */
    Q2bMat ffn_g, ffn_u, ffn_d;
    uint64_t attn_norm, post_norm;        /* int32 Q.16 arrays */
    uint64_t q_norm, k_norm;
    uint64_t conv, a, dt, ssm_norm;
} Q2bLayer;

typedef struct {
    uint32_t magic, pad;
    uint64_t total_bytes;
    uint64_t silu, sig, softplus, exp;    /* int32 tables */
    uint64_t rope;                        /* int32 Q.15: position x (32 cos, 32 sin) */
    uint64_t out_norm;
    Q2bMat emb;
    Q2bLayer layers[Q2B_LAYERS];
} Q2bPack;

#endif
