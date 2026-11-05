/* The tile pack: the GGUF's Q8_0 weights permuted, not changed, into what the VPU reads. Plain C, shared by
 * the host's packer and streamer and the card.
 *
 * A Q8_0 matrix goes in groups of 16 rows. Per group, per 32-column block, one 544 B tile:
 *     512 B   for each of the 32 columns, that column's 16 int8 weights
 *      32 B   the 16 rows' f16 block scales
 * Each 16 B of weights is one {sint8} load, each 32 B of scales one {float16} operand, both at their own
 * alignment. 34 B per row per block, exactly as in the GGUF.
 *
 * An F32 matrix of 16-row groups stores 16 floats per column: one 64 B line.
 *
 * The file: a header page, the dense section (everything every token reads), then every expert. An
 * expert's gate, up and down tiles are adjacent, so a miss is one sequential read. Lengths and offsets of
 * every read are multiples of 4096 for O_DIRECT. */
#ifndef TEXT_Q35_LAYOUT_H
#define TEXT_Q35_LAYOUT_H

#include <stdint.h>

#define Q35_PACK_MAGIC 0x3154355133ULL
#define Q35_TILE 544u
#define Q35_PAGE 4096u

enum {
    Q35_D = 2048,
    Q35_LAYERS = 40,
    Q35_VHEADS = 32,
    Q35_KHEADS = 16,
    Q35_ST = 128,
    Q35_KEY_DIM = Q35_KHEADS * Q35_ST,
    Q35_VAL_DIM = Q35_VHEADS * Q35_ST,
    Q35_QKV = 2 * Q35_KEY_DIM + Q35_VAL_DIM,
    Q35_QHEADS = 16,
    Q35_KVHEADS = 2,
    Q35_HD = 256,
    Q35_NEXP = 256,
    Q35_USED = 8,
    Q35_FF = 512,
    Q35_ROT = 64,
    Q35_VOCAB = 248320
};

static inline int q35_is_full(int il) { return (il + 1) % 4 == 0; }

/* bytes of a Q8_0 matrix in tiles */
static inline uint64_t q35_q8_bytes(uint64_t rows, uint64_t cols) { return rows / 16 * (cols / 32) * Q35_TILE; }
static inline uint64_t q35_group_bytes(uint64_t cols) { return cols / 32 * Q35_TILE; }

#define Q35_EXPERT_PART ((uint64_t)Q35_FF / 16 * (Q35_D / 32) * Q35_TILE)   /* gate or up: 1,114,112 B */
#define Q35_EXPERT_DOWN ((uint64_t)Q35_D / 16 * (Q35_FF / 32) * Q35_TILE)   /* down: 1,114,112 B */
#define Q35_EXPERT_BYTES (2 * Q35_EXPERT_PART + Q35_EXPERT_DOWN)            /* 3,342,336 B, 816 pages */

/* Offsets of one layer's dense data, from the start of the dense section. */
typedef struct {
    uint64_t attn_norm, post_norm, gate_sh, router, g_sh, u_sh, d_sh;
    uint64_t a, dt, ssm_norm, conv, qkv, gate, alpha, beta, out; /* Gated DeltaNet */
    uint64_t q_norm, k_norm, q, k, v, o;                         /* full attention */
    uint64_t end;
} Q35Dense;

static inline uint64_t q35_take(uint64_t *at, uint64_t bytes) {
    uint64_t here = *at;
    *at = (here + bytes + 63) / 64 * 64;
    return here;
}

static inline void q35_dense_layout(int il, uint64_t start, Q35Dense *L) {
    uint64_t at = start;
    L->attn_norm = q35_take(&at, Q35_D * 4);
    L->post_norm = q35_take(&at, Q35_D * 4);
    L->gate_sh = q35_take(&at, Q35_D * 4);
    L->router = q35_take(&at, (uint64_t)Q35_NEXP / 16 * Q35_D * 64);
    L->g_sh = q35_take(&at, q35_q8_bytes(Q35_FF, Q35_D));
    L->u_sh = q35_take(&at, q35_q8_bytes(Q35_FF, Q35_D));
    L->d_sh = q35_take(&at, q35_q8_bytes(Q35_D, Q35_FF));
    L->a = L->dt = L->ssm_norm = L->conv = L->qkv = L->gate = L->alpha = L->beta = L->out = 0;
    L->q_norm = L->k_norm = L->q = L->k = L->v = L->o = 0;
    if (q35_is_full(il)) {
        L->q_norm = q35_take(&at, Q35_HD * 4);
        L->k_norm = q35_take(&at, Q35_HD * 4);
        L->q = q35_take(&at, q35_q8_bytes(2 * Q35_QHEADS * Q35_HD, Q35_D));
        L->k = q35_take(&at, q35_q8_bytes(Q35_KVHEADS * Q35_HD, Q35_D));
        L->v = q35_take(&at, q35_q8_bytes(Q35_KVHEADS * Q35_HD, Q35_D));
        L->o = q35_take(&at, q35_q8_bytes(Q35_D, Q35_QHEADS * Q35_HD));
    } else {
        L->a = q35_take(&at, Q35_VHEADS * 4);
        L->dt = q35_take(&at, Q35_VHEADS * 4);
        L->ssm_norm = q35_take(&at, Q35_ST * 4);
        L->conv = q35_take(&at, (uint64_t)4 * Q35_QKV * 4); /* tap by tap: [4][QKV] */
        L->qkv = q35_take(&at, q35_q8_bytes(Q35_QKV, Q35_D));
        L->gate = q35_take(&at, q35_q8_bytes(Q35_VAL_DIM, Q35_D));
        L->alpha = q35_take(&at, q35_q8_bytes(Q35_VHEADS, Q35_D));
        L->beta = q35_take(&at, q35_q8_bytes(Q35_VHEADS, Q35_D));
        L->out = q35_take(&at, q35_q8_bytes(Q35_D, Q35_VAL_DIM));
    }
    L->end = at;
}

/* From the start of the dense section: the output norm, the output head, and the section's length. */
static inline void q35_dense_tail(uint64_t *out_norm, uint64_t *output, uint64_t *total) {
    Q35Dense L;
    uint64_t at = 0;
    for (int il = 0; il < Q35_LAYERS; il++) {
        q35_dense_layout(il, at, &L);
        at = L.end;
    }
    *out_norm = q35_take(&at, Q35_D * 4);
    *output = q35_take(&at, q35_q8_bytes(Q35_VOCAB, Q35_D));
    *total = (at + Q35_PAGE - 1) / Q35_PAGE * Q35_PAGE;
}

/* file offset of an expert, given where the dense section ends */
static inline uint64_t q35_expert_off(uint64_t experts_start, int il, int e) {
    return experts_start + ((uint64_t)il * Q35_NEXP + (uint64_t)e) * Q35_EXPERT_BYTES;
}

#define Q35_DENSE_OFF ((uint64_t)Q35_PAGE)

#endif
