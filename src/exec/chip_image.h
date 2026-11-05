/* The merged model file's computer as the 7120A holds it: the same tensors, laid out for its VPU. The host builds this from the file and moves it into
 * card memory; nothing here knows RV64 or the NPU. Plain C, shared by the host's builder (src/exec/chip_image.cc) and the card (src/card/chip_vpu.c).
 *
 * A matrix is stored by output lines of 16 rows. A line has one slice per column any of its rows reads: the column's index and a 64 B line of 16
 * weights, deduplicated across the whole model. Rows see their columns in ascending order, as the host's row by row definition does, so a line of
 * fused multiply-adds computes the same bits as sixteen rows of them.
 *
 * Every evaluated line keeps its last value here. hn at a sublayer changes in a few dims from one token to the next, so a line is computed again only
 * if a column it reads has changed: a line is a pure function of those columns, for any tensors. The fan tables say who reads what.
 *
 * All offsets count from the image's first byte. Everything is 64 B aligned where a line is read. */
#ifndef EXEC_CHIP_IMAGE_H
#define EXEC_CHIP_IMAGE_H

#include <stdint.h>

#define CHIP_IMAGE_MAGIC 0x31474D4950494843ULL
#define CHIP_LINE __attribute__((aligned(64)))
enum {
    CHIP_HIDDEN = 8192,
    CHIP_LINES = CHIP_HIDDEN / 16,
    CHIP_SETW = CHIP_LINES / 64,
    CHIP_ST = 128,
    CHIP_ST_LINES = CHIP_ST / 16,
    CHIP_BANK_LINES = 4,
    CHIP_BANK_OUTS = 64
};

/* One line of a linear form of hn: what it reads, where its last value lives, the token it last changed. */
typedef struct {
    uint32_t s0, len;
    uint64_t dst;
    uint32_t changed_at;
    uint32_t who;        /* whose input this is: kind << 24 | index: 1 a ChipConv, 2 a ChipHead, 0 no one to tell */
    uint32_t pad[2];
} ChipSell;

/* One line of the shared expert: 16 units, their gate and up forms. */
typedef struct {
    uint32_t g0, glen, u0, ulen;
    uint32_t line;
    uint32_t pad[11];
    float g[16] CHIP_LINE, s[16] CHIP_LINE;  /* the last gate value, and its silu */
} ChipUnit;

/* One line of the q, k, v projections with its causal conv. */
typedef struct {
    int32_t work;        /* the ChipSell of this line's projection */
    uint32_t conv_at;    /* the token its output last changed */
    uint32_t key;        /* the ChipKey that reads it */
    uint32_t pad[13];
    float cur[16] CHIP_LINE;       /* the projection */
    float w[4][16];      /* the taps, oldest first */
    float win[3][16];    /* a ring: the slot for token t is t mod 3 */
    float acc[16];       /* the last sum, before bf16 and silu */
    float conv[16];      /* silu(bf16(acc)) */
} ChipConv;

/* A key head: its q and k, normalized. */
typedef struct {
    int32_t q[CHIP_ST_LINES], k[CHIP_ST_LINES];  /* ChipConv indices, -1 for a line that stays zero */
    uint32_t head0, nheads;
    uint32_t q_at;       /* the token q last changed */
    uint32_t touched_at; /* the token a line of q, k or a head's v last changed */
    uint16_t qmask[CHIP_ST_LINES], kmask[CHIP_ST_LINES];  /* the nonzero lanes */
    uint32_t kany;       /* any lane of k */
    uint32_t pad1[3];
    float qv[CHIP_ST] CHIP_LINE, kv[CHIP_ST] CHIP_LINE;
} ChipKey;

/* A value head with something to read or write: the rest keep an empty state for ever. */
typedef struct {
    uint32_t hd;
    int32_t v[CHIP_ST_LINES];  /* ChipConv indices */
    int32_t z[CHIP_ST_LINES];  /* ChipSell ids */
    int32_t alpha, beta;       /* ChipSell ids, -1 for a row with nothing in it */
    uint32_t dt_bits, a_bits;
    uint32_t alpha_last, beta_last, e_bits, b_bits;  /* the last decay and write gain, computed once per distinct input */
    uint32_t fresh;
    uint32_t o_at;             /* the token o last changed */
    uint32_t mix_line;         /* its first line in the layer's mix */
    uint32_t touched_at;       /* the token a form of this head last changed */
    uint32_t pad0[3];
    uint64_t rows_nz[2];       /* which rows of S can be nonzero */
    uint32_t pad1[14];
    float alpha_v[16] CHIP_LINE, beta_v[16] CHIP_LINE;  /* lane 0 */
    float z_v[CHIP_ST] CHIP_LINE, o_v[CHIP_ST] CHIP_LINE;
    float S[CHIP_ST * CHIP_ST] CHIP_LINE;
} ChipHead;

/* A memory the file holds in one piece. kind: 0 the weights' tiles, 1 words, 2 rows of 4 int16, 3 bytes. */
typedef struct {
    int32_t kind, nra, nwa, nd;
    int32_t ra[CHIP_BANK_LINES], wa[CHIP_BANK_LINES], data[CHIP_BANK_LINES], we;  /* ChipSell ids, -1 for a line that stays zero */
    uint32_t nout;
    uint64_t o_mem, mem_bytes;
    uint16_t out[CHIP_BANK_OUTS];
    float vra[16 * CHIP_BANK_LINES] CHIP_LINE, vwa[16 * CHIP_BANK_LINES] CHIP_LINE, we_v[16] CHIP_LINE;
    float cur[16 * CHIP_BANK_LINES] CHIP_LINE;
    float taps[4][16 * CHIP_BANK_LINES] CHIP_LINE;
    float win[3][16 * CHIP_BANK_LINES] CHIP_LINE;
} ChipBank;

/* One sublayer's memo: who reads which dim, and what it last computed. */
typedef struct {
    uint64_t o_work;                   /* the attention side: ChipSell[]; the FFN side: ChipUnit[] */
    uint32_t n_work, n_y;
    uint64_t o_fan_start, o_fan_item;  /* hn dim -> work ids */
    uint64_t o_yv;                     /* y's values by dim, after the bf16 rounding: a row keeps its last */
    uint64_t o_yrow_dim, o_yrow_start, o_yrow_col, o_yrow_w;  /* y's rows: which dim, and its columns ascending */
    uint64_t o_yfan_start, o_yfan_item;
    uint64_t o_input;                  /* what y reads: the mix, or the units */
    uint64_t o_norm;                   /* the RMS norm's weights */
    uint64_t o_reads;                  /* a u16 per line of hn: the lanes anything here reads */
    uint64_t o_dirty, o_list;          /* a byte per work line, and this sublayer's list of them */
    uint64_t o_ydirty;                 /* a byte per y row: to compute */
    uint64_t norm_diff[CHIP_SETW];     /* lines where this norm differs from the sublayer before */
    uint64_t ynz[CHIP_SETW];           /* y's lines with anything in them */
    uint32_t any_norm_diff;
    uint32_t token_now;                /* the token a team member stamps a change with */
    uint32_t pad[14];
} ChipSub;

typedef struct {
    ChipSub a, f;
    uint64_t o_conv, o_keys, o_heads, o_banks;  /* ChipConv[], ChipKey[], ChipHead[], ChipBank[] */
    uint32_t n_conv, n_keys, n_heads, n_banks;
    uint64_t o_ssm_norm;                        /* CHIP_ST floats */
    uint64_t o_gi_idx, o_gi_w;                  /* the gate input's nonzeros, ascending */
    uint32_t n_gi;
    uint32_t gi_fresh, gi_last, pad0;
    uint64_t o_act, o_in;                       /* the conv lines still moving: u16[n_conv], and a byte per line: is it listed */
    uint32_t n_act;
    uint32_t pad1;
    float gate_s[16] CHIP_LINE;                 /* the FFN's gate, in every lane */
} ChipLayer;

typedef struct {
    uint64_t magic, image_bytes, total_bytes;
    uint32_t layers, vocab, go_token, byte_token, system_tokens, ctx;
    uint64_t o_layers;
    uint64_t o_col, o_woff, o_wtab;             /* slices */
    uint64_t nnz, slices, wlines;
    uint64_t o_emb_start, o_emb_col, o_emb_val; /* the embedding by rows: u32[vocab + 1], u16[], f32[] */
    uint64_t emb_nnz;
    uint64_t o_byte_f;                          /* i / 256, for 256 i */
    uint64_t o_x, o_hn, o_ytmp, o_hp;           /* the residual, its norm, this sublayer's y, and hn one token ago */
    uint64_t o_row_used;                        /* a byte per dim: does any row of the embedding read it */
    uint32_t eps_bits, e2_bits, over_h_bits, over_st_bits, q_scale_bits, one_bits;
    uint32_t pad[2];
} ChipHeader;

/* The exchange with the host, at the same place in the shared host memory as the text path's (src/text/q35_link.h): only one runs at a time. */
enum { CHIP_OP_NONE, CHIP_OP_BLOB, CHIP_OP_START, CHIP_OP_POKE };
enum { CHIP_EV_NONE, CHIP_EV_HASH, CHIP_EV_BYTE, CHIP_EV_READY, CHIP_EV_LOG, CHIP_EV_END, CHIP_EV_BAD, CHIP_EV_CKPT };
enum { CHIP_EVENTS = 1024, CHIP_POKES = 40000, CHIP_STATS = 40 };
#define CHIP_LINK_MAGIC 0x314B4E494C504843ULL
#define CHIP_POKE_OFFSET (256u * 1024)  /* within the part: ChipPoke[] */
/* Where a checkpoint's pack of windows and states goes in card memory: past the image and the engine's arrays. */
#define CHIP_PACK_AT(total) (((uint64_t)(total) + 4095) / 4096 * 4096 + (128ULL << 20))

typedef struct {
    uint32_t word;   /* into VM */
    int32_t value;
    uint32_t add;    /* 0 set, 1 add */
    uint32_t pad;
} ChipPoke;

typedef struct {
    uint64_t kind, a, b, c;
} ChipEvent;

typedef struct {
    volatile uint64_t magic;
    volatile uint64_t card_ready;
    char pad0[48];
    volatile uint64_t cmd_seq, cmd_op, cmd_a, cmd_b;  /* host */
    volatile uint64_t stop;                           /* host: read between tokens now and then */
    char pad1[24];
    volatile uint64_t res_seq;                        /* card */
    char pad2[56];
    volatile uint64_t ev_head;                        /* card: events written so far */
    char pad3[56];
    volatile uint64_t stats[CHIP_STATS];              /* card: ticks and counts, for the report */
    volatile uint64_t ck_total, ck_pack, ck_seq, ck_len, ck_ack;  /* a checkpoint on its way: card, card, card, card, host */
    volatile uint64_t res_n, res_token, res_hash;     /* host: where a resumed run goes on from */
    volatile ChipEvent ev[CHIP_EVENTS];
} ChipCtl;

#endif
