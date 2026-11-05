/* The merged model's computer on the 7120A's VPU: the forward pass of src/exec/tforward.cc, operation for operation, from the tensors in the image and
 * nothing else. It knows no RV64 and no NPU. Every float is in a zmm register: KNC's x87 keeps scalars in extended precision, so a float here is only ever
 * loaded, computed on in a vector, stored, or compared as bits.
 *
 * What makes it fast holds for any tensors:
 *   - x and hn change in few lines, so hn is recomputed where x changed, and a line whose value did not change is not written: other cores keep their
 *     copy of it. hn at a sublayer is compared with hn at that sublayer last token, on the lines that can differ.
 *   - A form's line, a unit, a head's output are pure functions of what they read: each keeps its last value and is computed again only if a column it
 *     reads changed. Fan tables go from a changed dim to its readers.
 *   - A zero column adds exactly nothing (vpu_sell_chunk) and a closed unit's up form need not be read.
 *
 * The team is a 4-ary heap over thread ids (a core's two threads are neighbours), each hop one cache line. An empty phase at 120 threads has cost about
 * 8 us (logs/t7/barrier1.txt), as much as about 100 lines' work on one thread, so a sublayer with few lines to do runs them on thread 0.
 */
#include <immintrin.h>
#include <stdlib.h>
#include <string.h>

#include "chip_engine.h"
#include "q35_elem.h"

void vpu_sell_chunk(const float *R, const uint32_t *idx, const uint32_t *woff, const float *wtab, uint32_t len, float *out);

#define ALIGNED64 __attribute__((aligned(64)))
enum { SPIN_DELAY = 8, TEAM_ARY = 4 };
enum { OP_NOP, OP_HN_FORMS, OP_UNITS, OP_STOP };
enum { TABLE_BITS = 21, MAX_OUT_DIMS = 64, MAX_STRAYS = 64, LOG_LINES = 32768 };

typedef struct {
    volatile uint64_t go;    /* generation << 32 | width << 24 | op << 22 | sublayer << 16 | lines */
    char pad0[56];
    volatile uint32_t done;
    char pad1[60];
} Sync;

typedef struct {
    uint32_t *chg;           /* the unit lines whose output changed: line id | lanes << 16 */
    uint32_t nchg;
    char pad[52];
} Member;

typedef struct {
    uint64_t key;
    uint32_t token;
    uint32_t used;
} Entry;

struct ChipEngine {
    uint8_t *base;
    const ChipHeader *h;
    ChipLayer *layers;
    const uint32_t *col, *woff;
    const float *wtab;
    const uint32_t *emb_start;
    const uint16_t *emb_col;
    const float *emb_val;
    const float *byte_f;
    const uint8_t *row_used;
    float *x, *hn, *ytmp, *hp;
    float *logv[2];          /* the lines hn was written with, per token, and which lines */
    uint16_t *logl[2];
    int nthreads, width, par_min, memo;
    Sync *sync;
    Member *mem;
    Entry *table;
    int32_t *vm;
    uint8_t *owner;          /* a line of work -> its member */
    /* everything above is read-only during a run; only thread 0 touches what follows */
    uint32_t gen __attribute__((aligned(64)));
    uint32_t token_no, r0, r1, r2;
    uint32_t scale_bits;
    uint32_t m_bits, big_n, big_bits[32];   /* prove_scale's bound and big dims */
    uint16_t big_dim[32];
    int proved, fast;
    float scale_line[16] ALIGNED64, scale_fixed[16] ALIGNED64;
    uint64_t xnz[CHIP_SETW];       /* lines of x that can be nonzero */
    uint64_t yset[CHIP_SETW];      /* lines of ytmp in use */
    uint64_t chg[CHIP_SETW];       /* lines of hn this sublayer changed */
    uint64_t diff[CHIP_SETW];      /* lines of hn that differ from last token's, at the sublayer before */
    uint32_t log_start[2][66];     /* where each sublayer's writes begin */
    uint32_t n_log, sub_no;
    int log_cur;
    uint16_t dmask[CHIP_LINES];    /* which lanes of hn differ from one token ago */
    volatile uint64_t chg_mask[3] ALIGNED64;
    /* the output head */
    int out_dims[MAX_OUT_DIMS], n_out;
    uint32_t out_max[MAX_OUT_DIMS];
    uint64_t full_scans, table_builds;
    uint32_t stray_dim[MAX_STRAYS], stray_bits[MAX_STRAYS];
    int n_strays, strays_told;
    uint8_t stray_seen[CHIP_HIDDEN];
    /* the report */
    uint64_t st[CHIP_STATS];
};

enum { S_TOKENS, S_T_EMBED, S_T_SCALE, S_T_HN, S_T_SNAP, S_T_FORMS, S_T_HEADS, S_T_BANKS, S_T_Y, S_T_APPLY, S_T_UNITS, S_T_OUT, S_PHASES, S_LINES_A, S_LINES_F, S_LINES_Y, S_PAR_LINES, S_HN_WRITES, S_CHG_UNITS, S_CONV, S_E_PASS, S_KEYED, S_O, S_MIX, S_FULL_SCANS, S_TABLE, S_T_TOKEN, S_T_CONV, S_MARKS, S_SNAP_DIFF, S_FAST };

static inline uint64_t ticks(void) {
    uint32_t lo, hi;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return (uint64_t)hi << 32 | lo;
}

void q35_pause(void) { __asm__ volatile("delay %0" : : "r"((long)32)); }
static inline void spin(void) { __asm__ volatile("delay %0" : : "r"((long)SPIN_DELAY)); }

/* ----------------------------------------------------------------------- lines, as bits and in zmm registers */

static inline __m512 v_bits(uint32_t u) { return _mm512_castsi512_ps(_mm512_set1_epi32((int)u)); }
static inline __m512 v_bc(const float *p) { return _mm512_extload_ps(p, _MM_UPCONV_PS_NONE, _MM_BROADCAST_1X16, 0); }
static inline uint32_t bits_of(const float *p) { return *(const uint32_t *)p; }

static inline __m512 bf16_lanes(__m512 v) {
    __m512i u = _mm512_castps_si512(v);
    __m512i odd = _mm512_and_si512(_mm512_srli_epi32(u, 16), _mm512_set1_epi32(1));
    u = _mm512_add_epi32(u, _mm512_add_epi32(_mm512_set1_epi32(0x7FFF), odd));
    u = _mm512_and_si512(u, _mm512_set1_epi32((int)0xFFFF0000u));
    return _mm512_castsi512_ps(u);
}

/* x < -87 is exactly 0, as src/text/elem.h's q_sigmoid: every closed gate relies on it */
static inline __m512 sigmoid0(__m512 x) { return v_blend(_mm512_cmplt_ps_mask(x, V_SET1(-87.0f)), _mm512_setzero_ps(), v_sigmoid(x)); }
static inline __m512 silu0(__m512 x) { return _mm512_mul_ps(x, sigmoid0(x)); }

/* as floats: KNC's header has no integer compares. A NaN counts as changed; +0 and -0 count as the same, and nothing downstream can tell them apart: a zero
 * column adds exactly nothing, whatever its sign. */
static inline uint32_t differs(__m512 a, __m512 b) { return _mm512_cmp_ps_mask(a, b, _CMP_NEQ_UQ); }
/* the lanes that are not zero, of either sign */
static inline uint32_t nonzero(__m512 a) { return _mm512_cmp_ps_mask(a, _mm512_setzero_ps(), _CMP_NEQ_UQ); }

/* --------------------------------------------------------------------------------- the team */

static inline void release_children(ChipEngine *e, int me, uint64_t msg, int width) {
    for (int c = TEAM_ARY * me + 1; c <= TEAM_ARY * me + TEAM_ARY && c < width; c++) e->sync[c].go = msg;
}

static inline void join_children(ChipEngine *e, int me, uint32_t g, int width) {
    for (int c = TEAM_ARY * me + 1; c <= TEAM_ARY * me + TEAM_ARY && c < width; c++)
        while (e->sync[c].done != g) spin();
}

/* A hn form's line, computed and kept if it changed. */
static inline void form_line(const ChipEngine *e, ChipSell *w, uint32_t t) {
    float tmp[16] ALIGNED64;
    vpu_sell_chunk(e->hn, e->col + w->s0, e->woff + w->s0, e->wtab, w->len, tmp);
    float *dst = (float *)(e->base + w->dst);
    const __m512 v = _mm512_load_ps(tmp);
    if (differs(v, _mm512_load_ps(dst))) {
        _mm512_store_ps(dst, v);
        w->changed_at = t;
    }
}

/* A line of units: silu(gate) x up x the expert's gate, bf16. Its gate's silu is computed only for a gate value it has not just seen, and a line of closed
 * units has no use for its up form. */
static inline uint32_t unit_line(const ChipEngine *e, ChipUnit *U, float *xs, const float *gate_s) {
    float g[16] ALIGNED64, u[16] ALIGNED64;
    vpu_sell_chunk(e->hn, e->col + U->g0, e->woff + U->g0, e->wtab, U->glen, g);
    const __m512 gv = _mm512_load_ps(g);
    __m512 s;
    if (differs(gv, _mm512_load_ps(U->g))) {
        s = silu0(gv);
        _mm512_store_ps(U->g, gv);
        _mm512_store_ps(U->s, s);
    } else {
        s = _mm512_load_ps(U->s);
    }
    __m512 now = _mm512_setzero_ps();
    if (nonzero(s)) {
        vpu_sell_chunk(e->hn, e->col + U->u0, e->woff + U->u0, e->wtab, U->ulen, u);
        now = bf16_lanes(_mm512_mul_ps(_mm512_mul_ps(s, _mm512_load_ps(u)), _mm512_load_ps(gate_s)));
    }
    float *dst = xs + 16 * (size_t)U->line;
    const uint32_t changed = differs(now, _mm512_load_ps(dst));
    if (changed) _mm512_store_ps(dst, now);
    return changed;
}

static void member_work(ChipEngine *e, int tid, int op, int sub, uint32_t lines, int width) {
    /* a line of work always falls to the same member, so what it keeps stays in that core's cache. On its own, thread 0 does them all. */
    const int alone = width == 1;
    const uint8_t *owner = e->owner;
    ChipLayer *L = &e->layers[sub >> 1];
    if (op == OP_HN_FORMS) {
        ChipSub *S = &L->a;
        ChipSell *work = (ChipSell *)(e->base + S->o_work);
        const uint32_t *list = (const uint32_t *)(e->base + S->o_list);
        const uint32_t now = S->token_now;
        for (uint32_t k = 0; k < lines; k++) {
            const uint32_t id = list[k];
            if (alone || owner[id] == tid) form_line(e, &work[id], now);
        }
    } else if (op == OP_UNITS) {
        ChipSub *S = &L->f;
        ChipUnit *work = (ChipUnit *)(e->base + S->o_work);
        const uint32_t *list = (const uint32_t *)(e->base + S->o_list);
        float *xs = (float *)(e->base + S->o_input);
        Member *M = &e->mem[tid];
        uint32_t n = 0;
        for (uint32_t k = 0; k < lines; k++) {
            const uint32_t id = list[k];
            if (!alone && owner[id] != tid) continue;
            const uint32_t ch = unit_line(e, &work[id], xs, L->gate_s);
            if (ch) M->chg[n++] = ch << 16 | id;
        }
        M->nchg = n;
        if (n) __sync_fetch_and_or(&e->chg_mask[tid >> 6], 1ULL << (tid & 63));
    }
}

void chip_engine_worker(ChipEngine *e, int tid) {
    uint64_t last = 0;
    for (;;) {
        uint64_t msg;
        while ((msg = e->sync[tid].go) == last) spin();
        last = msg;
        const uint32_t g = (uint32_t)(msg >> 32);
        const int width = (int)(msg >> 24 & 0xFF), op = (int)(msg >> 22 & 3), sub = (int)(msg >> 16 & 63);
        const uint32_t lines = (uint32_t)(msg & 0xFFFF);
        release_children(e, tid, msg, width);
        if (op != OP_STOP && op != OP_NOP) member_work(e, tid, op, sub, lines, width);
        join_children(e, tid, g, width);
        e->sync[tid].done = g;
        if (op == OP_STOP) return;
    }
}

static void team_run(ChipEngine *e, int op, int sub, uint32_t lines, int width) {
    const uint32_t g = ++e->gen;
    const uint64_t msg = (uint64_t)g << 32 | (uint64_t)(uint32_t)width << 24 | (uint64_t)(uint32_t)op << 22 | (uint64_t)(uint32_t)sub << 16 | lines;
    release_children(e, 0, msg, width);
    if (op != OP_NOP) member_work(e, 0, op, sub, lines, width);
    join_children(e, 0, g, width);
    e->st[S_PHASES]++;
}

void chip_engine_stop(ChipEngine *e) {
    const uint32_t g = ++e->gen;
    const uint64_t msg = (uint64_t)g << 32 | (uint64_t)(uint32_t)e->nthreads << 24 | (uint64_t)OP_STOP << 22;
    release_children(e, 0, msg, e->nthreads);
    join_children(e, 0, g, e->nthreads);
}

uint64_t chip_engine_phase_ticks(ChipEngine *e, int rounds, int width) {
    const uint64_t t0 = ticks();
    for (int i = 0; i < rounds; i++) team_run(e, OP_NOP, 0, 0, width);
    return (ticks() - t0) / (uint64_t)rounds;
}

/* ------------------------------------------------------------------------ the residual and its norm */

/* The sum with every small dim at a given magnitude and the big dims as they are: one pass over all the lines, in order. */
static __m512 fold_bound(const ChipEngine *e, uint32_t small_bits) {
    uint32_t line[16] ALIGNED64;
    __m512 acc = _mm512_setzero_ps();
    uint32_t b = 0;
    for (uint32_t l = 0; l < CHIP_LINES; l++) {
        for (int j = 0; j < 16; j++) line[j] = small_bits;
        for (; b < e->big_n && (uint32_t)(e->big_dim[b] >> 4) == l; b++) line[e->big_dim[b] & 15] = e->big_bits[b];
        const __m512 v = _mm512_load_ps((const float *)line);
        acc = V_FMA(v, v, acc);
    }
    return v_tree16(acc);
}

/* A rounded sum of squares is monotonic in every term: fma and add round to nearest, so a larger input never gives a smaller result. So if the sum with every small
 * dim at M equals the sum with every small dim at 0, every x between them has that same sum, for any tensors: the big dims (the ballast here) swallow the rest whole.
 * Found once, from a row of the embedding; each token and each y then only have to stay within M, which takes a compare instead of a pass over x. */
static void prove_scale(ChipEngine *e, uint32_t i0, uint32_t i1) {
    const uint32_t *ev = (const uint32_t *)e->emb_val;
    e->m_bits = 0;
    e->big_n = 0;
    for (uint32_t i = i0; i < i1; i++)
        if ((ev[i] & 0x7FFFFFFFu) >= 0x4B800000u) {  /* 2^24 and up */
            if (e->big_n == 32) return;
            e->big_dim[e->big_n] = e->emb_col[i];
            e->big_bits[e->big_n++] = ev[i];
        }
    if (!e->big_n) return;
    float a[16] ALIGNED64, b[16] ALIGNED64;
    const __m512 low = fold_bound(e, 0);
    _mm512_store_ps(a, low);
    for (int k = 30; k >= 0; k -= 2) {
        const uint32_t m = (uint32_t)(127 + k) << 23;
        _mm512_store_ps(b, fold_bound(e, m));
        if (bits_of(a) == bits_of(b)) {
            e->m_bits = m;
            _mm512_store_ps(e->scale_fixed, v_rsqrt(_mm512_add_ps(_mm512_mul_ps(low, v_bits(e->h->over_h_bits)), v_bits(e->h->eps_bits))));
            return;
        }
    }
}

/* q_rms_scale(x): the sum takes the lines in ascending order, and a zero line adds exactly nothing. */
static void compute_scale(ChipEngine *e) {
    __m512 acc = _mm512_setzero_ps();
    for (int w = 0; w < CHIP_SETW; w++) {
        uint64_t bits = e->xnz[w];
        while (bits) {
            const int l = __builtin_ctzll(bits);
            bits &= bits - 1;
            const __m512 v = _mm512_load_ps(e->x + 16 * ((size_t)w * 64 + (size_t)l));
            acc = V_FMA(v, v, acc);
        }
    }
    const __m512 s = _mm512_add_ps(_mm512_mul_ps(v_tree16(acc), v_bits(e->h->over_h_bits)), v_bits(e->h->eps_bits));
    _mm512_store_ps(e->scale_line, v_rsqrt(s));
}

static inline void hn_line(ChipEngine *e, const float *norm, size_t l, __m512 scale) {
    const __m512 now = bf16_lanes(_mm512_mul_ps(_mm512_mul_ps(_mm512_load_ps(e->x + 16 * l), scale), _mm512_load_ps(norm + 16 * l)));
    float *dst = e->hn + 16 * l;
    if (differs(now, _mm512_load_ps(dst))) {
        _mm512_store_ps(dst, now);
        if (e->n_log == LOG_LINES) abort();
        _mm512_store_ps(e->logv[e->log_cur] + 16 * (size_t)e->n_log, now);
        e->logl[e->log_cur][e->n_log++] = (uint16_t)l;
        e->chg[l >> 6] |= 1ULL << (l & 63);
        e->st[S_HN_WRITES]++;
    }
}

/* hn for this sublayer, from the lines of x that moved. */
static void update_hn(ChipEngine *e, ChipSub *S, const uint64_t *moved) {
    uint64_t t0 = ticks();
    e->log_start[e->log_cur][e->sub_no] = e->n_log;
    /* the first tokens check the proof against the sum itself, on every sublayer */
    const int check = e->fast && e->token_no <= 4096;
    if (!e->fast || check) compute_scale(e);
    if (e->fast) {
        if (check && bits_of(e->scale_line) != bits_of(e->scale_fixed)) abort();
        _mm512_store_ps(e->scale_line, _mm512_load_ps(e->scale_fixed));
    }
    const uint32_t bits = bits_of(e->scale_line);
    const int all = bits != e->scale_bits;
    e->scale_bits = bits;
    uint64_t t1 = ticks();
    e->st[S_T_SCALE] += t1 - t0;
    const __m512 scale = _mm512_load_ps(e->scale_line);
    const float *norm = (const float *)(e->base + S->o_norm);
    memset(e->chg, 0, sizeof e->chg);
    for (int w = 0; w < CHIP_SETW; w++) {
        uint64_t b = moved[w] | S->norm_diff[w];
        if (all) b |= e->xnz[w];
        while (b) {
            const int l = __builtin_ctzll(b);
            b &= b - 1;
            hn_line(e, norm, (size_t)w * 64 + (size_t)l, scale);
        }
    }
    e->st[S_T_HN] += ticks() - t1;
}

/* What hn differs from one token ago at this sublayer, and who reads it. hn is one array that only moves by logged writes, so the same writes replayed one token later
 * keep a second array exactly one token behind it, across the token's boundary too: one array, not a snapshot per sublayer, and no first sublayer to treat apart. A line's
 * difference can change only where either array just moved; elsewhere it stays what it was. Returns how many lines of work there are, listed. */
static uint32_t find_dirty(ChipEngine *e, ChipSub *S) {
    const uint64_t t0 = ticks();
    const int s = e->sub_no++, last = e->log_cur ^ 1;
    uint64_t moved[CHIP_SETW];
    memset(moved, 0, sizeof moved);
    const float *lv = e->logv[last];
    const uint16_t *ll = e->logl[last];
    const uint32_t k1 = e->log_start[last][s + 1];
    for (uint32_t k = e->log_start[last][s]; k < k1; k++) {
        const uint32_t line = ll[k];
        _mm512_store_ps(e->hp + 16 * (size_t)line, _mm512_load_ps(lv + 16 * (size_t)k));
        moved[line >> 6] |= 1ULL << (line & 63);
    }
    for (int w = 0; w < CHIP_SETW; w++) {
        uint64_t b = moved[w] | e->chg[w];
        while (b) {
            const int l = __builtin_ctzll(b);
            b &= b - 1;
            const size_t line = (size_t)w * 64 + (size_t)l;
            const uint32_t m = differs(_mm512_load_ps(e->hn + 16 * line), _mm512_load_ps(e->hp + 16 * line));
            e->dmask[line] = (uint16_t)m;
            if (m) e->diff[w] |= 1ULL << l;
            else e->diff[w] &= ~(1ULL << l);
            e->st[S_SNAP_DIFF]++;
        }
    }
    /* a byte per line of work: a mark is one store, no read of what is there. The fan tables are cold, and most differing dims have no reader here: a mask says so first. */
    uint8_t *dirty = e->base + S->o_dirty;
    uint32_t *list = (uint32_t *)(e->base + S->o_list);
    const uint32_t *fs = (const uint32_t *)(e->base + S->o_fan_start), *fi = (const uint32_t *)(e->base + S->o_fan_item);
    const uint16_t *reads = (const uint16_t *)(e->base + S->o_reads);
    const uint32_t words = (S->n_work + 7) / 8;
    for (int w = 0; w < CHIP_SETW; w++) {
        uint64_t b = e->diff[w];
        while (b) {
            const int l = __builtin_ctzll(b);
            b &= b - 1;
            const size_t line = (size_t)w * 64 + (size_t)l;
            uint32_t m = (uint32_t)e->dmask[line] & reads[line];
            while (m) {
                const int lane = __builtin_ctz(m);
                m &= m - 1;
                const size_t dim = 16 * line + (size_t)lane;
                const uint32_t f1 = fs[dim + 1];
                e->st[S_MARKS] += f1 - fs[dim];
                for (uint32_t i = fs[dim]; i < f1; i++) dirty[fi[i]] = 1;
            }
        }
    }
    uint32_t n = 0;
    if (!e->memo) {
        for (uint32_t i = 0; i < S->n_work; i++) list[n++] = i;
        memset(dirty, 0, (size_t)words * 8);
    } else {
        uint64_t *dw = (uint64_t *)dirty;
        for (uint32_t w = 0; w < words; w++) {
            uint64_t b = dw[w];
            if (!b) continue;
            dw[w] = 0;
            while (b) {
                const int l = __builtin_ctzll(b) >> 3;
                b &= ~(0xFFULL << (8 * l));
                list[n++] = w * 8 + (uint32_t)l;
            }
        }
    }
    e->st[S_T_SNAP] += ticks() - t0;
    return n;
}

/* Marks y's rows that read a changed input dim. */
static inline void y_mark(const ChipEngine *e, const ChipSub *S, size_t dim) {
    const uint32_t *fs = (const uint32_t *)(e->base + S->o_yfan_start), *fi = (const uint32_t *)(e->base + S->o_yfan_item);
    uint8_t *d = e->base + S->o_ydirty;
    const uint32_t f1 = fs[dim + 1];
    for (uint32_t i = fs[dim]; i < f1; i++) d[fi[i]] = 1;
}

/* y's rows whose input changed, each from its columns in ascending order as the definition has it, then every nonzero line into ytmp. A row is few entries, and a
 * line of 16 rows can be hundreds of slices. */
static void y_compute(ChipEngine *e, ChipSub *S) {
    const uint64_t t0 = ticks();
    uint8_t *dirty = e->base + S->o_ydirty;
    uint64_t *dw = (uint64_t *)dirty;
    const uint32_t *dims = (const uint32_t *)(e->base + S->o_yrow_dim), *st = (const uint32_t *)(e->base + S->o_yrow_start), *cols = (const uint32_t *)(e->base + S->o_yrow_col);
    const float *wv = (const float *)(e->base + S->o_yrow_w), *in = (const float *)(e->base + S->o_input);
    const uint32_t *inb = (const uint32_t *)in;
    float *yv = (float *)(e->base + S->o_yv);
    const uint32_t words = (S->n_y + 7) / 8;
    uint64_t touched[CHIP_SETW];
    memset(touched, 0, sizeof touched);
    float tmp[16] ALIGNED64;
    for (uint32_t w = 0; w < words; w++) {
        uint64_t b = e->memo ? dw[w] : ~0ULL;
        if (!b) continue;
        dw[w] = 0;
        while (b) {
            const int l = __builtin_ctzll(b) >> 3;
            b &= ~(0xFFULL << (8 * l));
            const uint32_t r = w * 8 + (uint32_t)l;
            if (r >= S->n_y) break;
            __m512 acc = _mm512_setzero_ps();
            const uint32_t i1 = st[r + 1];
            for (uint32_t i = st[r]; i < i1; i++) {
                const uint32_t c = cols[i];
                if (inb[c] << 1) acc = V_FMA(v_bc(&wv[i]), v_bc(&in[c]), acc);
            }
            _mm512_store_ps(tmp, bf16_lanes(acc));
            const uint32_t d = dims[r];
            ((uint32_t *)yv)[d] = *(const uint32_t *)tmp;
            touched[d >> 10] |= 1ULL << (d >> 4 & 63);
            e->st[S_LINES_Y]++;
        }
    }
    for (int w = 0; w < CHIP_SETW; w++) {
        uint64_t b = touched[w];
        while (b) {
            const int l = __builtin_ctzll(b);
            b &= b - 1;
            if (nonzero(_mm512_load_ps(yv + 16 * ((size_t)w * 64 + (size_t)l)))) S->ynz[w] |= 1ULL << l;
            else S->ynz[w] &= ~(1ULL << l);
        }
        b = S->ynz[w];
        e->yset[w] = b;
        while (b) {
            const int l = __builtin_ctzll(b);
            b &= b - 1;
            const size_t line = (size_t)w * 64 + (size_t)l;
            _mm512_store_ps(e->ytmp + 16 * line, _mm512_load_ps(yv + 16 * line));
        }
    }
    e->st[S_T_Y] += ticks() - t0;
}

/* x += y, ytmp back to zeros; x's lines grow. While every small dim of x stays within the bound, the scale stays what prove_scale showed. */
static void y_apply(ChipEngine *e) {
    const uint64_t t0 = ticks();
    const __m512 zero = _mm512_setzero_ps(), bound = v_bits(e->m_bits);
    const __m512i mag = _mm512_set1_epi32(0x7FFFFFFF);
    for (int w = 0; w < CHIP_SETW; w++) {
        uint64_t b = e->yset[w];
        e->xnz[w] |= b;
        while (b) {
            const int l = __builtin_ctzll(b);
            b &= b - 1;
            const size_t line = (size_t)w * 64 + (size_t)l;
            const __m512 y = _mm512_load_ps(e->ytmp + 16 * line), nx = _mm512_add_ps(_mm512_load_ps(e->x + 16 * line), y);
            _mm512_store_ps(e->x + 16 * line, nx);
            _mm512_store_ps(e->ytmp + 16 * line, zero);
            if (e->fast && _mm512_mask_cmp_ps_mask((__mmask16)nonzero(y), _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(nx), mag)), bound, _CMP_NLE_US)) e->fast = 0;
        }
    }
    e->st[S_T_APPLY] += ticks() - t0;
}

static inline void y_add(ChipEngine *e, uint32_t d, const float *value) {
    float *line = e->ytmp + (d & ~15u);
    const __m512 v = _mm512_load_ps(line);
    _mm512_store_ps(line, _mm512_mask_add_ps(v, (__mmask16)(1u << (d & 15)), v, v_bc(value)));
    e->yset[d >> 10] |= 1ULL << (d >> 4 & 63);
}

/* --------------------------------------------------------------------------------- the attention heads */

static inline void conv_step(ChipEngine *e, ChipConv *C, uint32_t t) {
    e->st[S_CONV]++;
    const __m512 cur = _mm512_load_ps(C->cur);
    __m512 acc = V_FMA(_mm512_load_ps(C->w[0]), _mm512_load_ps(C->win[e->r0]), _mm512_setzero_ps());
    acc = V_FMA(_mm512_load_ps(C->w[1]), _mm512_load_ps(C->win[e->r1]), acc);
    acc = V_FMA(_mm512_load_ps(C->w[2]), _mm512_load_ps(C->win[e->r2]), acc);
    acc = V_FMA(_mm512_load_ps(C->w[3]), cur, acc);
    _mm512_store_ps(C->win[e->r0], cur);
    if (differs(acc, _mm512_load_ps(C->acc))) {
        _mm512_store_ps(C->acc, acc);
        const __m512 now = silu0(bf16_lanes(acc));
        if (differs(now, _mm512_load_ps(C->conv))) {
            _mm512_store_ps(C->conv, now);
            C->conv_at = t;
        }
    }
}

/* q and k of a key head, when a line of theirs has moved. */
static void key_step(ChipEngine *e, ChipKey *K, const ChipConv *conv, uint32_t t) {
    uint32_t any_k = 0;
    __m512 sq = _mm512_setzero_ps(), sk = _mm512_setzero_ps();
    for (int j = 0; j < CHIP_ST_LINES; j++) {
        if (K->q[j] >= 0) {
            const __m512 v = _mm512_load_ps(conv[K->q[j]].conv);
            sq = V_FMA(v, v, sq);
        }
        if (K->k[j] >= 0) {
            const __m512 v = _mm512_load_ps(conv[K->k[j]].conv);
            sk = V_FMA(v, v, sk);
        }
    }
    const __m512 e2 = v_bits(e->h->e2_bits);
    sq = v_tree16(sq);
    sk = v_tree16(sk);
    /* sq > e2 ? sq : e2 */
    const __m512 rq = v_rsqrt(v_blend(_mm512_cmplt_ps_mask(e2, sq), sq, e2)), rk = v_rsqrt(v_blend(_mm512_cmplt_ps_mask(e2, sk), sk, e2));
    const __m512 qs = v_bits(e->h->q_scale_bits), zero = _mm512_setzero_ps();
    for (int j = 0; j < CHIP_ST_LINES; j++) {
        const __m512 q = K->q[j] >= 0 ? _mm512_mul_ps(_mm512_mul_ps(_mm512_load_ps(conv[K->q[j]].conv), rq), qs) : zero;
        const __m512 k = K->k[j] >= 0 ? _mm512_mul_ps(_mm512_load_ps(conv[K->k[j]].conv), rk) : zero;
        if (differs(q, _mm512_load_ps(K->qv + 16 * j))) {
            _mm512_store_ps(K->qv + 16 * j, q);
            K->q_at = t;
        }
        _mm512_store_ps(K->kv + 16 * j, k);
        K->qmask[j] = (uint16_t)nonzero(q);
        K->kmask[j] = (uint16_t)nonzero(k);
        any_k |= K->kmask[j];
    }
    K->kany = any_k;
}

static void head_step(ChipEngine *e, ChipLayer *L, const ChipKey *K, ChipHead *H, const ChipConv *conv, uint32_t t) {
    const uint32_t alpha = bits_of(H->alpha_v), beta = bits_of(H->beta_v);
    float tmp[16] ALIGNED64;
    if (H->fresh || alpha != H->alpha_last) {
        const __m512 sp = v_softplus(_mm512_add_ps(v_bc(H->alpha_v), v_bits(H->dt_bits)));
        _mm512_store_ps(tmp, v_exp(_mm512_mul_ps(sp, v_bits(H->a_bits))));
        H->e_bits = bits_of(tmp);
        H->alpha_last = alpha;
    }
    if (H->fresh || beta != H->beta_last) {
        _mm512_store_ps(tmp, sigmoid0(v_bc(H->beta_v)));
        H->b_bits = bits_of(tmp);
        H->beta_last = beta;
    }
    H->fresh = 0;
    float *S = H->S;
    int state_moved = 0;
    if (H->e_bits != e->h->one_bits) {
        /* a zero row stays zero */
        const __m512 dec = v_bits(H->e_bits);
        if (H->rows_nz[0] | H->rows_nz[1]) state_moved = 1;
        for (int w = 0; w < 2; w++) {
            uint64_t b = H->rows_nz[w];
            while (b) {
                const int r = __builtin_ctzll(b);
                b &= b - 1;
                float *row = S + (size_t)(w * 64 + r) * CHIP_ST;
                for (int j = 0; j < CHIP_ST_LINES; j++) _mm512_store_ps(row + 16 * j, _mm512_mul_ps(_mm512_load_ps(row + 16 * j), dec));
            }
            if ((H->e_bits << 1) == 0) H->rows_nz[w] = 0;
        }
        e->st[S_E_PASS]++;
    }
    /* the key's write */
    if (K->kany) {
        __m512 kv[CHIP_ST_LINES];
        for (int j = 0; j < CHIP_ST_LINES; j++) kv[j] = _mm512_setzero_ps();
        for (int lj = 0; lj < CHIP_ST_LINES; lj++) {
            uint32_t m = K->kmask[lj];
            while (m) {
                const int lane = __builtin_ctz(m);
                m &= m - 1;
                const int i = lj * 16 + lane;
                const __m512 ki = v_bc(&K->kv[i]);
                const float *row = S + (size_t)i * CHIP_ST;
                for (int j = 0; j < CHIP_ST_LINES; j++) kv[j] = V_FMA(_mm512_load_ps(row + 16 * j), ki, kv[j]);
            }
        }
        const __m512 b = v_bits(H->b_bits);
        for (int j = 0; j < CHIP_ST_LINES; j++) {
            const __m512 v = H->v[j] >= 0 ? _mm512_load_ps(conv[H->v[j]].conv) : _mm512_setzero_ps();
            kv[j] = _mm512_mul_ps(_mm512_sub_ps(v, kv[j]), b);
        }
        for (int lj = 0; lj < CHIP_ST_LINES; lj++) {
            uint32_t m = K->kmask[lj];
            while (m) {
                const int lane = __builtin_ctz(m);
                m &= m - 1;
                const int i = lj * 16 + lane;
                const __m512 ki = v_bc(&K->kv[i]);
                float *row = S + (size_t)i * CHIP_ST;
                for (int j = 0; j < CHIP_ST_LINES; j++) _mm512_store_ps(row + 16 * j, V_FMA(ki, kv[j], _mm512_load_ps(row + 16 * j)));
                H->rows_nz[i >> 6] |= 1ULL << (i & 63);
            }
        }
        state_moved = 1;
        e->st[S_KEYED]++;
    }
    /* the read */
    int o_new = 0;
    if (state_moved || K->q_at == t || !e->memo) {
        __m512 o[CHIP_ST_LINES];
        for (int j = 0; j < CHIP_ST_LINES; j++) o[j] = _mm512_setzero_ps();
        for (int lj = 0; lj < CHIP_ST_LINES; lj++) {
            uint32_t m = K->qmask[lj];
            while (m) {
                const int lane = __builtin_ctz(m);
                m &= m - 1;
                const int i = lj * 16 + lane;
                const __m512 qi = v_bc(&K->qv[i]);
                const float *row = S + (size_t)i * CHIP_ST;
                for (int j = 0; j < CHIP_ST_LINES; j++) o[j] = V_FMA(_mm512_load_ps(row + 16 * j), qi, o[j]);
            }
        }
        for (int j = 0; j < CHIP_ST_LINES; j++)
            if (differs(o[j], _mm512_load_ps(H->o_v + 16 * j))) {
                _mm512_store_ps(H->o_v + 16 * j, o[j]);
                o_new = 1;
            }
        e->st[S_O]++;
    }
    const int z_new = H->touched_at == t;
    if (!(o_new || z_new || !e->memo)) return;
    /* the mix: its norm, and its gate */
    e->st[S_MIX]++;
    __m512 acc = _mm512_setzero_ps();
    for (int j = 0; j < CHIP_ST_LINES; j++) {
        const __m512 v = _mm512_load_ps(H->o_v + 16 * j);
        acc = V_FMA(v, v, acc);
    }
    const __m512 sc = v_rsqrt(_mm512_add_ps(_mm512_mul_ps(v_tree16(acc), v_bits(e->h->over_st_bits)), v_bits(e->h->eps_bits)));
    const float *norm = (const float *)(e->base + L->o_ssm_norm);
    float *mix = (float *)(e->base + L->a.o_input) + 16 * (size_t)H->mix_line;
    for (int j = 0; j < CHIP_ST_LINES; j++) {
        const __m512 o = _mm512_load_ps(H->o_v + 16 * j);
        __m512 m = _mm512_setzero_ps();
        if (nonzero(o)) m = bf16_lanes(_mm512_mul_ps(bf16_lanes(_mm512_mul_ps(_mm512_mul_ps(o, sc), _mm512_load_ps(norm + 16 * j))), silu0(_mm512_load_ps(H->z_v + 16 * j))));
        uint32_t ch = differs(m, _mm512_load_ps(mix + 16 * j));
        if (!ch) continue;
        _mm512_store_ps(mix + 16 * j, m);
        while (ch) {
            const int lane = __builtin_ctz(ch);
            ch &= ch - 1;
            y_mark(e, &L->a, ((size_t)H->mix_line + (size_t)j) * 16 + (size_t)lane);
        }
    }
}

/* ------------------------------------------------------------------------------------- the memories */

static inline uint64_t bits_up(const float *v, int n) {
    uint64_t out = 0;
    for (int line = 0; line * 16 < n; line++) out |= (uint64_t)nonzero(_mm512_load_ps(v + 16 * line)) << (16 * line);
    return n >= 64 ? out : out & ((1ULL << n) - 1);
}

static void bank_step(ChipEngine *e, ChipBank *B) {
    const uint64_t ra = bits_up(B->vra, B->nra);
    uint8_t *mem = e->base + B->o_mem;
    if (B->kind != 0) {
        /* the data's taps, whether or not it is written: the ring goes on */
        float vd[16 * CHIP_BANK_LINES] ALIGNED64;
        for (int line = 0; line * 16 < B->nd; line++) {
            const size_t l = 16 * (size_t)line;
            const __m512 cur = _mm512_load_ps(B->cur + l);
            __m512 s = _mm512_add_ps(_mm512_mul_ps(_mm512_load_ps(B->taps[0] + l), _mm512_load_ps(B->win[e->r0] + l)), _mm512_mul_ps(_mm512_load_ps(B->taps[1] + l), _mm512_load_ps(B->win[e->r1] + l)));
            s = _mm512_add_ps(s, _mm512_mul_ps(_mm512_load_ps(B->taps[2] + l), _mm512_load_ps(B->win[e->r2] + l)));
            s = _mm512_add_ps(s, _mm512_mul_ps(_mm512_load_ps(B->taps[3] + l), cur));
            _mm512_store_ps(vd + l, s);
            _mm512_store_ps(B->win[e->r0] + l, cur);
        }
        if (bits_of(B->we_v) << 1) {
            const uint64_t wa = bits_up(B->vwa, B->nwa);
            if (B->kind == 1) {
                ((uint32_t *)mem)[wa] = (uint32_t)bits_up(vd, 32);
            } else if (B->kind == 3) {
                mem[wa] = (uint8_t)bits_up(vd, 8);
            } else {
                const uint64_t w = bits_up(vd, 64);
                for (int k = 0; k < 4; k++) ((uint16_t *)mem)[wa * 4 + (uint64_t)k] = (uint16_t)(w >> (16 * k));
            }
        }
    }
    if (B->kind == 0) {
        const uint8_t *t = mem + (ra >> 4) * 576;
        const uint32_t r = (uint32_t)ra & 15;
        for (int i = 0; i < 32; i++) {
            const uint32_t v = t[16 * i + r] ^ 0x80u;
            if (v) y_add(e, B->out[i], &e->byte_f[v]);
        }
        for (int i = 0; i < 4; i++) {
            const uint32_t v = t[512 + 4 * r + (uint32_t)i];
            if (v) y_add(e, B->out[32 + i], &e->byte_f[v]);
        }
        return;
    }
    uint32_t word;
    int n;
    if (B->kind == 3) {
        word = mem[ra];
        n = 8;
    } else if (B->kind == 1) {
        word = ((const uint32_t *)mem)[ra];
        n = 32;
    } else {
        word = ((const uint16_t *)mem)[ra];
        n = 16;
    }
    static const uint32_t one_bits = 0x3F800000u;
    for (int i = 0; i < n; i++)
        if (word >> i & 1) y_add(e, B->out[i], (const float *)&one_bits);
}

/* ------------------------------------------------------------------------------------------ a layer */

static inline void run_lines(ChipEngine *e, int op, int sub, uint32_t n) {
    if (n >= (uint32_t)e->par_min && e->width > 1) {
        e->st[S_PAR_LINES] += n;
        team_run(e, op, sub, n, e->width);
    } else {
        member_work(e, 0, op, sub, n, 1);
    }
}

static void attention(ChipEngine *e, ChipLayer *L, int il, const uint64_t *moved) {
    ChipSub *S = &L->a;
    const uint32_t t = e->token_no;
    update_hn(e, S, moved);
    const uint32_t n = find_dirty(e, S);
    e->st[S_LINES_A] += n;
    uint64_t t0 = ticks();
    ChipConv *conv = (ChipConv *)(e->base + L->o_conv);
    ChipKey *keys = (ChipKey *)(e->base + L->o_keys);
    ChipHead *heads = (ChipHead *)(e->base + L->o_heads);
    const ChipSell *work = (const ChipSell *)(e->base + S->o_work);
    uint16_t *act = (uint16_t *)(e->base + L->o_act);
    uint8_t *in = e->base + L->o_in;
    if (n) {
        S->token_now = t;
        run_lines(e, OP_HN_FORMS, il * 2, n);
        /* who has to look again: a conv line for the next four tokens, a head for this one */
        const uint32_t *list = (const uint32_t *)(e->base + S->o_list);
        for (uint32_t k = 0; k < n; k++) {
            const ChipSell *w = &work[list[k]];
            if (w->changed_at != t) continue;
            const uint32_t kind = w->who >> 24, idx = w->who & 0xFFFFFF;
            if (kind == 1 && !in[idx]) {
                in[idx] = 1;
                act[L->n_act++] = (uint16_t)idx;
            } else if (kind == 2) {
                heads[idx].touched_at = t;
            }
        }
    }
    uint64_t t1 = ticks();
    e->st[S_T_FORMS] += t1 - t0;
    if (e->memo) {
        uint32_t keep = 0;
        for (uint32_t i = 0; i < L->n_act; i++) {
            const uint32_t idx = act[i];
            ChipConv *C = &conv[idx];
            const ChipSell *W = &work[C->work];
            conv_step(e, C, t);
            if (C->conv_at == t) keys[C->key].touched_at = t;
            if (t - W->changed_at < 3) act[keep++] = (uint16_t)idx;
            else in[idx] = 0;
        }
        L->n_act = keep;
    } else {
        for (uint32_t i = 0; i < L->n_conv; i++) conv_step(e, &conv[i], t);
    }
    e->st[S_T_CONV] += ticks() - t1;
    const uint32_t one = e->h->one_bits;
    for (uint32_t k = 0; k < L->n_keys; k++) {
        ChipKey *K = &keys[k];
        if (!e->memo || K->touched_at == t) key_step(e, K, conv, t);
        for (uint32_t h = 0; h < K->nheads; h++) {
            ChipHead *H = &heads[K->head0 + h];
            /* nothing it reads moved, it decays nothing, and no key writes: as it was */
            if (e->memo && !H->fresh && H->touched_at != t && H->e_bits == one && !K->kany && K->q_at != t) continue;
            head_step(e, L, K, H, conv, t);
        }
    }
    uint64_t t2 = ticks();
    e->st[S_T_HEADS] += t2 - t1;
    y_compute(e, S);
    t2 = ticks();
    ChipBank *banks = (ChipBank *)(e->base + L->o_banks);
    for (uint32_t b = 0; b < L->n_banks; b++) bank_step(e, &banks[b]);
    e->st[S_T_BANKS] += ticks() - t2;
    y_apply(e);
}

static void ffn(ChipEngine *e, ChipLayer *L, int il) {
    ChipSub *S = &L->f;
    uint64_t moved[CHIP_SETW];
    memcpy(moved, e->yset, sizeof moved);
    update_hn(e, S, moved);
    uint32_t n = find_dirty(e, S);
    uint64_t t0 = ticks();
    /* the expert's gate: if it moved, every unit may */
    const uint32_t *gi = (const uint32_t *)(e->base + L->o_gi_idx);
    const float *gw = (const float *)(e->base + L->o_gi_w);
    __m512 dot = _mm512_setzero_ps();
    for (uint32_t i = 0; i < L->n_gi; i++) dot = V_FMA(v_bc(&gw[i]), v_bc(&e->hn[gi[i]]), dot);
    float tmp[16] ALIGNED64;
    _mm512_store_ps(tmp, dot);
    uint32_t *list = (uint32_t *)(e->base + S->o_list);
    if (L->gi_fresh || bits_of(tmp) != L->gi_last) {
        const __m512 was = _mm512_load_ps(L->gate_s), now = sigmoid0(dot);
        _mm512_store_ps(L->gate_s, now);
        if (L->gi_fresh || differs(was, now)) {
            n = S->n_work;
            for (uint32_t i = 0; i < n; i++) list[i] = i;
        }
        L->gi_last = bits_of(tmp);
        L->gi_fresh = 0;
    }
    e->st[S_LINES_F] += n;
    if (n) {
        if (n >= (uint32_t)e->par_min && e->width > 1) {
            e->st[S_PAR_LINES] += n;
            team_run(e, OP_UNITS, il * 2 + 1, n, e->width);
        } else {
            member_work(e, 0, OP_UNITS, il * 2 + 1, n, 1);
        }
        const ChipUnit *work = (const ChipUnit *)(e->base + S->o_work);
        for (int w = 0; w < 3; w++) {
            uint64_t bits = e->chg_mask[w];
            if (!bits) continue;
            e->chg_mask[w] = 0;
            while (bits) {
                const int tid = w * 64 + __builtin_ctzll(bits);
                bits &= bits - 1;
                const Member *M = &e->mem[tid];
                for (uint32_t i = 0; i < M->nchg; i++) {
                    const uint32_t id = M->chg[i] & 0xFFFF;
                    uint32_t m = M->chg[i] >> 16;
                    const size_t line = work[id].line;
                    e->st[S_CHG_UNITS]++;
                    while (m) {
                        const int lane = __builtin_ctz(m);
                        m &= m - 1;
                        y_mark(e, S, line * 16 + (size_t)lane);
                    }
                }
            }
        }
    }
    e->st[S_T_UNITS] += ticks() - t0;
    y_compute(e, S);
    y_apply(e);
}

/* ------------------------------------------------------------------------------------- the output head */

static inline int gt0(uint32_t u) { return (int32_t)u > 0 && u <= 0x7F800000u; }

static inline uint64_t mix64(uint64_t k) {
    k ^= k >> 33;
    k *= 0xFF51AFD7ED558CCDULL;
    k ^= k >> 33;
    return k;
}

/* the table of rows by the signs of their values on the dims the output has reached */
static void build_table(ChipEngine *e) {
    const uint64_t t0 = ticks();
    memset(e->table, 0, sizeof(Entry) << TABLE_BITS);
    for (int k = 0; k < e->n_out; k++) e->out_max[k] = 0;
    for (uint32_t t = 0; t < e->h->vocab; t++) {
        uint32_t vals[MAX_OUT_DIMS];
        memset(vals, 0, (size_t)e->n_out * 4);
        for (uint32_t i = e->emb_start[t]; i < e->emb_start[t + 1]; i++) {
            const int d = e->emb_col[i];
            for (int k = 0; k < e->n_out; k++)
                if (e->out_dims[k] == d) vals[k] = ((const uint32_t *)e->emb_val)[i];
        }
        uint64_t key = 0;
        for (int k = 0; k < e->n_out; k++) {
            key = key << 1 | (uint64_t)gt0(vals[k]);
            const uint32_t mag = vals[k] & 0x7FFFFFFFu;
            if (mag > e->out_max[k]) e->out_max[k] = mag;
        }
        uint64_t p = mix64(key) >> (64 - TABLE_BITS);
        while (e->table[p].used && e->table[p].key != key) p = (p + 1) & ((1ULL << TABLE_BITS) - 1);
        if (!e->table[p].used) {
            e->table[p].used = 1;
            e->table[p].key = key;
            e->table[p].token = t;
        }
    }
    e->table_builds++;
    e->st[S_TABLE] += ticks() - t0;
}

/* The argmax of logit = row . x, tied to the embedding. A row with the largest value of the right sign on every dim x is nonzero on reaches the sum of the
 * largest terms, which no row can pass. As bits: nothing here needs a float. Should no row do that, the rows are scanned on the x87 as long doubles. */
static uint32_t next_token(ChipEngine *e) {
    const uint32_t *xb = (const uint32_t *)e->x;
    int dims[MAX_OUT_DIMS], n = 0;
    for (int w = 0; w < CHIP_SETW; w++) {
        uint64_t b = e->xnz[w];
        while (b) {
            const int l = __builtin_ctzll(b);
            b &= b - 1;
            const size_t line = (size_t)w * 64 + (size_t)l;
            uint32_t m = nonzero(_mm512_load_ps(e->x + 16 * line));
            while (m) {
                const int lane = __builtin_ctz(m);
                m &= m - 1;
                const int d = (int)(line * 16) + lane;
                if (e->row_used[d]) {
                    if (n == MAX_OUT_DIMS) abort();
                    dims[n++] = d;
                } else if (!e->stray_seen[d]) {
                    e->stray_seen[d] = 1;
                    if (e->n_strays < MAX_STRAYS) {
                        e->stray_dim[e->n_strays] = (uint32_t)d;
                        e->stray_bits[e->n_strays++] = xb[d];
                    }
                }
            }
        }
    }
    /* a new dim in the output: the union grows and the table follows */
    int grew = 0;
    for (int i = 0; i < n; i++) {
        int k = 0;
        while (k < e->n_out && e->out_dims[k] != dims[i]) k++;
        if (k == e->n_out) {
            if (e->n_out == MAX_OUT_DIMS) abort();
            int at = e->n_out++;
            while (at > 0 && e->out_dims[at - 1] > dims[i]) {
                e->out_dims[at] = e->out_dims[at - 1];
                at--;
            }
            e->out_dims[at] = dims[i];
            grew = 1;
        }
    }
    if (grew) build_table(e);
    uint64_t key = 0;
    for (int k = 0; k < e->n_out; k++) key = key << 1 | (uint64_t)gt0(xb[e->out_dims[k]]);
    uint64_t p = mix64(key) >> (64 - TABLE_BITS);
    while (e->table[p].used && e->table[p].key != key) p = (p + 1) & ((1ULL << TABLE_BITS) - 1);
    if (e->table[p].used) {
        const uint32_t tok = e->table[p].token;
        int good = 1;
        /* one pass over the row, the output's dims and the row's both ascending */
        uint32_t i = e->emb_start[tok];
        const uint32_t end = e->emb_start[tok + 1];
        const uint32_t *ev = (const uint32_t *)e->emb_val;
        for (int k = 0; k < e->n_out && good; k++) {
            const int d = e->out_dims[k];
            while (i < end && e->emb_col[i] < d) i++;
            const uint32_t x = xb[d];
            if (!(x << 1)) continue;
            const uint32_t v = i < end && e->emb_col[i] == d ? ev[i] : 0;
            good = v == ((x & 0x80000000u) | e->out_max[k]);
        }
        if (good) return tok;
    }
    uint32_t best = 0;
    long double top = -1e300L;
    const float *evf = e->emb_val;
    for (uint32_t t = 0; t < e->h->vocab; t++) {
        long double dot = 0;
        for (uint32_t i = e->emb_start[t]; i < e->emb_start[t + 1]; i++) {
            const int d = e->emb_col[i];
            if (xb[d] << 1) dot += (long double)evf[i] * (long double)e->x[d];
        }
        if (dot > top) {
            top = dot;
            best = t;
        }
    }
    e->full_scans++;
    e->st[S_FULL_SCANS]++;
    return best;
}

/* --------------------------------------------------------------------------------------------- a token */

uint32_t chip_engine_step(ChipEngine *e, uint32_t token) {
    const uint64_t t_start = ticks();
    const uint32_t t = ++e->token_no;
    e->r0 = t % 3;
    e->r1 = (t + 1) % 3;
    e->r2 = (t + 2) % 3;
    /* last token's x and hn go, this token's row comes in */
    uint64_t moved[CHIP_SETW], row_lines[CHIP_SETW];
    memset(row_lines, 0, sizeof row_lines);
    const __m512 zero = _mm512_setzero_ps();
    for (int w = 0; w < CHIP_SETW; w++) {
        uint64_t b = e->xnz[w];
        moved[w] = b;
        e->xnz[w] = 0;
        while (b) {
            const int l = __builtin_ctzll(b);
            b &= b - 1;
            _mm512_store_ps(e->x + 16 * ((size_t)w * 64 + (size_t)l), zero);
        }
    }
    const uint32_t i0 = e->emb_start[token], i1 = e->emb_start[token + 1];
    const uint32_t *ev = (const uint32_t *)e->emb_val;
    for (uint32_t i = i0; i < i1; i++) {
        const uint32_t d = e->emb_col[i], line = d >> 4;
        ((uint32_t *)e->x)[d] = ev[i];
        row_lines[line >> 6] |= 1ULL << (line & 63);
    }
    for (int w = 0; w < CHIP_SETW; w++) {
        moved[w] |= row_lines[w];
        e->xnz[w] = row_lines[w];
    }
    if (!e->proved) {
        prove_scale(e, i0, i1);
        e->proved = 1;
    }
    /* the scale is the proved one if this row has the same big dims and nothing else above the bound */
    int fast = e->m_bits != 0;
    uint32_t nbig = 0;
    for (uint32_t i = i0; i < i1 && fast; i++)
        if ((ev[i] & 0x7FFFFFFFu) > e->m_bits) {
            if (nbig < e->big_n && e->big_dim[nbig] == e->emb_col[i] && e->big_bits[nbig] == ev[i]) nbig++;
            else fast = 0;
        }
    e->fast = fast && nbig == e->big_n;
    e->st[S_FAST] += (uint64_t)e->fast;
    e->st[S_T_EMBED] += ticks() - t_start;
    const int layers = (int)e->h->layers;
    for (int il = 0; il < layers; il++) {
        ChipLayer *L = &e->layers[il];
        attention(e, L, il, il == 0 ? moved : e->yset);
        ffn(e, L, il);
    }
    const uint64_t t_o = ticks();
    const uint32_t next = next_token(e);
    e->log_start[e->log_cur][e->sub_no] = e->n_log;
    e->log_cur ^= 1;
    e->n_log = 0;
    e->sub_no = 0;
    e->st[S_T_OUT] += ticks() - t_o;
    e->st[S_TOKENS]++;
    e->st[S_T_TOKEN] += ticks() - t_start;
    return next;
}

/* ----------------------------------------------------------------------------------------------- setup */

/* A product that lands in the denormals, and back up: 2^-99 x 2^-35 = 2^-134, then x 2^35 = 2^-99. */
static int denormals_kept(uint32_t *seen) {
    const uint32_t a = 0x0E000000u, b = 0x2E000000u, c = 0x51000000u;  /* 2^-99, 2^-35, 2^35 */
    float o[16] ALIGNED64;
    _mm512_store_ps(o, _mm512_mul_ps(_mm512_mul_ps(v_bits(a), v_bits(b)), v_bits(c)));
    *seen = bits_of(o);
    /* and a compare must see it as nonzero */
    const __m512 tiny = _mm512_mul_ps(v_bits(a), v_bits(b));
    return *seen == a && nonzero(tiny) != 0;
}

int chip_vpu_setup(uint32_t *seen) {
    if (denormals_kept(seen)) return 1;
    /* the VPU's control word: bit 15 flush to zero off, bit 6 denormals used */
    uint32_t cw;
    __asm__ volatile("stmxcsr %0" : "=m"(cw));
    cw &= ~(1u << 15);
    cw &= ~(1u << 6);
    __asm__ volatile("ldmxcsr %0" : : "m"(cw));
    return denormals_kept(seen);
}

uint64_t chip_engine_bytes(int nthreads) {
    const uint64_t chg = (uint64_t)CHIP_HIDDEN * 4 + 64;
    return sizeof(ChipEngine) + 4096 + sizeof(Sync) * (uint64_t)nthreads + 64 + sizeof(Member) * (uint64_t)nthreads + 64 + chg * (uint64_t)nthreads + 2 * ((uint64_t)LOG_LINES * 66 + 64) + 65536 + (sizeof(Entry) << TABLE_BITS) + 128;
}

ChipEngine *chip_engine_new(void *image, void *work, int nthreads, int width, int par_min, int memo) {
    uint8_t *base = image;
    const ChipHeader *h = image;
    if (h->magic != CHIP_IMAGE_MAGIC) return NULL;
    uint8_t *w = (uint8_t *)(((uintptr_t)work + 63) & ~(uintptr_t)63);
    ChipEngine *e = (ChipEngine *)w;
    w += (sizeof(ChipEngine) + 63) & ~(size_t)63;
    memset(e, 0, sizeof *e);
    e->base = base;
    e->h = h;
    e->layers = (ChipLayer *)(base + h->o_layers);
    e->col = (const uint32_t *)(base + h->o_col);
    e->woff = (const uint32_t *)(base + h->o_woff);
    e->wtab = (const float *)(base + h->o_wtab);
    e->emb_start = (const uint32_t *)(base + h->o_emb_start);
    e->emb_col = (const uint16_t *)(base + h->o_emb_col);
    e->emb_val = (const float *)(base + h->o_emb_val);
    e->byte_f = (const float *)(base + h->o_byte_f);
    e->row_used = base + h->o_row_used;
    e->x = (float *)(base + h->o_x);
    e->hn = (float *)(base + h->o_hn);
    e->ytmp = (float *)(base + h->o_ytmp);
    e->hp = (float *)(base + h->o_hp);
    e->nthreads = nthreads;
    e->width = width;
    e->par_min = par_min;
    e->memo = memo;
    e->sync = (Sync *)w;
    w += sizeof(Sync) * (size_t)nthreads;
    e->mem = (Member *)w;
    w += sizeof(Member) * (size_t)nthreads;
    for (int t = 0; t < nthreads; t++) {
        e->mem[t].chg = (uint32_t *)w;
        w += ((size_t)CHIP_HIDDEN * 4 + 64) & ~(size_t)63;
    }
    for (int i = 0; i < 2; i++) {
        e->logv[i] = (float *)w;
        w += (size_t)LOG_LINES * 64;
        e->logl[i] = (uint16_t *)w;
        w += ((size_t)LOG_LINES * 2 + 63) & ~(size_t)63;
    }
    e->owner = w;
    for (uint32_t id = 0; id < 65536; id++) e->owner[id] = (uint8_t)(id % (uint32_t)width);
    w += 65536;
    e->table = (Entry *)(((uintptr_t)w + 63) & ~(uintptr_t)63);
    for (uint32_t il = 0; il < h->layers; il++)
        for (uint32_t b = 0; b < e->layers[il].n_banks; b++) {
            const ChipBank *B = &((const ChipBank *)(base + e->layers[il].o_banks))[b];
            if (B->kind == 1) e->vm = (int32_t *)(base + B->o_mem);
        }
    return e;
}

int32_t *chip_engine_vm(ChipEngine *e) { return e->vm; }
uint32_t chip_engine_token_no(const ChipEngine *e) { return e->token_no; }

/* After a resume every conv line looks again, until its ring has gone round. */
void chip_engine_wake(ChipEngine *e) {
    for (uint32_t il = 0; il < e->h->layers; il++) {
        ChipLayer *L = &e->layers[il];
        uint16_t *act = (uint16_t *)(e->base + L->o_act);
        uint8_t *in = e->base + L->o_in;
        for (uint32_t i = 0; i < L->n_conv; i++) {
            act[i] = (uint16_t)i;
            in[i] = 1;
        }
        L->n_act = L->n_conv;
    }
}

void chip_engine_stats(const ChipEngine *e, uint64_t out[CHIP_STATS]) { memcpy(out, e->st, sizeof e->st); }

int chip_engine_strays(ChipEngine *e, uint32_t *dims, uint32_t *bits, int room) {
    int n = 0;
    while (e->strays_told < e->n_strays && n < room) {
        dims[n] = e->stray_dim[e->strays_told];
        bits[n] = e->stray_bits[e->strays_told];
        e->strays_told++;
        n++;
    }
    return n;
}
