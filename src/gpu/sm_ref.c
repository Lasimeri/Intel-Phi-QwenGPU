/* See sm_ref.h and docs/machine.md. Port of kernel.run_blocks (qwenGPU, MIT) widened: 32 lanes, uniform registers,
 * banks, the 2B's warp instructions, the launch queue. Every token the circuit must write passes through sm_tok().
 *
 * A record runs lane by lane, as the circuit does (one lane's stages per forward pass): lane L's stages, its part
 * of the record (its live-out VALs, or one LSTEP), its commits (its register live-outs, its stores, its sup writes),
 * then lane L + 1; after the last lane the late uniform work (reductions, scalars, and every uniform
 * instruction) and the uniform live-outs. A value final only after the last lane is "late": a lane instruction
 * never reads one in its block (the driver's placer forbids it; this reference faults). */
#include "sm_ref.h"

#include <stdio.h>
#include <string.h>

static void sm_tok(SmMachine *m, uint32_t tok) {
    m->sink.n++;
    m->sink.hash = sm_fnv(m->sink.hash, tok);
    if (m->sink.emit) m->sink.emit(m->sink.ctx, tok);
}

/* a 32-bit value: VLO then VHI, or VLO then VEND for the last item of a part */
static void sm_val32(SmMachine *m, uint32_t v, int last) {
    sm_tok(m, SM_VLO0 + (v & 0xFFFF));
    sm_tok(m, (last ? SM_VEND0 : SM_VHI0) + (v >> 16));
}

static void fault(SmMachine *m, const char *what) {
    m->fault = 1;
    snprintf(m->fault_what, sizeof m->fault_what, "%s (block %u, warp %u, descriptor %u)", what, m->b, m->w, m->qi);
}

void sm_init(SmMachine *m, SmImage *img) {
    memset(m, 0, sizeof *m);
    m->img = img;
    m->sink.hash = SM_FNV0;
}

void sm_go(SmMachine *m) {
    if (!m->halted) return;
    m->halted = 0;
    sm_tok(m, SM_GO);
}

static int compare(uint32_t a, uint32_t b, int cmp, int sgn) {
    if (sgn) {
        int32_t x = (int32_t)a, y = (int32_t)b;
        switch (cmp) {
        case SM_CMP_EQ: return x == y;
        case SM_CMP_NE: return x != y;
        case SM_CMP_LT: return x < y;
        case SM_CMP_LE: return x <= y;
        case SM_CMP_GT: return x > y;
        default: return x >= y;
        }
    }
    switch (cmp) {
    case SM_CMP_EQ: return a == b;
    case SM_CMP_NE: return a != b;
    case SM_CMP_LT: return a < b;
    case SM_CMP_LE: return a <= b;
    case SM_CMP_GT: return a > b;
    default: return a >= b;
    }
}

static uint32_t isqrt64(uint64_t v) {
    uint64_t r = 0, bit = 1ull << 62;
    while (bit > v) bit >>= 2;
    while (bit) {
        if (v >= r + bit) {
            v -= r + bit;
            r = (r >> 1) + bit;
        } else {
            r >>= 1;
        }
        bit >>= 2;
    }
    return (uint32_t)r;
}

static int class_of(int op) {
    switch (op) {
    case SM_MAD: case SM_MULHI: case SM_MULSH: case SM_SUP: return SM_CL_MUL;
    case SM_Q8DOT: return SM_CL_DOT;
    case SM_SDOT: return SM_CL_STATE;
    case SM_AND: case SM_OR: case SM_XOR: case SM_SHL: case SM_SHR: case SM_CLZ: return SM_CL_LOGIC;
    case SM_WSUM: case SM_WMAX: return SM_CL_RED;
    case SM_UDIV: case SM_USQRT: return SM_CL_SCALAR;
    default: return SM_CL_ANY;
    }
}

/* the ops whose result is uniform whatever the operands */
static int forced_uniform(int op) { return op == SM_WSUM || op == SM_WMAX || op == SM_UDIV || op == SM_USQRT; }
/* the ops whose value is final only after the last lane */
static int late_op(int op) { return forced_uniform(op); }
/* the ops that write a register (sup: the state element after the update) */
static int writes_reg(int op) { return !(op == SM_SETP || op == SM_STV || op == SM_SCFG || op == SM_NOP); }

/* which operand fields an op reads: the encoder's rules (lower.cc) */
static void uses(const SmInstr *i, int *ra, int *rb, int *rc) {
    *ra = 1;
    *rb = i->ub != SM_SRC_IMM;
    *rc = 0;
    switch (i->op) {
    case SM_TID: case SM_SREG: *ra = *rb = 0; break;
    case SM_SHL: case SM_SHR: *rb = 0; *rc = i->sh == 63; break;
    case SM_CLZ: case SM_LD: case SM_WSUM: case SM_WMAX: *rb = 0; break;
    case SM_MAD: case SM_Q8DOT: case SM_SDOT: case SM_UDIV: *rc = 1; break;
    case SM_MULSH: *rc = i->sh == 63; break;
    case SM_USQRT: *rb = 0; *rc = 1; break;
    case SM_SUP: *rc = 1; break;
    case SM_SCFG: *rb = 0; break;
    default: break;
    }
}

/* a load from a bank, the address modulo the bank's size */
static uint32_t load(SmMachine *m, int bank, uint32_t a) {
    SmImage *g = m->img;
    switch (bank) {
    case SM_BANK_M: return g->m_words ? (uint32_t)g->m[a % g->m_words] : 0;
    case SM_BANK_A8: {
        if (!g->a_blocks) return 0;
        uint32_t blk = (a >> 6) % g->a_blocks, j = a & 63;
        if (j >= 32) return 0;
        return (uint32_t)(int32_t)(int8_t)g->a[blk * 36 + j];
    }
    case SM_BANK_A32: {
        if (!g->a_blocks) return 0;
        uint32_t blk = (a >> 6) % g->a_blocks;
        int32_t s;
        memcpy(&s, g->a + blk * 36 + 32, 4);
        return (uint32_t)s;
    }
    case SM_BANK_S: return g->s_elems ? (uint32_t)(int32_t)g->s[a % g->s_elems] : 0;
    case SM_BANK_C: return g->c_bytes ? (uint32_t)(int32_t)g->c[a % g->c_bytes] : 0;
    case SM_BANK_W8: {
        if (!g->w_blocks) return 0;
        uint32_t blk = (a >> 6) % g->w_blocks, j = a & 63;
        if (j >= 32) return 0;
        return (uint32_t)(int32_t)(int8_t)g->w[(uint64_t)blk * 34 + 2 + j];
    }
    case SM_BANK_W32: {
        if (!g->w_blocks) return 0;
        uint16_t h;
        memcpy(&h, g->w + (uint64_t)((a >> 6) % g->w_blocks) * 34, 2);
        return (uint32_t)(int32_t)g->f16_q24[h];
    }
    default: return 0;
    }
}

static void store(SmMachine *m, int bank, uint32_t a, uint32_t v) {
    if (bank == SM_BANK_M && a >= m->watch_lo && a < m->watch_hi) {
        fprintf(stderr, "store M[%u] = %d  (block %u, warp %u, descriptor %u) u:", a, (int32_t)v, m->b, m->w, m->qi - 1);
        for (int i = 0; i < SM_NUREG; i++) fprintf(stderr, " %u", m->ureg[m->w % SM_WPB_MAX][i]);
        fprintf(stderr, "\n");
    }
    SmImage *g = m->img;
    switch (bank) {
    case SM_BANK_M:
        if (g->m_words) g->m[a % g->m_words] = (int32_t)v;
        break;
    case SM_BANK_A8: {
        if (!g->a_blocks) break;
        uint32_t blk = (a >> 6) % g->a_blocks, j = a & 63;
        if (j < 32) g->a[blk * 36 + j] = (uint8_t)v;
        break;
    }
    case SM_BANK_A32: {
        if (!g->a_blocks) break;
        uint32_t blk = (a >> 6) % g->a_blocks;
        int32_t s = (int32_t)v;
        memcpy(g->a + blk * 36 + 32, &s, 4);
        break;
    }
    case SM_BANK_S:
        if (g->s_elems) g->s[a % g->s_elems] = (int16_t)v;
        break;
    case SM_BANK_C:
        if (g->c_bytes) g->c[a % g->c_bytes] = (int8_t)v;
        break;
    default: break;
    }
}

/* q8dot: one W block against one A block, q2b_fix's gemv arithmetic */
static int32_t q8dot(SmMachine *m, int bank, uint32_t wblk, uint32_t ablk) {
    SmImage *g = m->img;
    if (!g->a_blocks) return 0;
    const uint8_t *ap = g->a + (uint64_t)(ablk % g->a_blocks) * 36;
    const int8_t *wq;
    int64_t mw;
    if (bank == 0) {
        if (!g->w_blocks) return 0;
        const uint8_t *wp = g->w + (uint64_t)(wblk % g->w_blocks) * 34;
        uint16_t h;
        memcpy(&h, wp, 2);
        mw = g->f16_q24[h];
        wq = (const int8_t *)(wp + 2);
    } else {
        const uint8_t *wp = g->a + (uint64_t)(wblk % g->a_blocks) * 36;
        int32_t s;
        memcpy(&s, wp + 32, 4);
        mw = s;
        wq = (const int8_t *)wp;
    }
    int32_t inner = 0;
    for (int i = 0; i < 32; i++) inner += (int32_t)wq[i] * (int32_t)(int8_t)ap[i];
    int32_t xs;
    memcpy(&xs, ap + 32, 4);
    int64_t t = sm_shr_round((int64_t)inner * mw, 16);
    int64_t u = sm_shr_round(t * (int64_t)xs, 16);
    return (int32_t)(uint32_t)u;
}

/* the sup writes of one lane, applied after its stages */
typedef struct {
    uint32_t a[SM_STAGES];
    uint32_t v[SM_STAGES];
    int n;
} SupBuf;

static int in_supbuf(const SmMachine *m, const SupBuf *b, uint32_t a) {
    const uint32_t n = m->img->s_elems;
    for (int i = 0; i < b->n; i++)
        if (n && b->a[i] % n == a % n) return 1;
    return 0;
}

/* The cross-lane hazard guard. A lane's store lands before the next lane's pass, so a later lane loading the same place
 * would see it: that is lane-serial semantics, but no kernel of the 2B may depend on it (the lane-parallel reading
 * of the same CUDA code would differ), so the reference faults. Keys: a bank class and the reduced index. */
enum { HZ_M, HZ_A, HZ_S, HZ_C };
static inline uint32_t hz_slot(uint64_t key) {
    key ^= key >> 29;
    key *= 0x9E3779B97F4A7C15ull;
    return (uint32_t)(key >> 40) & 2047;
}
static void hz_note(SmMachine *m, int cls, uint32_t idx) {
    const uint64_t key = (uint64_t)cls << 32 | idx;
    uint32_t h = hz_slot(key);
    for (;;) {
        if (m->hz_gen[h] != m->hz_cur) {
            m->hz_gen[h] = m->hz_cur;
            m->hz_key[h] = key;
            m->hz_n++;
            return;
        }
        if (m->hz_key[h] == key) return;
        h = (h + 1) & 2047;
    }
}
static int hz_hit(const SmMachine *m, int cls, uint32_t idx) {
    if (!m->hz_n) return 0;
    const uint64_t key = (uint64_t)cls << 32 | idx;
    uint32_t h = hz_slot(key);
    for (;;) {
        if (m->hz_gen[h] != m->hz_cur) return 0;
        if (m->hz_key[h] == key) return 1;
        h = (h + 1) & 2047;
    }
}
/* the class and reduced index of a bank access; -1 for a bank that is never written (W) */
static int hz_of(const SmMachine *m, int bank, uint32_t a, uint32_t *idx) {
    const SmImage *g = m->img;
    switch (bank) {
    case SM_BANK_M: *idx = g->m_words ? a % g->m_words : 0; return HZ_M;
    case SM_BANK_A8: case SM_BANK_A32: *idx = g->a_blocks ? (a >> 6) % g->a_blocks : 0; return HZ_A;
    case SM_BANK_S: *idx = g->s_elems ? a % g->s_elems : 0; return HZ_S;
    case SM_BANK_C: *idx = g->c_bytes ? a % g->c_bytes : 0; return HZ_C;
    default: return -1;
    }
}
static int hz_load(SmMachine *m, int bank, uint32_t a) {
    uint32_t idx;
    const int cls = hz_of(m, bank, a, &idx);
    return cls >= 0 && hz_hit(m, cls, idx);
}
static void hz_store(SmMachine *m, int bank, uint32_t a) {
    uint32_t idx;
    const int cls = hz_of(m, bank, a, &idx);
    if (cls >= 0) hz_note(m, cls, idx);
}

/* one block for the current warp: lane by lane, then the uniform work; the record's tokens, the commits, the branch */
static void run_block(SmMachine *m) {
    SmImage *g = m->img;
    const SmBlock *blk = &g->rom[m->b];
    SmLive lv;
    sm_live_decode(blk->live, &lv);
    const int slot = m->w % SM_WPB_MAX;
    uint32_t (*regs)[SM_LANES] = m->regs[slot];
    uint32_t *uregs = m->ureg[slot];
    const uint32_t t0 = m->w * SM_LANES;

    /* --- the block's shape: which stages are active, uniform, late; the operand fields in use --- */
    SmInstr ins[SM_STAGES];
    uint8_t active[SM_STAGES], uni[SM_STAGES], late[SM_STAGES], use_a[SM_STAGES], use_b[SM_STAGES], use_c[SM_STAGES];
    {
        int lastw[SM_NREG], ulastw[SM_NUREG];
        for (int r = 0; r < SM_NREG; r++) lastw[r] = -1;
        for (int r = 0; r < SM_NUREG; r++) ulastw[r] = -1;
        for (int s = 0; s < SM_STAGES; s++) {
            sm_decode(blk->rows[s], &ins[s]);
            active[s] = ins[s].valid && ins[s].op != SM_NOP;
            uni[s] = late[s] = use_a[s] = use_b[s] = use_c[s] = 0;
            if (!active[s]) continue;
            if (!(g->classes[s] & class_of(ins[s].op))) {
                fault(m, "an instruction in a stage without its unit");
                return;
            }
            m->instrs++;
            m->by_op[ins[s].op]++;
            int ua, ub, uc;
            uses(&ins[s], &ua, &ub, &uc);
            use_a[s] = (uint8_t)ua;
            use_b[s] = (uint8_t)ub;
            use_c[s] = (uint8_t)uc;
            uni[s] = ins[s].ud || forced_uniform(ins[s].op);
            int reads_late = 0;
            if (ua && ins[s].ra != SM_ZERO) {
                int w = ins[s].ua ? ulastw[ins[s].ra] : lastw[ins[s].ra];
                if (w >= 0 && late[w]) reads_late = 1;
            }
            if (ub && ins[s].rb != SM_ZERO) {
                int w = ins[s].ub == SM_SRC_UNIFORM ? ulastw[ins[s].rb] : lastw[ins[s].rb];
                if (w >= 0 && late[w]) reads_late = 1;
            }
            if (uc && ins[s].rc != SM_ZERO) {
                int w = ins[s].uc ? ulastw[ins[s].rc] : lastw[ins[s].rc];
                if (w >= 0 && late[w]) reads_late = 1;
            }
            const int cl = class_of(ins[s].op);
            late[s] = (uint8_t)(late_op(ins[s].op) || reads_late);
            if (reads_late && (!uni[s] || cl == SM_CL_SCALAR)) {
                fault(m, "a lane instruction reads a late result in its block");
                return;
            }
            if (writes_reg(ins[s].op) && ins[s].rd != SM_ZERO) {
                if (uni[s]) ulastw[ins[s].rd] = s;
                else lastw[ins[s].rd] = s;
            }
        }
    }
    int lane_live = 0, nred = 0;
    for (int i = 0; i < lv.nlive; i++) {
        const int s = lv.src[i];
        if (!active[s]) {
            fault(m, "a live-out of a stage without a result");
            return;
        }
        if (!uni[s]) lane_live++;
    }
    for (int s = 0; s < SM_STAGES; s++) nred += active[s] && class_of(ins[s].op) == SM_CL_RED;

    /* --- state across the lanes --- */
    uint32_t red_sum[SM_STAGES];
    int32_t red_max[SM_STAGES];
    uint8_t red_any[SM_STAGES];
    uint8_t stage_pred[SM_STAGES][SM_LANES];
    uint8_t on0[SM_STAGES];  /* lane 0's guard, for a uniform store */
    memset(red_sum, 0, sizeof red_sum);
    memset(red_any, 0, sizeof red_any);
    memset(stage_pred, 0, sizeof stage_pred);
    memset(on0, 1, sizeof on0);
    int32_t cfg_e = m->scfg_e, cfg_sh1 = m->scfg_sh1, cfg_sh2 = m->scfg_sh2;

    m->records++;
    sm_tok(m, sm_exec(m->b, m->phase));
    m->phase = 0;
    m->hz_cur++;
    m->hz_n = 0;

    /* --- each lane: its stages, its part of the record, its commits --- */
    for (int L = 0; L < SM_LANES; L++) {
        uint32_t work[SM_NREG], uwork[SM_NUREG];
        for (int r = 0; r < SM_NREG; r++) work[r] = regs[r][L];
        memcpy(uwork, uregs, sizeof uwork);
        uint32_t res[SM_STAGES];
        uint8_t has_res[SM_STAGES], preds[SM_NPRED];
        uint8_t stv_on[SM_STAGES];
        uint32_t stv_addr[SM_STAGES], stv_val[SM_STAGES];
        SupBuf sup;
        sup.n = 0;
        memset(has_res, 0, sizeof has_res);
        memset(preds, 0, sizeof preds);
        memset(stv_on, 0, sizeof stv_on);
        int32_t e = cfg_e, sh1 = cfg_sh1, sh2 = cfg_sh2;
        for (int s = 0; s < SM_STAGES; s++) {
            if (!active[s]) continue;
            const SmInstr *in = &ins[s];
            /* late work waits for the last lane; a reduction accumulates this lane's operand below */
            const uint32_t A = !use_a[s] ? 0 : in->ua ? (in->ra == SM_ZERO ? 0 : uwork[in->ra]) : (in->ra == SM_ZERO ? 0 : work[in->ra]);
            const uint32_t B = in->ub == SM_SRC_IMM ? in->imm : !use_b[s] ? 0 : in->ub == SM_SRC_UNIFORM ? (in->rb == SM_ZERO ? 0 : uwork[in->rb]) : (in->rb == SM_ZERO ? 0 : work[in->rb]);
            const uint32_t C = !use_c[s] ? 0 : in->uc ? (in->rc == SM_ZERO ? 0 : uwork[in->rc]) : (in->rc == SM_ZERO ? 0 : work[in->rc]);
            const int on = !in->gvalid || (preds[in->g] != in->gneg);
            if (L == 0) on0[s] = (uint8_t)on;
            uint32_t r = 0;
            int writes = writes_reg(in->op);
            switch (in->op) {
            case SM_ADD: r = A + B; break;
            case SM_SUB: r = A - B; break;
            case SM_MAD: r = A * B + C; break;
            case SM_MULHI: r = (uint32_t)(((uint64_t)A * B) >> 32); break;
            case SM_MULSH: {
                int64_t p = (int64_t)(int32_t)A * (int64_t)(int32_t)B;
                r = (uint32_t)sm_shr_round(p, in->sh == 63 ? (int)(int32_t)C : in->sh);
                break;
            }
            case SM_AND: r = A & B; break;
            case SM_OR: r = A | B; break;
            case SM_XOR: r = A ^ B; break;
            case SM_CLZ: r = A ? (uint32_t)__builtin_clz(A) : 32u; break;
            case SM_SHL: {
                uint32_t k = in->sh == 63 ? (C & 63) : in->sh;
                r = k >= 32 ? 0 : A << k;
                break;
            }
            case SM_SHR: {
                uint32_t k = in->sh == 63 ? (C & 63) : in->sh;
                if (in->sgn) r = (uint32_t)((int32_t)A >> (k > 31 ? 31 : k));
                else r = k >= 32 ? 0 : A >> k;
                break;
            }
            case SM_MAX: r = compare(A, B, SM_CMP_LT, in->sgn) ? B : A; break;
            case SM_MIN: r = compare(A, B, SM_CMP_LT, in->sgn) ? A : B; break;
            case SM_SETP:
                preds[in->pd] = (uint8_t)compare(A, B, in->cmp, in->sgn);
                stage_pred[s][L] = preds[in->pd];
                break;
            case SM_SELP: r = on ? A : B; break;
            case SM_TID: r = t0 + (uint32_t)L; break;
            case SM_SREG: {
                uint32_t sh = in->imm & 31, wd = in->imm >> 8 & 31;
                r = ((t0 + (uint32_t)L) >> sh) & (wd >= 32 ? 0xFFFFFFFFu : (1u << wd) - 1);
                break;
            }
            case SM_LD:
                r = load(m, in->bank, A + in->imm);
                if (in->bank == SM_BANK_S && in_supbuf(m, &sup, A + in->imm)) {
                    fault(m, "a lane reads a state element its sup wrote in this block");
                    return;
                }
                if (hz_load(m, in->bank, A + in->imm)) {
                    fault(m, "a lane loads what an earlier lane of its block stored");
                    return;
                }
                break;
            case SM_STV:
                stv_on[s] = (uint8_t)on;
                stv_addr[s] = A + in->imm;
                stv_val[s] = B;
                break;
            case SM_Q8DOT:
                /* an immediate rb is the whole A block index and nothing is added to the W address; a register rb and ra both
                 * advance by imm */
                r = C + (uint32_t)(in->ub == SM_SRC_IMM ? q8dot(m, in->bank, A, B) : q8dot(m, in->bank, A + in->imm, B + in->imm));
                {
                    const uint32_t ablk = in->ub == SM_SRC_IMM ? B : B + in->imm, xblk = in->ub == SM_SRC_IMM ? A : A + in->imm;
                    if (hz_load(m, SM_BANK_A8, ablk << 6) || (in->bank == 1 && hz_load(m, SM_BANK_A8, xblk << 6))) {
                        fault(m, "a lane loads what an earlier lane of its block stored");
                        return;
                    }
                }
                break;
            case SM_SDOT:
                if (in_supbuf(m, &sup, A + in->imm)) {
                    fault(m, "a lane reads a state element its sup wrote in this block");
                    return;
                }
                if (hz_load(m, SM_BANK_S, A + in->imm)) {
                    fault(m, "a lane loads what an earlier lane of its block stored");
                    return;
                }
                r = C + (uint32_t)((int32_t)load(m, SM_BANK_S, A + in->imm) * (int32_t)(int8_t)B);
                break;
            case SM_SCFG:
                e = (int32_t)(A & 0xFFFF);
                sh1 = (int32_t)(A >> 16 & 127) - 32;
                sh2 = (int32_t)(A >> 23 & 127) - 32;
                break;
            case SM_SUP: {
                /* the result is the state element after the update: unchanged for a lane whose guard is off */
                const uint32_t a = A + in->imm;
                if (in_supbuf(m, &sup, a)) {
                    fault(m, "a lane writes a state element twice in this block");
                    return;
                }
                if (hz_load(m, SM_BANK_S, a)) {
                    fault(m, "a lane loads what an earlier lane of its block stored");
                    return;
                }
                /* rd is read and written: the largest magnitude this lane has written, carried across a run of sups */
                const int32_t acc = in->rd == SM_ZERO ? 0 : (int32_t)work[in->rd];
                r = (uint32_t)acc;
                if (!on) break;
                int64_t old = (int32_t)load(m, SM_BANK_S, a);
                int64_t val = sm_shr_round(old * e, sh1) + sm_shr_round((int64_t)(int32_t)C * (int64_t)(int32_t)B, 16);
                int64_t sv = sm_shr_round(val, sh2);
                if (sv > 32767) sv = 32767;
                if (sv < -32767) sv = -32767;
                sup.a[sup.n] = a;
                sup.v[sup.n] = (uint32_t)(int32_t)sv;
                sup.n++;
                const int32_t mag = (int32_t)(sv < 0 ? -sv : sv);
                r = (uint32_t)(mag > acc ? mag : acc);
                break;
            }
            case SM_WSUM:
                writes = 0;
                if (on) red_sum[s] += A;
                break;
            case SM_WMAX:
                writes = 0;
                if (on && (!red_any[s] || (int32_t)A > red_max[s])) {
                    red_max[s] = (int32_t)A;
                    red_any[s] = 1;
                }
                break;
            case SM_UDIV: case SM_USQRT:
                writes = 0;  /* late: after the last lane */
                break;
            default:
                fault(m, "unknown op");
                return;
            }
            if (!writes || late[s]) continue;
            has_res[s] = 1;
            res[s] = r;
            if (in->rd == SM_ZERO) continue;
            if (uni[s]) uwork[in->rd] = r;
            else work[in->rd] = r;
        }
        /* this lane's part of the record: its live-outs, then each reduction's running partial (the circuit reads the
         * previous lane's partial from these tokens), or one LSTEP when there is neither */
        if (lane_live || nred) {
            uint32_t items[SM_LMAX + SM_STAGES];
            int n = 0;
            for (int i = 0; i < lv.nlive; i++) {
                const int s = lv.src[i];
                if (uni[s]) continue;
                if (ins[s].op == SM_STV) {
                    items[n++] = stv_val[s];
                } else {
                    if (!has_res[s]) {
                        fault(m, "a live-out of a stage without a result");
                        return;
                    }
                    items[n++] = res[s];
                }
            }
            for (int s = 0; s < SM_STAGES; s++) {
                if (!active[s] || class_of(ins[s].op) != SM_CL_RED) continue;
                items[n++] = ins[s].op == SM_WSUM ? red_sum[s] : (uint32_t)(red_any[s] ? red_max[s] : INT32_MIN);
            }
            for (int i = 0; i < n; i++) sm_val32(m, items[i], i == n - 1);
        } else {
            sm_tok(m, SM_LSTEP);
            m->lsteps++;
        }
        /* its commits: register live-outs and stores in LIVE order, then its state writes */
        for (int i = 0; i < lv.nlive; i++) {
            const int s = lv.src[i];
            if (uni[s]) continue;
            if (ins[s].op == SM_STV) {
                if (stv_on[s]) {
                    store(m, ins[s].bank, stv_addr[s], stv_val[s]);
                    hz_store(m, ins[s].bank, stv_addr[s]);
                }
            } else if (ins[s].rd != SM_ZERO) {
                regs[ins[s].rd][L] = res[s];
            }
        }
        for (int i = 0; i < sup.n; i++) {
            store(m, SM_BANK_S, sup.a[i], sup.v[i]);
            hz_store(m, SM_BANK_S, sup.a[i]);
        }
        if (L == SM_LANES - 1) {
            cfg_e = e;
            cfg_sh1 = sh1;
            cfg_sh2 = sh2;
        }
    }
    m->scfg_e = cfg_e;
    m->scfg_sh1 = cfg_sh1;
    m->scfg_sh2 = cfg_sh2;

    /* --- the uniform work, with every lane done: reductions, scalars and the uniform instructions in stage order --- */
    uint32_t uwork[SM_NUREG], ures[SM_STAGES], ustv_addr[SM_STAGES], ustv_val[SM_STAGES];
    uint8_t uhas[SM_STAGES];
    memcpy(uwork, uregs, sizeof uwork);
    memset(uhas, 0, sizeof uhas);
    for (int s = 0; s < SM_STAGES; s++) {
        if (!active[s] || !uni[s]) continue;
        const SmInstr *in = &ins[s];
        const uint32_t A = !use_a[s] ? 0 : (in->ra == SM_ZERO ? 0 : uwork[in->ra]);
        const uint32_t B = in->ub == SM_SRC_IMM ? in->imm : !use_b[s] ? 0 : (in->rb == SM_ZERO ? 0 : uwork[in->rb]);
        const uint32_t C = !use_c[s] ? 0 : (in->rc == SM_ZERO ? 0 : uwork[in->rc]);
        uint32_t r = 0;
        switch (in->op) {
        case SM_ADD: r = A + B; break;
        case SM_SUB: r = A - B; break;
        case SM_MAD: r = A * B + C; break;
        case SM_MULHI: r = (uint32_t)(((uint64_t)A * B) >> 32); break;
        case SM_MULSH: {
            int64_t p = (int64_t)(int32_t)A * (int64_t)(int32_t)B;
            r = (uint32_t)sm_shr_round(p, in->sh == 63 ? (int)(int32_t)C : in->sh);
            break;
        }
        case SM_AND: r = A & B; break;
        case SM_OR: r = A | B; break;
        case SM_XOR: r = A ^ B; break;
        case SM_CLZ: r = A ? (uint32_t)__builtin_clz(A) : 32u; break;
        case SM_SHL: {
            uint32_t k = in->sh == 63 ? (C & 63) : in->sh;
            r = k >= 32 ? 0 : A << k;
            break;
        }
        case SM_SHR: {
            uint32_t k = in->sh == 63 ? (C & 63) : in->sh;
            if (in->sgn) r = (uint32_t)((int32_t)A >> (k > 31 ? 31 : k));
            else r = k >= 32 ? 0 : A >> k;
            break;
        }
        case SM_MAX: r = compare(A, B, SM_CMP_LT, in->sgn) ? B : A; break;
        case SM_MIN: r = compare(A, B, SM_CMP_LT, in->sgn) ? A : B; break;
        case SM_SELP: r = on0[s] ? A : B; break;
        case SM_SREG: {  /* a field of the thread index above the lane bits: the same in every lane */
            uint32_t sh = in->imm & 31, wd = in->imm >> 8 & 31;
            r = (t0 >> sh) & (wd >= 32 ? 0xFFFFFFFFu : (1u << wd) - 1);
            break;
        }
        case SM_LD: r = load(m, in->bank, A + in->imm); break;
        case SM_STV:
            ustv_addr[s] = A + in->imm;
            ustv_val[s] = B;
            continue;
        case SM_Q8DOT: r = C + (uint32_t)(in->ub == SM_SRC_IMM ? q8dot(m, in->bank, A, B) : q8dot(m, in->bank, A + in->imm, B + in->imm)); break;
        case SM_SDOT: r = C + (uint32_t)((int32_t)load(m, SM_BANK_S, A + in->imm) * (int32_t)(int8_t)B); break;
        case SM_WSUM: r = red_sum[s]; break;
        case SM_WMAX: r = (uint32_t)(red_any[s] ? red_max[s] : INT32_MIN); break;
        case SM_UDIV: {
            uint64_t n = (uint64_t)C << 32 | A;
            r = B ? (uint32_t)(n / B) : 0;
            break;
        }
        case SM_USQRT: r = isqrt64((uint64_t)C << 32 | A); break;
        default:
            fault(m, "a uniform instruction of an unexpected kind");
            return;
        }
        uhas[s] = 1;
        ures[s] = r;
        if (in->rd != SM_ZERO) uwork[in->rd] = r;
    }
    /* the uniform live-outs and their commits; a reduction's value is already in the stream (the last lane's partial) */
    int nuni_items = 0, emitted = 0;
    for (int i = 0; i < lv.nlive; i++) {
        const int s = lv.src[i];
        if (uni[s] && class_of(ins[s].op) != SM_CL_RED) nuni_items++;
    }
    for (int i = 0; i < lv.nlive; i++) {
        const int s = lv.src[i];
        if (!uni[s]) continue;
        if (ins[s].op == SM_STV) {
            sm_val32(m, ustv_val[s], ++emitted == nuni_items);
            if (on0[s]) store(m, ins[s].bank, ustv_addr[s], ustv_val[s]);
            continue;
        }
        if (!uhas[s]) {
            fault(m, "a live-out of a stage without a result");
            return;
        }
        if (class_of(ins[s].op) != SM_CL_RED) sm_val32(m, ures[s], ++emitted == nuni_items);
        if (ins[s].rd != SM_ZERO) uregs[ins[s].rd] = ures[s];
    }

    /* where control goes */
    if (m->trace_desc_on && m->qi - 1 == m->trace_desc && m->w == 0) {
        fprintf(stderr, "d %u block +%u committed, u:", m->qi - 1, m->b - m->entry);
        for (int i = 0; i < SM_NUREG; i++) fprintf(stderr, " %u", uregs[i]);
        fprintf(stderr, "\n");
    }
    const uint32_t wpb = 1u << lv.lwpb;
    const int last_warp = (m->w % wpb) == wpb - 1;
    switch (lv.branch) {
    case SM_BR_NEXT: m->b++; break;
    case SM_BR_JUMP: m->b = lv.target; break;
    case SM_BR_COND: {
        int taken;
        if (lv.any) {
            int anyp = 0;
            for (int L = 0; L < SM_LANES; L++) anyp |= stage_pred[lv.stage][L];
            taken = anyp != lv.neg;
        } else {
            taken = stage_pred[lv.stage][0] != lv.neg;
        }
        m->b = taken ? lv.target : m->b + 1;
        break;
    }
    case SM_BR_END:
        if (last_warp) {
            m->w++;
            m->b = m->entry;
            m->phase = 1;
            if (m->w >= m->warps) m->running = 0;
            else sm_tok(m, sm_warp_tok(m->w));
        } else {
            m->w++;
            m->b = m->R;
            sm_tok(m, sm_warp_tok(m->w));
        }
        break;
    case SM_BR_BAR:
        if (last_warp) {
            m->w -= wpb - 1;
            m->b++;
            m->phase = 1;
        } else {
            m->w++;
            m->b = m->R;
        }
        sm_tok(m, sm_warp_tok(m->w));  /* the warp changes either way (a thread block of one warp keeps its w: told anyway) */
        break;
    default:
        fault(m, "unknown branch");
        return;
    }
    if (m->running && m->b >= g->nblk) fault(m, "control past the ROM");
}

static void start_launch(SmMachine *m, uint32_t entry, uint32_t warps, uint32_t lwpb, uint32_t args) {
    for (int slot = 0; slot < SM_WPB_MAX; slot++) m->ureg[slot][SM_UARGS] = args;
    m->w = 0;
    m->b = entry;
    m->entry = entry;
    m->R = entry;
    m->warps = warps ? warps : 256;  /* the descriptor's 8-bit field: 0 means 256 */
    m->lwpb = lwpb;
    m->phase = 1;
    m->running = 1;
    m->launches++;
}

int sm_run(SmMachine *m, uint64_t max_records) {
    SmImage *g = m->img;
    const uint64_t stop = m->records + max_records;
    while (!m->fault) {
        if (m->halted) return 0;
        if (m->stop_req) return 4;
        if (m->running) {
            if (m->records >= stop) return 1;
            if (m->phase) m->R = m->b;
            run_block(m);
            continue;
        }
        /* the queue */
        uint32_t qi = m->qi;
        if (m->stop_qi && qi == m->stop_qi) return 3;
        if (2 * qi + 1 >= g->m_words) {
            fault(m, "the queue runs off M");
            break;
        }
        uint32_t w0 = (uint32_t)g->m[SM_M_QUEUE + 2 * qi], w1 = (uint32_t)g->m[SM_M_QUEUE + 2 * qi + 1];
        uint32_t kind = w0 & 3;
        sm_tok(m, sm_next(qi));
        m->qi = qi + 1;
        if (kind == SM_Q_LAUNCH) {
            start_launch(m, w0 >> 2 & 4095, w0 >> 14 & 255, w0 >> 22 & 3, w1);
        } else if (kind == SM_Q_BRANCH) {
            uint32_t target = (w0 >> 24) | (w1 >> 24) << 8;
            if (g->m_words && g->m[(w1 & 0xFFFFFF) % g->m_words]) m->qi = target;
        } else if (kind == SM_Q_EMIT) {
            sm_tok(m, SM_EMIT);
            sm_val32(m, g->m_words ? (uint32_t)g->m[(w1 & 0xFFFFFF) % g->m_words] : 0, 1);
        } else {
            sm_tok(m, SM_HALT);
            m->halted = 1;
            return 0;
        }
    }
    return 2;
}
