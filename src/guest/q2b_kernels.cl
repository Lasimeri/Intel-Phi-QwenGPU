// The 2B as kernels of the embedded GPU, in OpenCL C (docs/machine.md; the device header src/cl/qgpucl.h): every
// number src/text/q2b_fix.cc's, launch by launch from the queue, nothing computed by a host. One work-item per
// element; a warp per 32; a work-group of 4 or 8 warps where a vector's lanes must meet at a barrier. No per-lane
// branch anywhere but the early exit; loops are uniform.
//
// Conventions: a kernel's pointer parameters are words of M (the builder bakes them as constants, or passes them in
// the args block through u14) for the big arrays; the job words, the scalars and the function tables are read and
// written by word address through __ldm / __stvm, which the compiler cannot move under a branch. `pos` and the
// context come from the job words; per-head words are runs of 32.
#include "qgpucl.h"
#include "q2b_map.h"

#define LANE_INDEX ((int)get_global_id(0))
#define GROUP ((int)get_group_id(0))
#define LOCAL ((int)get_local_id(0))
#define WARP_IN_BLOCK (get_local_id(0) >> 5)
#define JOB(w) __ldm((u32)(w))
#define SC(w) __ldm((u32)(w))
#define SET_SC(w, v) __stvm((u32)(w), (v))
#define UNROLL __attribute__((opencl_unroll_hint))
// a work-group's first warp starts a reduction across the group's warps in a scratch word; later warps fold in
#define BLOCK_MAX(w, v) do { i32 cur_ = SC(w); SET_SC((w), imax(WARP_IN_BLOCK == 0 ? (i32)0x80000000 : cur_, (v))); } while (0)
#define BLOCK_SUM(w, v) do { i32 cur_ = SC(w); SET_SC((w), (WARP_IN_BLOCK == 0 ? 0 : cur_) + (v)); } while (0)

static inline i32 iabs(i32 v) { return v < 0 ? -v : v; }
static inline i32 imax(i32 a, i32 b) { return a > b ? a : b; }
static inline i32 imin(i32 a, i32 b) { return a < b ? a : b; }
static inline i32 clamp127(i32 v) { return imin(imax(v, -127), 127); }

// the tables: silu of Q.16 in Q.16, sigmoid of Q.16 in Q.15, softplus, exp(-d) of Q.16 d >= 0 in Q.15
static inline i32 silu(i32 x) {
    i32 idx = imin(imax((x >> 8) + 2048, 0), 4095);
    i32 t = __ldm((u32)M_SILU + (u32)idx);
    return x >= (8 << 16) ? x : (x < -(8 << 16) ? 0 : t);
}
static inline i32 sigmoid(i32 x) {
    i32 idx = imin(imax((x >> 8) + 2048, 0), 4095);
    i32 t = __ldm((u32)M_SIG + (u32)idx);
    return x >= (8 << 16) ? 32768 : (x < -(8 << 16) ? 0 : t);
}
static inline i32 softplus(i32 x) {
    i32 idx = imin(imax((x >> 8) + 4096, 0), 8191);
    i32 t = __ldm((u32)M_SP + (u32)idx);
    return x >= (16 << 16) ? x : (x < -(16 << 16) ? 0 : t);
}
static inline i32 exp_neg(i32 d) {
    i32 idx = imin(imax(d >> 5, 0), 16383);
    i32 t = __ldm((u32)M_EXP + (u32)idx);
    return d <= 0 ? 32768 : (idx >= 16383 && d >= (8 << 16) ? 0 : t);
}
// (big * 256 + 63) / 127, at most INT32_MAX: the Q.24 scale of a vector whose largest magnitude is big
static inline i32 scale_of(u32 big) {
    u32 q = __udiv(big << 8 | 63u, big >> 24, 127u);
    return big >= 0x3F800000u ? 0x7FFFFFFF : (i32)q;
}
// 2^48 / (rms8 * scale) in 32 bits: the divisor shifted into 32 bits by k, the numerator by the same k
static inline i32 div48(u32 rms8, u32 scale) {
    u32 hi = __umulhi(rms8, scale), lo = rms8 * scale;
    i32 k = hi ? 32 - __clz32(hi) : 0;
    u32 d32 = hi ? (hi << (32 - k)) | (lo >> k) : lo;
    u32 nhi = 1u << (16 - k);
    u32 q = __udiv(0u, nhi, d32);
    i32 rs = nhi >= d32 ? 0x7FFFFFFF : (i32)q;
    return scale ? rs : 0;
}

// ----------------------------------------------------------------------------------------------------------------
// the embedding: x[i] = (q * scale + 2^7) >> 8 from the token's row of W. 64 warps.
__kernel void embed(__global int *x, int wbase, __global const int *toks) {
    int i = LANE_INDEX;
    int pos = JOB(JOB_POS);
    int tok = toks[pos];
    u32 blk = (u32)(wbase + tok * 64 + (i >> 5));
    i32 q = __ldw8(blk * 64 + (u32)(i & 31));
    i32 mw = __ldw32(blk * 64);
    x[i] = __mulsh(q, mw, 8);
}

// ----------------------------------------------------------------------------------------------------------------
// the RMS norm of a vector of n (a multiple of 32) in five launches: the largest magnitude, the reciprocal, the
// quantized sum of squares, the scale, the scaled product with the weights
__kernel void norm_a(__global const int *x) {
    int i = LANE_INDEX;
    i32 m = __wmax(iabs(x[i]));
    i32 cur = SC(SC_BIG);
    i32 prev = GROUP == 0 ? 0 : cur;
    SET_SC(SC_BIG, imax(prev, m));
}
__kernel void norm_b(void) {
    u32 big = (u32)SC(SC_BIG);
    SET_SC(SC_R, (i32)__udiv(127u << 24, 0u, big));
    SET_SC(SC_SCALE, scale_of(big));
}
__kernel void norm_c(__global const int *x) {
    int i = LANE_INDEX;
    i32 r = SC(SC_R);
    i32 q = clamp127(__mulsh(x[i], r, 24));
    i32 s = __wsum(q * q);
    i32 cur = SC(SC_S);
    i32 prev = GROUP == 0 ? 0 : cur;
    SET_SC(SC_S, prev + s);
}
__kernel void norm_d(int n) {
    u32 s = (u32)SC(SC_S);
    u32 scale = (u32)SC(SC_SCALE);
    u32 rms8 = __usqrt(__udiv(s << 16, s >> 16, (u32)n), 0u);
    SET_SC(SC_RS, div48(rms8, scale));
}
__kernel void norm_e(__global const int *x, __global const int *w, __global int *out) {
    int i = LANE_INDEX;
    i32 rs = SC(SC_RS);
    out[i] = __mulsh(__mulsh(x[i], rs, 16), w[i], 16);
}

// ----------------------------------------------------------------------------------------------------------------
// one A block per warp: the int8 lanes and the block's Q.24 scale
__kernel void quant_blocks(__global const int *x, int ablk0) {
    int i = LANE_INDEX;
    i32 v = x[i];
    u32 big = (u32)__wmax(iabs(v));
    i32 r = (i32)__udiv(127u << 24, 0u, big);
    i32 q = clamp127(__mulsh(v, r, 24));
    u32 blk = (u32)(ablk0 + GROUP);
    __stva8(blk * 64 + (u32)LOCAL, q);
    __stva32(blk * 64 + 32, scale_of(big));
}

// ----------------------------------------------------------------------------------------------------------------
// gemv: lane = row, 64 or 192 Q8 blocks against the A blocks from ablk0 (a constant), the W row from wbase (an arg)
#define GEMV_BODY(NB)                                                   \
    int r = LANE_INDEX;                                                 \
    u32 wrow = (u32)wbase + (u32)r * (NB);                              \
    i32 acc = 0;                                                        \
    UNROLL for (int b = 0; b < (NB); b++) acc = __q8dot(acc, wrow + (u32)b, (u32)(ablk0 + b)); \
    y[r] = acc;

__kernel void gemv_h64(int wbase, int ablk0, __global int *y) { GEMV_BODY(64) }
__kernel void gemv_ffx192(int wbase, int ablk0, __global int *y) { GEMV_BODY(192) }
// the GDN projection goes into the window's slot of this position
__kernel void gemv_ring64(int wbase, int ablk0, __global int *ring) {
    int pos = JOB(JOB_POS);
    __global int *y = ring + (pos & 3) * Q2B_QKV;
    GEMV_BODY(64)
}
// the logits of 8192 rows from row0 (an arg); the best so far and its row in the scalars (row0 = 0 starts them)
__kernel void gemv_logits(int wbase, int ablk0, int row0) {
    int r = row0 + LANE_INDEX;
    u32 wrow = (u32)wbase + (u32)r * 64;
    i32 acc = 0;
    UNROLL for (int b = 0; b < 64; b++) acc = __q8dot(acc, wrow + (u32)b, (u32)(ablk0 + b));
    i32 wm = __wmax(acc);
    i32 cand = acc == wm ? -r : (i32)0x80000000;
    i32 idx = -__wmax(cand);
    int first = row0 == 0 && GROUP == 0;
    i32 cur_best = SC(SC_BEST), cur_idx = SC(SC_BEST_IDX);
    i32 best = first ? (i32)0x80000000 : cur_best;
    i32 bidx = first ? 0 : cur_idx;
    int take = wm > best;
    SET_SC(SC_BEST, take ? wm : best);
    SET_SC(SC_BEST_IDX, take ? idx : bidx);
}

// ----------------------------------------------------------------------------------------------------------------
// elementwise
__kernel void silu_mul(__global const int *g, __global const int *u, __global int *x) {
    int i = LANE_INDEX;
    x[i] = __mulsh(silu(g[i]), u[i], 16);
}
__kernel void add_into(__global int *x, __global const int *y) {
    int i = LANE_INDEX;
    x[i] = x[i] + y[i];
}
// the causal conv over the window's 4 slots (oldest first: slot (pos + t + 1) & 3 for tap t), then silu
__kernel void conv_silu(__global const int *ring, __global const int *w, __global int *conv) {
    int c = LANE_INDEX;
    int pos = JOB(JOB_POS);
    i32 acc = 0;
    UNROLL for (int t = 0; t < 4; t++) {
        int slot = (pos + t + 1) & 3;
        acc = acc + __mulsh(w[t * Q2B_QKV + c], ring[slot * Q2B_QKV + c], 16);
    }
    conv[c] = silu(acc);
}
// per value head: sigmoid(beta) and the decay exp(softplus(alpha + dt) * a); 1 warp, lanes past 16 are scratch
__kernel void gdn_gates(__global const int *alpha, __global const int *beta_raw, __global const int *dt, __global const int *a, __global int *decay,
                        __global int *betas) {
    int i = LANE_INDEX;
    betas[i] = sigmoid(beta_raw[i]);
    i32 gate = __mulsh(softplus(alpha[i] + dt[i]), a[i], 16);
    decay[i] = exp_neg(-gate);
}

// ----------------------------------------------------------------------------------------------------------------
// a vector of 128 per work-group (4 warps), 16 groups a launch: int8 lanes into q (words), and a scale per vector:
// unit: the 1 / length (over sqrt(128) for the query); scale: the value's Q.24 scale
#define QV128_HEAD                                                                   \
    int v = GROUP;                                                                   \
    int j = LOCAL;                                                                   \
    i32 xv = x[v * 128 + j];                                                         \
    i32 m = __wmax(iabs(xv));                                                        \
    BLOCK_MAX(SC_TMP + v, m);                                                        \
    barrier(CLK_GLOBAL_MEM_FENCE);                                                   \
    u32 big = (u32)SC(SC_TMP + v);                                                   \
    i32 r = (i32)__udiv(127u << 24, 0u, big);                                        \
    i32 qq = clamp127(__mulsh(xv, r, 24));                                           \
    q[v * 128 + j] = qq;

__kernel void quant_vec128_unit(__global const int *x, __global int *q, __global int *scales, int for_query) {
    QV128_HEAD
    i32 s = __wsum(qq * qq);
    BLOCK_SUM(SC_TMP2 + v, s);
    barrier(CLK_GLOBAL_MEM_FENCE);
    u32 s2 = (u32)SC(SC_TMP2 + v);
    u32 len8 = __usqrt(s2 << 16, s2 >> 16);
    u32 us = __udiv(0u, 1u, len8);
    scales[v] = for_query ? (i32)__umulhi(us, 2896u << 17) : (i32)us;
}
__kernel void quant_vec128_scale(__global const int *x, __global int *q, __global int *scales) {
    QV128_HEAD
    scales[v] = scale_of(big);
}

// ----------------------------------------------------------------------------------------------------------------
// one GDN head per work-group of 4 warps (lane j = a column of the state), 16 groups a launch. Phases: the state
// read with k, delta and its largest, the exponent, the update, the read with q, the norm and the gate. srow0 is
// the layer's first state element; se / smax are the layer's per-head words.
__kernel void gdn_head(int srow0, __global int *se_w, __global int *smax_w, __global const int *ki, __global const int *qi, __global const int *vi,
                       __global const int *decay, __global const int *betas, __global const int *ks, __global const int *qs, __global const int *vs,
                       __global const int *z, __global const int *ssm_norm, __global int *mix, __global int *kreal_w, __global int *o_w, __global int *dbg,
                       __global int *on_w, __global int *sz_w) {
    int hd = GROUP;
    int j = LOCAL;
    u32 srow = (u32)srow0 + (u32)hd * (Q2B_ST * Q2B_ST) + (u32)j;
    __global const int *k = ki + hd * Q2B_ST;
    __global const int *qv = qi + hd * Q2B_ST;
    i32 e = decay[hd], b = betas[hd], kscale = ks[hd], qscale = qs[hd], vscale = vs[hd];
    i32 se = se_w[hd];
    // the state read with the key
    i32 a = 0;
    UNROLL for (int i = 0; i < Q2B_ST; i++) a = __sdot(a, srow + (u32)(i * Q2B_ST), k[i]);
    i32 m = __mulsh(e, kscale, 15);
    i32 kv24 = __mulshr(a, m, se);
    i32 v24 = vi[hd * Q2B_ST + j] * vscale;
    i32 delta = __mulsh(v24 - kv24, b, 15);
    i32 md = __wmax(iabs(delta));
    BLOCK_MAX(SC_TMP + hd, md);
    kreal_w[hd * Q2B_ST + j] = k[j] * kscale;
    barrier(CLK_GLOBAL_MEM_FENCE);
    // the exponent from the bound, and the update's constants
    u32 maxdelta = (u32)SC(SC_TMP + hd);
    // the bound in two words: t1 fits; t2 = (127 kscale * maxdelta + 2^15) >> 16 is up to 2^42
    u32 t1 = (u32)__mulshr(smax_w[hd], e, se - 17);
    u32 a127 = (u32)(127 * kscale);
    u32 plo = a127 * maxdelta, phi = __umulhi(a127, maxdelta);
    u32 plo2 = plo + 32768u;
    u32 phi2 = phi + (plo2 < plo ? 1u : 0u);
    u32 t2lo = (phi2 << 16) | (plo2 >> 16), t2hi = phi2 >> 16;
    u32 blo = t1 + t2lo;
    u32 bhi = t2hi + (blo < t1 ? 1u : 0u);
    i32 lhi = 64 - __clz32(bhi), llo = 32 - __clz32(blo);  /* both computed, then chosen: no branch */
    i32 bitlen = bhi ? lhi : (blo ? llo : 0);
    i32 nse = (bhi | blo) ? 47 - bitlen : se;
    i32 cfg = (e & 0xFFFF) | ((se - 17 + 32) << 16) | ((32 - nse + 32) << 23);
    __scfg(cfg);
    __global const int *kreal = kreal_w + hd * Q2B_ST;
    i32 smv = 0;  /* the largest magnitude this lane wrote: sup carries it */
    UNROLL for (int i = 0; i < Q2B_ST; i++) smv = __sup(smv, srow + (u32)(i * Q2B_ST), delta, kreal[i]);
    i32 sm = __wmax(smv);
    BLOCK_MAX(SC_TMP2 + hd, sm);
    barrier(CLK_GLOBAL_MEM_FENCE);
    // the state read with the query, and the largest |o|
    i32 o = 0;
    UNROLL for (int i = 0; i < Q2B_ST; i++) o = __sdot(o, srow + (u32)(i * Q2B_ST), qv[i]);
    o_w[hd * Q2B_ST + j] = o;  /* the probe compares it with the reference's o */
    i32 om = __wmax(iabs(o));
    BLOCK_MAX(SC_TMP3 + hd, om);
    smax_w[hd] = SC(SC_TMP2 + hd);
    se_w[hd] = nse;
    barrier(CLK_GLOBAL_MEM_FENCE);
    // the norm sees o's real magnitude: quantize o, its sum of squares, then the scale f over 2^8
    u32 obig = (u32)SC(SC_TMP3 + hd);
    i32 r = (i32)__udiv(127u << 24, 0u, obig);
    i32 oq = clamp127(__mulsh(o, r, 24));
    i32 s2w = __wsum(oq * oq);
    BLOCK_SUM(SC_TMP + hd, s2w);
    barrier(CLK_GLOBAL_MEM_FENCE);
    u32 s2 = (u32)SC(SC_TMP + hd);
    i32 oscale = scale_of(obig);
    i32 isq = (i32)__usqrt(s2 << 9, s2 >> 23);
    i32 rms24 = __mulshr(__mulsh(isq, oscale, 16), qscale, nse);
    u32 lo = (u32)rms24 * (u32)rms24, hi = __umulhi((u32)rms24, (u32)rms24);
    u32 lo2 = lo + 281474977u;
    u32 hi2 = hi + (lo2 < lo ? 1u : 0u);
    u32 root24 = __usqrt(lo2, hi2);
    i32 f8 = (i32)__udiv((u32)qscale << 20, (u32)qscale >> 12, root24);
    dbg[hd * 8 + 0] = (i32)obig;  /* the head's scalar chain, for the probe */
    dbg[hd * 8 + 1] = (i32)s2;
    dbg[hd * 8 + 2] = isq;
    dbg[hd * 8 + 3] = rms24;
    dbg[hd * 8 + 4] = (i32)root24;
    dbg[hd * 8 + 5] = f8;
    dbg[hd * 8 + 6] = oscale;
    dbg[hd * 8 + 7] = nse;
    i32 on = __mulsh(__mulshr(o, f8, nse + 4), ssm_norm[j], 16);
    i32 sz = silu(z[hd * Q2B_ST + j]);
    on_w[hd * Q2B_ST + j] = on;  /* the probe compares both with the reference */
    sz_w[hd * Q2B_ST + j] = sz;
    mix[hd * Q2B_ST + j] = __mulsh(on, sz, 16);
}

// ----------------------------------------------------------------------------------------------------------------
// the attention layer: a norm of 256 per work-group of 8 warps (nvec groups a launch), RoPE, the quantizations
__kernel void norm256(__global const int *x, int xstride, __global const int *w, __global int *out) {
    int v = GROUP;
    int j = LOCAL;
    i32 xv = x[v * xstride + j];
    i32 m = __wmax(iabs(xv));
    BLOCK_MAX(SC_TMP + v, m);
    barrier(CLK_GLOBAL_MEM_FENCE);
    u32 big = (u32)SC(SC_TMP + v);
    i32 r = (i32)__udiv(127u << 24, 0u, big);
    i32 q = clamp127(__mulsh(xv, r, 24));
    i32 s = __wsum(q * q);
    BLOCK_SUM(SC_TMP2 + v, s);
    barrier(CLK_GLOBAL_MEM_FENCE);
    u32 scale = (u32)scale_of(big);
    u32 s2 = (u32)SC(SC_TMP2 + v);
    u32 rms8 = __usqrt(__udiv(s2 << 16, s2 >> 16, 256u), 0u);
    i32 rs = div48(rms8, scale);
    out[v * 256 + j] = __mulsh(__mulsh(xv, rs, 16), w[j], 16);
}
// RoPE on nvec vectors of 256: lane (vector, i < 32) rotates the pair (i, i + 32) with this position's row
__kernel void rotate(__global int *x, __global const int *rope) {
    int t = LANE_INDEX;
    int v = t >> 5, i = t & 31;
    int pos = JOB(JOB_POS);
    __global const int *row = rope + pos * Q2B_ROT;
    i32 a = x[v * 256 + i], bb = x[v * 256 + i + 32], c = row[i], s = row[32 + i];
    x[v * 256 + i] = __mulsh(a, c, 15) - __mulsh(bb, s, 15);
    x[v * 256 + i + 32] = __mulsh(a, s, 15) + __mulsh(bb, c, 15);
}
// a vector of 256 per work-group: the largest magnitude, then the int8 of each lane and the Q.24 scale
#define QV256_HEAD                                                 \
    int v = GROUP;                                                 \
    int j = LOCAL;                                                 \
    i32 xv = x[v * 256 + j];                                       \
    i32 m = __wmax(iabs(xv));                                      \
    BLOCK_MAX(SC_TMP + v, m);                                      \
    barrier(CLK_GLOBAL_MEM_FENCE);                                 \
    u32 big = (u32)SC(SC_TMP + v);                                 \
    i32 r = (i32)__udiv(127u << 24, 0u, big);                      \
    i32 q = clamp127(__mulsh(xv, r, 24));

// into 8 A blocks from ablk0 + v * 8, the scale also to scales[v]
__kernel void quant_vec256(__global const int *x, int ablk0, __global int *scales) {
    QV256_HEAD
    u32 blk = (u32)(ablk0 + v * 8) + (u32)WARP_IN_BLOCK;
    __stva8(blk * 64 + (u32)(j & 31), q);
    i32 sc_v = scale_of(big);
    __stva32(blk * 64 + 32, sc_v);
    scales[v] = sc_v;
}
// the key cache: the kv head's vector into its 8 blocks at this position (kc_layer = the layer's first block)
__kernel void quant_kc(__global const int *x, int kc_layer) {
    QV256_HEAD
    int pos = JOB(JOB_POS);
    u32 blk = (u32)kc_layer + ((u32)v * Q2B_CTX + (u32)pos) * 8 + (u32)WARP_IN_BLOCK;
    __stva8(blk * 64 + (u32)(j & 31), q);
    __stva32(blk * 64 + 32, scale_of(big));
}
// the value cache, transposed: v[i] of this position into block (kv head, i, pos / 32), element pos % 32; its scale
// to the layer's vs words
__kernel void quant_vt(__global const int *x, int vt_layer, __global int *vsc, int vsmax_w) {
    QV256_HEAD
    int pos = JOB(JOB_POS);
    u32 blk = (u32)vt_layer + ((u32)v * Q2B_HD + (u32)j) * (Q2B_CTX / 32) + ((u32)pos >> 5);
    __stva8(blk * 64 + ((u32)pos & 31), q);
    i32 vs = scale_of(big);
    vsc[pos * 2 + v] = vs;
    i32 old = __ldm((u32)(vsmax_w + v));
    __stvm((u32)(vsmax_w + v), imax(old, vs));
}
// scores of one query head (aq = its 8 A blocks; kvh its kv head) over the positions 0..pos: lane = position, in
// chunks of 256; p[t] in Q.16 over 16, and the top into the scalars
__kernel void attn_scores(int kc_layer, int aq, int kvh, __global int *p) {
    int n = JOB(JOB_POS) + 1;
    int lane = LANE_INDEX;
    for (int base = 0; base < n; base += 256) {
        int t = imin(base + lane, n - 1);
        u32 kb = (u32)kc_layer + ((u32)kvh * Q2B_CTX + (u32)t) * 8;
        i32 acc = 0;
        UNROLL for (int b = 0; b < 8; b++) acc = __q8dota(acc, kb + (u32)b, (u32)(aq + b));
        i32 pt = __mulsh(acc, 1, 4);
        p[t] = pt;
        i32 top = __wmax(pt);
        i32 cur = SC(SC_TOP);
        i32 prev = (base == 0 && WARP_IN_BLOCK == 0)  /* one work-group of 8 warps: its first warp starts the fold */ ? (i32)0x80000000 : cur;
        SET_SC(SC_TOP, imax(prev, top));
    }
}
// e[t] = exp(-(top - p[t])) in Q.15, 0 past the context; the sum into the scalars
__kernel void attn_softmax(__global const int *p, __global int *e) {
    int n = JOB(JOB_POS) + 1;
    int lane = LANE_INDEX;
    i32 top = SC(SC_TOP);
    for (int base = 0; base < n; base += 256) {
        int t = base + lane;
        i32 pt = p[imin(t, n - 1)];
        i32 ev = exp_neg(top - pt);
        ev = t < n ? ev : 0;
        e[t] = ev;
        i32 s = __wsum(ev);
        i32 cur = SC(SC_SUM);
        i32 prev = (base == 0 && WARP_IN_BLOCK == 0)  /* one work-group of 8 warps: its first warp starts the fold */ ? 0 : cur;
        SET_SC(SC_SUM, prev + s);
    }
}
// g[t] = pr[t] * (vs[t] >> s), the u8 probability by the reciprocal of the sum, s from the kv head's largest vs so that g
// has 16 bits at most; quantized per 32 positions into the F blocks (af = their first block) with the residual
// g - (gq * gs >> 8) into the F2 blocks (af2): warp = a block of positions
__kernel void attn_gq(__global const int *e, int vsc_w, int kvh, int af, int af2, int vsmax_w) {
    int n = JOB(JOB_POS) + 1;
    int lane = LANE_INDEX;
    u32 sum = (u32)SC(SC_SUM);
    u32 rinv = __udiv(0u, 1u, sum);
    u32 vsmax = (u32)__ldm((u32)(vsmax_w + kvh));
    i32 vbits = 32 - __clz32(vsmax);
    i32 s = vbits > 16 ? vbits - 16 : 0;
    for (int base = 0; base < n; base += 256) {
        int t = base + lane;
        int tc = imin(t, n - 1);
        u32 pr = __umulhi((u32)e[tc] * 255u + (sum >> 1), rinv);
        u32 g = pr * ((u32)__ldm((u32)(vsc_w + tc * 2 + kvh)) >> s);
        g = t < n ? g : 0u;
        u32 gbig = (u32)__wmax((i32)g);
        i32 gr = (i32)__udiv(127u << 24, 0u, gbig);
        i32 gq = clamp127(__mulsh((i32)g, gr, 24));
        i32 gs = gbig ? scale_of(gbig) : 0;
        u32 blk = (u32)(t >> 5);
        __stva8(((u32)af + blk) * 64 + (u32)(t & 31), gq);
        __stva32(((u32)af + blk) * 64 + 32, gs);
        i32 rr = (i32)g - __mulsh(gq, gs, 8);  /* the residual in g's units */
        u32 rbig = (u32)__wmax(iabs(rr));
        i32 rrq = (i32)__udiv(127u << 24, 0u, rbig);
        i32 rq = clamp127(__mulsh(rr, rrq, 24));
        __stva8(((u32)af2 + blk) * 64 + (u32)(t & 31), rq);
        __stva32(((u32)af2 + blk) * 64 + 32, rbig ? scale_of(rbig) : 0);
    }
}
// p.V for one query head: lane = output dim, the V^T blocks of its kv head against the F and F2 blocks, 3 pairs a
// record; then the gate: oa[i] = ((acc * 257 >> (24 - s)) * sigmoid(gt[i])) >> 15
__kernel void attn_pv(int vt_layer, int kvh, int af, int af2, __global const int *gt, __global int *oa, int vsmax_w) {
    int n = JOB(JOB_POS) + 1;
    int nb = (n + 31) >> 5;
    int i = LANE_INDEX;
    u32 vtb = (u32)vt_layer + ((u32)kvh * Q2B_HD + (u32)i) * (Q2B_CTX / 32);
    u32 fb = (u32)af, fb2 = (u32)af2;
    i32 acc = 0;
    int b = 0;
    for (; b + 3 <= nb; b += 3) {
        u32 vb = vtb + (u32)b, ab = fb + (u32)b, ab2 = fb2 + (u32)b;
        acc = __q8dota(acc, vb + 0, ab + 0);
        acc = __q8dota(acc, vb + 0, ab2 + 0);
        acc = __q8dota(acc, vb + 1, ab + 1);
        acc = __q8dota(acc, vb + 1, ab2 + 1);
        acc = __q8dota(acc, vb + 2, ab + 2);
        acc = __q8dota(acc, vb + 2, ab2 + 2);
    }
    for (; b < nb; b++) {
        acc = __q8dota(acc, vtb + (u32)b, fb + (u32)b);
        acc = __q8dota(acc, vtb + (u32)b, fb2 + (u32)b);
    }
    u32 vsmax = (u32)__ldm((u32)(vsmax_w + kvh));
    i32 vbits = 32 - __clz32(vsmax);
    i32 s = vbits > 16 ? vbits - 16 : 0;
    i32 v = __mulshr(acc, 257, 24 - s);
    oa[i] = __mulsh(v, sigmoid(gt[i]), 15);
}

// ----------------------------------------------------------------------------------------------------------------
// the job: before the logits, whether this position is still prompt (skip them); after, the next token, the
// position, the turn's end. One warp, uniform.
__kernel void step_pre(void) {
    int pos = JOB(JOB_POS), n = JOB(JOB_N);
    SET_SC(SC_SKIP, pos + 1 < n ? 1 : 0);
}
__kernel void step_end(__global int *toks) {
    int pos = JOB(JOB_POS), n = JOB(JOB_N), gen = JOB(JOB_GEN), made = JOB(JOB_MADE);
    int prompt = pos + 1 < n;
    int next = SC(SC_BEST_IDX);
    int nstops = JOB(JOB_NSTOPS);
    int stop = 0;
    for (int k = 0; k < nstops; k++) stop = stop | (JOB(JOB_STOPS + k) == next ? 1 : 0);
    int newmade = prompt ? made : made + 1;
    int done = (newmade >= gen) | stop | (n + 1 >= Q2B_CTX);
    int cont = prompt ? 1 : (done ? 0 : 1);
    int newn = prompt ? n : n + 1;
    int cur = toks[n];
    toks[n] = prompt ? cur : next;  /* a reply becomes the next position's token */
    SET_SC(JOB_NEXT, next);
    SET_SC(JOB_MADE, newmade);
    SET_SC(JOB_N, newn);
    SET_SC(JOB_POS, pos + 1);
    SET_SC(JOB_CONT, cont);
    SET_SC(SC_NOEMIT, prompt);
}
