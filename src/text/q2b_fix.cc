/* The 2B in integers, in exactly the arithmetic the GPU's kernels execute (docs/machine.md): 32-bit lane values, 64-bit
 * products only inside mulsh and the Q8 block dot, uniform divisions of a 64-bit numerator by a 32-bit divisor, integer
 * square roots, tables for silu, sigmoid, softplus and exp. Host, a test reference: the kernels are checked against it
 * logit by logit, and it is checked against the exact 2B (tests/text/q2b_fix_check.cc).
 *
 * Formats: activations Q.16 in int32; a matrix input is int8 with one Q.24 scale per 32 columns (an A block); a GDN
 * q, k, v and an attention q, k, v are int8 with one Q.24 scale per vector; GDN state is int16 with one power-of-two
 * exponent per head and layer; attention probabilities are u8, folded with the value scale and quantized per 32
 * positions. Where a 32-bit result could overflow, it wraps, as the machine's registers do. */
#include "q2b_fix.h"

#include <sys/mman.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "q35.h"

namespace q2b {

namespace {

[[noreturn]] void die(const std::string &what) {
    fprintf(stderr, "q2b_fix: %s\n", what.c_str());
    exit(1);
}

std::string blk(int il, const char *suffix) { return "blk." + std::to_string(il) + "." + suffix; }

/* A constant from the file, as a table in M holds it: Q.16. */
std::vector<int32_t> read_q16(const gguf::File &f, const std::string &name) {
    const gguf::Tensor &t = f.tensor(name);
    if (t.type != gguf::T_F32) die(name + " is not F32");
    std::vector<float> v(t.nbytes / 4);
    gguf::read_at(f, t.offset, v.data(), t.nbytes);
    std::vector<int32_t> out(v.size());
    for (size_t i = 0; i < v.size(); i++) out[i] = static_cast<int32_t>(std::lrint(static_cast<double>(v[i]) * 65536.0));
    return out;
}

double half_value(uint16_t h) {
    const int sign = h >> 15, exp = h >> 10 & 31, man = h & 1023;
    double m = exp ? 1.0 + man / 1024.0 : man / 1024.0;
    double v = std::ldexp(m, (exp ? exp : 1) - 15);
    return sign ? -v : v;
}

/* Function units, on grids, made once. */
enum { FN_STEP = 8, FN_RANGE = 8, EXP_STEP = 11, EXP_RANGE = 8, SP_RANGE = 16 };
std::vector<int32_t> SILU_T, SIG_T, SOFTPLUS_T, EXP_T;

void make_tables() {
    if (!SILU_T.empty()) return;
    const int n = 2 * FN_RANGE << FN_STEP;
    SILU_T.resize(static_cast<size_t>(n));
    SIG_T.resize(static_cast<size_t>(n));
    for (int i = 0; i < n; i++) {
        const double x = (static_cast<double>(i) - n * 0.5) / static_cast<double>(1 << FN_STEP), s = 1.0 / (1.0 + std::exp(-x));
        SILU_T[static_cast<size_t>(i)] = static_cast<int32_t>(std::lrint(x * s * 65536.0));
        SIG_T[static_cast<size_t>(i)] = static_cast<int32_t>(std::lrint(s * 32768.0));
    }
    const int m = 2 * SP_RANGE << FN_STEP;
    SOFTPLUS_T.resize(static_cast<size_t>(m));
    for (int i = 0; i < m; i++) {
        const double x = (static_cast<double>(i) - m * 0.5) / static_cast<double>(1 << FN_STEP);
        SOFTPLUS_T[static_cast<size_t>(i)] = static_cast<int32_t>(std::lrint(std::log1p(std::exp(x)) * 65536.0));
    }
    const int e = EXP_RANGE << EXP_STEP;
    EXP_T.resize(static_cast<size_t>(e));
    for (int i = 0; i < e; i++) EXP_T[static_cast<size_t>(i)] = static_cast<int32_t>(std::lrint(std::exp(-i / static_cast<double>(1 << EXP_STEP)) * 32768.0));
}

/* silu of Q.16, in Q.16 */
inline int32_t silu(int32_t x) {
    if (x >= FN_RANGE << 16) return x;
    if (x < -(FN_RANGE << 16)) return 0;
    return SILU_T[static_cast<size_t>((x >> (16 - FN_STEP)) + (FN_RANGE << FN_STEP))];
}
/* sigmoid of Q.16, in Q.15 */
inline int32_t sigmoid(int32_t x) {
    if (x >= FN_RANGE << 16) return 32768;
    if (x < -(FN_RANGE << 16)) return 0;
    return SIG_T[static_cast<size_t>((x >> (16 - FN_STEP)) + (FN_RANGE << FN_STEP))];
}
inline int32_t softplus(int32_t x) {
    if (x >= SP_RANGE << 16) return x;
    if (x < -(SP_RANGE << 16)) return 0;
    return SOFTPLUS_T[static_cast<size_t>((x >> (16 - FN_STEP)) + (SP_RANGE << FN_STEP))];
}
/* exp(-d) of Q.16 d >= 0, in Q.15 */
inline int32_t exp_neg(int64_t d) {
    if (d <= 0) return 32768;
    const int64_t i = d >> (16 - EXP_STEP);
    if (i >= EXP_RANGE << EXP_STEP) return 0;
    return EXP_T[static_cast<size_t>(i)];
}

/* --- the machine's arithmetic, exactly (src/gpu/sm_ref.c) ------------------------------------------------------- */

inline int64_t shr_round(int64_t v, int s) {
    if (s <= 0) return static_cast<int64_t>(static_cast<uint64_t>(v) << -s);
    return (v + (1LL << (s - 1))) >> s;
}
/* mulsh: the signed 64-bit product, rounded and shifted, wrapped to 32 bits; a negative s is a left shift */
inline int32_t mulsh(int32_t a, int32_t b, int s) { return static_cast<int32_t>(static_cast<uint32_t>(shr_round(static_cast<int64_t>(a) * b, s))); }
/* mulhi: the high word of the unsigned product */
inline uint32_t mulhi(uint32_t a, uint32_t b) { return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) >> 32); }
/* udiv: a 64-bit numerator over a 32-bit divisor, the quotient's low word; 0 by 0 */
inline uint32_t udiv(uint64_t n, uint32_t d) { return d ? static_cast<uint32_t>(n / d) : 0u; }
/* usqrt: the guest's restoring root of a 64-bit value */
inline uint32_t isqrt(uint64_t v) {
    uint64_t r = 0, bit = 1ULL << 62;
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
    return static_cast<uint32_t>(r);
}
inline int bitlen32(uint32_t v) { return v ? 32 - __builtin_clz(v) : 0; }
inline int bitlen64(uint64_t v) { return v ? 64 - __builtin_clzll(v) : 0; }
inline int32_t clamp127(int64_t v) { return static_cast<int32_t>(std::clamp<int64_t>(v, -127, 127)); }
/* the Q.24 scale of a vector whose largest magnitude is big (Q.16): (big * 256 + 63) / 127, at most INT32_MAX */
inline int32_t scale_of(uint32_t big) {
    const uint64_t n = static_cast<uint64_t>(big) * 256 + 63;
    const uint64_t q = n / 127;
    return q > INT32_MAX ? INT32_MAX : static_cast<int32_t>(q);
}
/* 2^48 / den for a 64-bit den: den shifted into 32 bits by k, the numerator by the same k; INT32_MAX when it does not fit */
inline int32_t div48(uint64_t den) {
    if (!den) return 0;
    const uint32_t hi = static_cast<uint32_t>(den >> 32);
    const int k = bitlen32(hi);  /* 0 when den fits 32 bits */
    const uint32_t d32 = static_cast<uint32_t>(den >> k);
    const uint32_t nhi = 1u << (16 - k);  /* 2^(48 - k) = nhi : 0 */
    if (nhi >= d32) return INT32_MAX;
    return static_cast<int32_t>(udiv(static_cast<uint64_t>(nhi) << 32, d32));
}

/* int8 of a vector with one scale: value Q.24 = q * scale. Returns the scale, 0 for a zero vector. */
int32_t quantize_vec(const int32_t *v, size_t n, int8_t *q) {
    uint32_t big = 0;
    for (size_t i = 0; i < n; i++) big = std::max<uint32_t>(big, static_cast<uint32_t>(std::abs(static_cast<int64_t>(v[i]))));
    if (!big) {
        memset(q, 0, n);
        return 0;
    }
    const int32_t r = static_cast<int32_t>(udiv(127ULL << 24, big));
    for (size_t i = 0; i < n; i++) q[i] = static_cast<int8_t>(clamp127(shr_round(static_cast<int64_t>(v[i]) * r, 24)));
    return scale_of(big);
}

/* 1 / the length of an int8 vector, Q.24 of a real unit vector: length Q.8 by the root */
int32_t unit_scale(const int8_t *q, size_t n) {
    uint32_t s = 0;
    for (size_t i = 0; i < n; i++) s += static_cast<uint32_t>(static_cast<int>(q[i]) * static_cast<int>(q[i]));
    const uint32_t len8 = isqrt(static_cast<uint64_t>(s) << 16);
    return static_cast<int32_t>(udiv(1ULL << 32, len8));
}

}  // namespace

struct Fix::BlockAct {
    std::vector<int8_t> q;
    std::vector<int32_t> xs;
};

struct Fix::Layer {
    std::vector<int32_t> attn_norm, post_norm, q_norm, k_norm, conv, a, dt, ssm_norm;
    std::vector<int16_t> state;  /* VHEADS x ST x ST */
    int64_t smax[VHEADS];        /* the largest |s| of the last step: the exponent's bound */
    int se[VHEADS];              /* state value = s * 2^-se */
    std::vector<int32_t> win;    /* the last 3 projections, oldest first */
    std::vector<int8_t> kc, vc;  /* attention history: position x KVHEADS x HD */
    std::vector<int32_t> ks, vs; /* position x KVHEADS */
    int32_t vsmax[KVHEADS] = {0, 0};  /* the largest vs so far per kv head */
    const gguf::Tensor *wqkv = nullptr, *wgate = nullptr, *walpha = nullptr, *wbeta = nullptr, *wout = nullptr;
    const gguf::Tensor *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr;
    const gguf::Tensor *g = nullptr, *u = nullptr, *d = nullptr;
};

Fix::Fix(const gguf::File &f) : f_(f), layers_(LAYERS), scale_q24_(65536) {
    check_shapes(f);
    make_tables();
    void *m = mmap(nullptr, f.size, PROT_READ, MAP_SHARED, f.fd, 0);
    if (m == MAP_FAILED) die("mmap failed");
    map_ = static_cast<const uint8_t *>(m);
    for (uint32_t h = 0; h < 65536; h++) scale_q24_[h] = static_cast<int64_t>(std::llrint(half_value(static_cast<uint16_t>(h)) * 16777216.0));
    out_norm_ = read_q16(f, "output_norm.weight");
    for (int il = 0; il < LAYERS; il++) {
        Layer &L = layers_[static_cast<size_t>(il)];
        L.attn_norm = read_q16(f, blk(il, "attn_norm.weight"));
        L.post_norm = read_q16(f, blk(il, "post_attention_norm.weight"));
        L.g = &f.tensor(blk(il, "ffn_gate.weight"));
        L.u = &f.tensor(blk(il, "ffn_up.weight"));
        L.d = &f.tensor(blk(il, "ffn_down.weight"));
        if (is_full_attention(il)) {
            L.q_norm = read_q16(f, blk(il, "attn_q_norm.weight"));
            L.k_norm = read_q16(f, blk(il, "attn_k_norm.weight"));
            L.wq = &f.tensor(blk(il, "attn_q.weight"));
            L.wk = &f.tensor(blk(il, "attn_k.weight"));
            L.wv = &f.tensor(blk(il, "attn_v.weight"));
            L.wo = &f.tensor(blk(il, "attn_output.weight"));
        } else {
            L.conv = read_q16(f, blk(il, "ssm_conv1d.weight"));
            L.a = read_q16(f, blk(il, "ssm_a"));
            L.dt = read_q16(f, blk(il, "ssm_dt.bias"));
            L.ssm_norm = read_q16(f, blk(il, "ssm_norm.weight"));
            L.wqkv = &f.tensor(blk(il, "attn_qkv.weight"));
            L.wgate = &f.tensor(blk(il, "attn_gate.weight"));
            L.walpha = &f.tensor(blk(il, "ssm_alpha.weight"));
            L.wbeta = &f.tensor(blk(il, "ssm_beta.weight"));
            L.wout = &f.tensor(blk(il, "ssm_out.weight"));
            L.state.resize(VHEADS * ST * ST);
            L.win.resize(3 * QKV);
        }
    }
    reset();
}

Fix::~Fix() {
    if (map_) munmap(const_cast<uint8_t *>(map_), f_.size);
}

Fix::GdnState Fix::gdn_state(int il) const {
    const Layer &L = layers_[static_cast<size_t>(il)];
    return GdnState{L.state.data(), L.se, L.smax, L.win.data()};
}

Fix::AttnState Fix::attn_state(int il) const {
    const Layer &L = layers_[static_cast<size_t>(il)];
    return AttnState{L.kc.data(), L.vc.data(), L.ks.data(), L.vs.data(), L.ks.size()};
}

void Fix::reset() {
    pos_ = 0;
    for (Layer &L : layers_) {
        std::fill(L.state.begin(), L.state.end(), static_cast<int16_t>(0));
        for (int &s : L.se) s = 15;
        for (int64_t &s : L.smax) s = 0;
        std::fill(L.win.begin(), L.win.end(), 0);
        L.kc.clear();
        L.vc.clear();
        L.ks.clear();
        L.vs.clear();
        L.vsmax[0] = L.vsmax[1] = 0;
    }
}

/* one A block per 32 elements: the int8 lanes and the Q.24 scale */
void Fix::quantize_blocks(const int32_t *v, size_t n, BlockAct &a) {
    a.q.resize(n);
    a.xs.resize(n / 32);
    for (size_t b = 0; b < n / 32; b++) a.xs[b] = quantize_vec(v + b * 32, 32, &a.q[b * 32]);
    cost_.quant += n / 2 + (n / 32) * 40;
}

/* y[r] = sum over the blocks of the q8dot of W's block against the input's: t = (inner * mw + 2^15) >> 16,
 * u = (t * xs + 2^15) >> 16, the sum wrapping in 32 bits as the lane register does */
void Fix::gemv(const gguf::Tensor &t, uint64_t rows, const BlockAct &x, int32_t *y) {
    const uint64_t cols = t.dims[0], row_bytes = gguf::row_bytes(gguf::T_Q8_0, cols), blocks = cols / 32;
    const uint8_t *base = map_ + t.offset;
    const int64_t *tab = scale_q24_.data();
    const int8_t *xq = x.q.data();
    const int32_t *xs = x.xs.data();
    int64_t most_y = 0, most_t = 0;
    uint64_t wrap_acc = 0;
    for (uint64_t r = 0; r < rows; r++) {
        const uint8_t *p = base + r * row_bytes;
        uint32_t acc = 0;
        int64_t acc64 = 0;
        for (uint64_t b = 0; b < blocks; b++, p += 34) {
            uint16_t h;
            memcpy(&h, p, 2);
            const int8_t *q = reinterpret_cast<const int8_t *>(p + 2);
            const int8_t *xb = xq + b * 32;
            int32_t inner = 0;
            for (int i = 0; i < 32; i++) inner += static_cast<int32_t>(q[i]) * static_cast<int32_t>(xb[i]);
            const int64_t tv = shr_round(static_cast<int64_t>(inner) * tab[h], 16);
            most_t = std::max<int64_t>(most_t, std::llabs(static_cast<long long>(tv)));
            const int64_t u = shr_round(tv * xs[b], 16);
            acc += static_cast<uint32_t>(u);
            acc64 += u;
        }
        most_y = std::max<int64_t>(most_y, std::llabs(static_cast<long long>(acc64)));
        if (acc64 != static_cast<int32_t>(acc)) wrap_acc++;
        y[r] = static_cast<int32_t>(acc);
    }
    widths_.y = std::max(widths_.y, most_y);
    widths_.t = std::max(widths_.t, most_t);
    widths_.wrap_acc += wrap_acc;
    cost_.dot += rows * blocks * 2;
}

/* x normalized by its rms, times the weights w, Q.16: rs = 2^48 / (rms8 * scale) is Q.16 of 1 / rms. */
void Fix::norm(const int32_t *x, size_t n, const int32_t *w, int32_t *out) {
    std::vector<int8_t> q(n);
    const int32_t scale = quantize_vec(x, n, q.data());
    uint32_t s = 0;
    for (size_t i = 0; i < n; i++) s += static_cast<uint32_t>(static_cast<int>(q[i]) * static_cast<int>(q[i]));
    const uint32_t rms8 = isqrt(udiv(static_cast<uint64_t>(s) << 16, static_cast<uint32_t>(n)));
    const int32_t rs = div48(static_cast<uint64_t>(rms8) * static_cast<uint32_t>(scale));
    for (size_t i = 0; i < n; i++) out[i] = mulsh(mulsh(x[i], rs, 16), w ? w[i] : 65536, 16);
    cost_.rms += 2 + 120;
    cost_.elem += 2 * 2 * n;
}

void Fix::ffn(int il, const BlockAct &h, int32_t *out) {
    Layer &L = layers_[static_cast<size_t>(il)];
    std::vector<int32_t> g(FF), u(FF), x(FF);
    gemv(*L.g, FF, h, g.data());
    gemv(*L.u, FF, h, u.data());
    for (int i = 0; i < FF; i++) x[static_cast<size_t>(i)] = mulsh(silu(g[static_cast<size_t>(i)]), u[static_cast<size_t>(i)], 16);
    cost_.elem += 4 * FF;
    BlockAct a;
    quantize_blocks(x.data(), FF, a);
    gemv(*L.d, D, a, out);
    if (tap_) (*tap_)("ffn_out", il, out, D);
}

void Fix::layer_gdn(int il, const BlockAct &h, int32_t *out) {
    Layer &L = layers_[static_cast<size_t>(il)];
    std::vector<int32_t> qkv(QKV), z(VAL_DIM), conv(QKV);
    int32_t alpha[VHEADS], beta_raw[VHEADS];
    gemv(*L.wqkv, QKV, h, qkv.data());
    gemv(*L.wgate, VAL_DIM, h, z.data());
    gemv(*L.walpha, VHEADS, h, alpha);
    gemv(*L.wbeta, VHEADS, h, beta_raw);
    if (tap_) (*tap_)("linear_attn_qkv_mixed", il, qkv.data(), QKV);
    if (tap_) (*tap_)("z", il, z.data(), VAL_DIM);
    int32_t beta[VHEADS], decay[VHEADS];
    for (int i = 0; i < VHEADS; i++) {
        beta[i] = sigmoid(beta_raw[i]);
        /* gate = softplus(alpha + dt) * a, a < 0: decay = exp(gate) */
        const int32_t gate = mulsh(softplus(alpha[i] + L.dt[static_cast<size_t>(i)]), L.a[static_cast<size_t>(i)], 16);
        decay[i] = exp_neg(-static_cast<int64_t>(gate));
    }
    cost_.elem += 2 * 3 * VHEADS;
    /* the causal conv: three remembered projections and this one, oldest first; each tap a mulsh, the sum wrapping */
    int32_t *win = L.win.data();
    for (int c = 0; c < QKV; c++) {
        const int32_t *w = &L.conv[static_cast<size_t>(c) * CONV];
        uint32_t acc = 0;
        for (int t = 0; t < 3; t++) acc += static_cast<uint32_t>(mulsh(w[t], win[static_cast<size_t>(t) * QKV + static_cast<size_t>(c)], 16));
        acc += static_cast<uint32_t>(mulsh(w[3], qkv[static_cast<size_t>(c)], 16));
        conv[static_cast<size_t>(c)] = silu(static_cast<int32_t>(acc));
    }
    cost_.elem += (2 * CONV + 2) * QKV;
    if (tap_) (*tap_)("conv_output_silu", il, conv.data(), QKV);
    memmove(win, win + QKV, 2 * QKV * sizeof(int32_t));
    memcpy(win + 2 * QKV, qkv.data(), QKV * sizeof(int32_t));
    cost_.quant += 3 * QKV / 2;
    /* per key head, q and k in int8, and their scales to real unit-length vectors */
    std::vector<int8_t> qi(KEY_DIM), ki(KEY_DIM), vi(VAL_DIM);
    int32_t qs[KHEADS], ks[KHEADS];
    int32_t vs[VHEADS];
    for (int kh = 0; kh < KHEADS; kh++) {
        quantize_vec(&conv[static_cast<size_t>(kh) * ST], ST, &qi[static_cast<size_t>(kh) * ST]);
        quantize_vec(&conv[static_cast<size_t>(KEY_DIM + kh * ST)], ST, &ki[static_cast<size_t>(kh) * ST]);
        /* q gets a further 1 / sqrt(ST): 2896 / 2^15, as the high word of a product */
        const uint32_t us = static_cast<uint32_t>(unit_scale(&qi[static_cast<size_t>(kh) * ST], ST));
        qs[kh] = static_cast<int32_t>(mulhi(us, 2896u << 17));
        ks[kh] = unit_scale(&ki[static_cast<size_t>(kh) * ST], ST);
    }
    for (int hd = 0; hd < VHEADS; hd++) vs[hd] = quantize_vec(&conv[static_cast<size_t>(2 * KEY_DIM + hd * ST)], ST, &vi[static_cast<size_t>(hd) * ST]);
    cost_.quant += (2 * KHEADS + VHEADS) * (ST / 2 + 40);
    cost_.rms += 2 * KHEADS * 120;
    if (tap_) {  /* the head's inputs, for the stage probe */
        std::vector<int32_t> t(static_cast<size_t>(VAL_DIM));
        for (int i = 0; i < KEY_DIM; i++) t[static_cast<size_t>(i)] = qi[static_cast<size_t>(i)];
        (*tap_)("gdn_qi", il, t.data(), KEY_DIM);
        for (int i = 0; i < KEY_DIM; i++) t[static_cast<size_t>(i)] = ki[static_cast<size_t>(i)];
        (*tap_)("gdn_ki", il, t.data(), KEY_DIM);
        for (int i = 0; i < VAL_DIM; i++) t[static_cast<size_t>(i)] = vi[static_cast<size_t>(i)];
        (*tap_)("gdn_vi", il, t.data(), VAL_DIM);
        (*tap_)("gdn_qs", il, qs, KHEADS);
        (*tap_)("gdn_ks", il, ks, KHEADS);
        (*tap_)("gdn_vs", il, vs, VHEADS);
        (*tap_)("gdn_decay", il, decay, VHEADS);
        (*tap_)("gdn_beta", il, beta, VHEADS);
    }
    std::vector<int32_t> kreal_all(static_cast<size_t>(VHEADS) * ST), o_all(static_cast<size_t>(VHEADS) * ST), dbg_all(static_cast<size_t>(VHEADS) * 8),
        on_all(static_cast<size_t>(VHEADS) * ST), sz_all(static_cast<size_t>(VHEADS) * ST);
    std::vector<int32_t> mix(VAL_DIM);
    int64_t most_state = 0;
    for (int hd = 0; hd < VHEADS; hd++) {
        int16_t *S = &L.state[static_cast<size_t>(hd) * ST * ST];
        const int kh = hd % KHEADS;
        const int8_t *kk = &ki[static_cast<size_t>(kh) * ST], *qq = &qi[static_cast<size_t>(kh) * ST], *vv = &vi[static_cast<size_t>(hd) * ST];
        const int32_t e = decay[hd], b = beta[hd], kscale = ks[kh], qscale = qs[kh];
        const int se = L.se[hd];
        /* the state read with the key: sdot per element, wrapping */
        int32_t a[ST] = {0};
        for (int i = 0; i < ST; i++) {
            const int16_t *row = S + i * ST;
            const int32_t kv = kk[i];
            for (int j = 0; j < ST; j++) a[j] = static_cast<int32_t>(static_cast<uint32_t>(a[j]) + static_cast<uint32_t>(row[j] * kv));
        }
        const int32_t m = mulsh(e, kscale, 15);
        int32_t delta[ST];  /* Q.24 */
        uint32_t maxdelta = 0;
        for (int j = 0; j < ST; j++) {
            const int64_t kv64 = shr_round(static_cast<int64_t>(a[j]) * m, se);
            const int32_t kv24 = static_cast<int32_t>(static_cast<uint32_t>(kv64));
            widths_.wrap_kv24 += kv64 != kv24;
            const int64_t v64 = static_cast<int64_t>(vv[j]) * vs[hd];
            const int32_t v24 = static_cast<int32_t>(static_cast<uint32_t>(v64));
            widths_.wrap_v24 += v64 != v24;
            delta[j] = mulsh(static_cast<int32_t>(static_cast<uint32_t>(v24) - static_cast<uint32_t>(kv24)), b, 15);
            maxdelta = std::max<uint32_t>(maxdelta, static_cast<uint32_t>(std::abs(delta[j])));
        }
        /* the window: 15 bits and a sign, one power of two per head and layer, from a bound the kernel already has:
         * the old term at most smax * e, the new one at most 127 * kscale * maxdelta. The bound is 64 bits: the kernel
         * carries it as two words */
        const uint64_t bound = static_cast<uint64_t>(static_cast<uint32_t>(mulsh(static_cast<int32_t>(L.smax[hd]), e, se - 17)))
                               + static_cast<uint64_t>(shr_round(static_cast<int64_t>(127 * kscale) * maxdelta, 16));
        const int nse = bound ? 47 - bitlen64(bound) : se;
        /* S' = e S + k delta, in Q.32, then the window */
        int32_t kreal[ST];
        for (int i = 0; i < ST; i++) kreal[i] = static_cast<int32_t>(static_cast<uint32_t>(kk[i]) * static_cast<uint32_t>(kscale));
        for (int i = 0; i < ST; i++) kreal_all[static_cast<size_t>(hd * ST + i)] = kreal[i];
        uint32_t smax = 0;
        for (int i = 0; i < ST; i++) {
            int16_t *row = S + i * ST;
            for (int j = 0; j < ST; j++) {
                const int64_t val = shr_round(static_cast<int64_t>(row[j]) * e, se + 15 - 32) + shr_round(static_cast<int64_t>(kreal[i]) * delta[j], 16);
                most_state = std::max<int64_t>(most_state, std::llabs(static_cast<long long>(val)));
                int64_t sv = shr_round(val, 32 - nse);
                sv = std::clamp<int64_t>(sv, -32767, 32767);
                row[j] = static_cast<int16_t>(sv);
                smax = std::max<uint32_t>(smax, static_cast<uint32_t>(sv < 0 ? -sv : sv));
            }
        }
        /* the state read with the query */
        int32_t o[ST] = {0};
        for (int i = 0; i < ST; i++) {
            const int16_t *row = S + i * ST;
            const int32_t qv = qq[i];
            for (int j = 0; j < ST; j++) o[j] = static_cast<int32_t>(static_cast<uint32_t>(o[j]) + static_cast<uint32_t>(row[j] * qv));
        }
        for (int j = 0; j < ST; j++) o_all[static_cast<size_t>(hd * ST + j)] = o[j];
        L.se[hd] = nse;
        L.smax[hd] = smax;
        /* The norm is 1 / sqrt(mean(o^2) + eps) with eps = 1e-6: o is tiny in real units, about the size of sqrt(eps), so
         * eps matters and the norm must see o's real magnitude. real o = raw * qs * 2^-(nse + 24). */
        int8_t oq[ST];
        const int32_t oscale = quantize_vec(o, ST, oq);
        uint32_t s2 = 0;
        for (int j = 0; j < ST; j++) s2 += static_cast<uint32_t>(static_cast<int>(oq[j]) * static_cast<int>(oq[j]));
        const int32_t isq = static_cast<int32_t>(isqrt(static_cast<uint64_t>(s2) << 9));    /* (s2 << 16) / ST */
        const int64_t rms64 = shr_round(static_cast<int64_t>(mulsh(isq, oscale, 16)) * qscale, nse);
        const int32_t rms24 = static_cast<int32_t>(static_cast<uint32_t>(rms64));             /* real rms, Q.24 */
        widths_.wrap_rms24 += rms64 != rms24;
        const uint32_t root24 = isqrt(static_cast<uint64_t>(static_cast<uint32_t>(rms24)) * static_cast<uint32_t>(rms24) + 281474977ULL);  /* 1e-6 is Q.48 281474977 */
        const int32_t f8 = static_cast<int32_t>(udiv(static_cast<uint64_t>(static_cast<uint32_t>(qscale)) << 20, root24));  /* (qscale << 28) / root24, over 2^8 */
        {
            uint32_t obig = 0;
            for (int j = 0; j < ST; j++) obig = std::max<uint32_t>(obig, static_cast<uint32_t>(std::abs(o[j])));
            int32_t *dd = &dbg_all[static_cast<size_t>(hd) * 8];
            dd[0] = static_cast<int32_t>(obig);
            dd[1] = static_cast<int32_t>(s2);
            dd[2] = isq;
            dd[3] = rms24;
            dd[4] = static_cast<int32_t>(root24);
            dd[5] = f8;
            dd[6] = oscale;
            dd[7] = nse;
        }
        int32_t *mm = &mix[static_cast<size_t>(hd) * ST];
        const int32_t *zz = &z[static_cast<size_t>(hd) * ST];
        for (int j = 0; j < ST; j++) {
            const int32_t on = mulsh(mulsh(o[j], f8, nse + 4), L.ssm_norm[static_cast<size_t>(j)], 16);
            on_all[static_cast<size_t>(hd * ST + j)] = on;
            sz_all[static_cast<size_t>(hd * ST + j)] = silu(zz[j]);
            mm[j] = mulsh(on, silu(zz[j]), 16);
        }
    }
    if (tap_) {  /* the head's state words, for the stage probe */
        int32_t se32[VHEADS], smax32[VHEADS];
        for (int hd = 0; hd < VHEADS; hd++) {
            se32[hd] = L.se[hd];
            smax32[hd] = static_cast<int32_t>(L.smax[hd]);
        }
        (*tap_)("gdn_kreal", il, kreal_all.data(), VHEADS * ST);
        (*tap_)("gdn_o", il, o_all.data(), VHEADS * ST);
        (*tap_)("gdn_dbg", il, dbg_all.data(), VHEADS * 8);
        (*tap_)("gdn_on", il, on_all.data(), VHEADS * ST);
        (*tap_)("gdn_sz", il, sz_all.data(), VHEADS * ST);
        (*tap_)("gdn_se", il, se32, VHEADS);
        (*tap_)("gdn_smax", il, smax32, VHEADS);
    }
    widths_.state_q32 = std::max(widths_.state_q32, most_state);
    cost_.state += VHEADS * (ST * 4 + 2 * 2 * ST * ST + ST * 4);
    cost_.rms += VHEADS * 122;
    cost_.elem += VHEADS * ST * (2 * 2 + 2 + 2);
    BlockAct a;
    quantize_blocks(mix.data(), VAL_DIM, a);
    if (tap_) {  /* the A blocks of mix, for the stage probe: int8 lanes and Q.24 scales */
        std::vector<int32_t> t(static_cast<size_t>(VAL_DIM));
        for (int i = 0; i < VAL_DIM; i++) t[static_cast<size_t>(i)] = a.q[static_cast<size_t>(i)];
        (*tap_)("mix_q", il, t.data(), VAL_DIM);
        (*tap_)("mix_xs", il, a.xs.data(), VAL_DIM / 32);
    }
    if (tap_) (*tap_)("final_output", il, mix.data(), VAL_DIM);
    gemv(*L.wout, D, a, out);
    if (tap_) (*tap_)("linear_attn_out", il, out, D);
}

void Fix::layer_full(int il, const BlockAct &h, int32_t *out) {
    Layer &L = layers_[static_cast<size_t>(il)];
    std::vector<int32_t> qg(2 * QHEADS * HD), kk(KVHEADS * HD), vv(KVHEADS * HD);
    gemv(*L.wq, 2 * QHEADS * HD, h, qg.data());
    gemv(*L.wk, KVHEADS * HD, h, kk.data());
    gemv(*L.wv, KVHEADS * HD, h, vv.data());
    /* cos and sin of this position, Q.15: the RoPE table in M */
    float cf[ROT / 2], sf[ROT / 2];
    q35::rope_row(pos_, cf, sf);
    int32_t c[ROT / 2], s[ROT / 2];
    for (int i = 0; i < ROT / 2; i++) {
        c[i] = static_cast<int32_t>(std::lrint(static_cast<double>(cf[i]) * 32768.0));
        s[i] = static_cast<int32_t>(std::lrint(static_cast<double>(sf[i]) * 32768.0));
    }
    const auto rotate = [&](int32_t *x) {
        for (int i = 0; i < ROT / 2; i++) {
            const int32_t a = x[i], b = x[i + ROT / 2];
            x[i] = static_cast<int32_t>(static_cast<uint32_t>(mulsh(a, c[i], 15)) - static_cast<uint32_t>(mulsh(b, s[i], 15)));
            x[i + ROT / 2] = static_cast<int32_t>(static_cast<uint32_t>(mulsh(a, s[i], 15)) + static_cast<uint32_t>(mulsh(b, c[i], 15)));
        }
    };
    std::vector<int32_t> q(QHEADS * HD);
    std::vector<int8_t> qi(QHEADS * HD);
    int32_t qscale[QHEADS];
    for (int hq = 0; hq < QHEADS; hq++) {
        int32_t *dst = &q[static_cast<size_t>(hq) * HD];
        norm(&qg[static_cast<size_t>(hq) * 2 * HD], HD, L.q_norm.data(), dst);
        rotate(dst);
        qscale[hq] = quantize_vec(dst, HD, &qi[static_cast<size_t>(hq) * HD]);
    }
    if (tap_) {  /* the queries after norm + RoPE, their int8 blocks and scales (one scale per head, repeated per block) */
        (*tap_)("attn_query", il, q.data(), QHEADS * HD);
        std::vector<int32_t> t(static_cast<size_t>(QHEADS) * HD);
        for (int i = 0; i < QHEADS * HD; i++) t[static_cast<size_t>(i)] = qi[static_cast<size_t>(i)];
        (*tap_)("aq_q", il, t.data(), QHEADS * HD);
        int32_t qsx[QHEADS * HD / 32];
        for (int hq = 0; hq < QHEADS; hq++)
            for (int b = 0; b < HD / 32; b++) qsx[hq * (HD / 32) + b] = qscale[hq];
        (*tap_)("aq_xs", il, qsx, QHEADS * HD / 32);
    }
    std::vector<int32_t> kvec(static_cast<size_t>(KVHEADS) * HD);
    const size_t base8 = L.kc.size();
    L.kc.resize(base8 + KVHEADS * HD);
    L.vc.resize(base8 + KVHEADS * HD);
    for (int hk = 0; hk < KVHEADS; hk++) {
        int32_t *x = &kk[static_cast<size_t>(hk) * HD];
        std::vector<int32_t> tmp(HD);
        norm(x, HD, L.k_norm.data(), tmp.data());
        rotate(tmp.data());
        std::copy(tmp.begin(), tmp.end(), kvec.begin() + static_cast<long>(hk) * HD);
        L.ks.push_back(quantize_vec(tmp.data(), HD, &L.kc[base8 + static_cast<size_t>(hk) * HD]));
        L.vs.push_back(quantize_vec(&vv[static_cast<size_t>(hk) * HD], HD, &L.vc[base8 + static_cast<size_t>(hk) * HD]));
        L.vsmax[hk] = std::max(L.vsmax[hk], L.vs.back());
    }
    if (tap_) {  /* the keys after norm + RoPE, this position's key and value blocks per kv head, the value scales */
        (*tap_)("attn_k", il, kvec.data(), KVHEADS * HD);
        std::vector<int32_t> t(static_cast<size_t>(HD));
        int32_t sx[HD / 32], vs2[KVHEADS], vsm[KVHEADS];
        for (int hk = 0; hk < KVHEADS; hk++) {
            for (int i = 0; i < HD; i++) t[static_cast<size_t>(i)] = L.kc[base8 + static_cast<size_t>(hk) * HD + static_cast<size_t>(i)];
            (*tap_)(hk == 0 ? "kc0_q" : "kc1_q", il, t.data(), HD);
            for (int b = 0; b < HD / 32; b++) sx[b] = L.ks[base8 / HD + static_cast<size_t>(hk)];
            (*tap_)(hk == 0 ? "kc0_xs" : "kc1_xs", il, sx, HD / 32);
            for (int i = 0; i < HD; i++) t[static_cast<size_t>(i)] = L.vc[base8 + static_cast<size_t>(hk) * HD + static_cast<size_t>(i)];
            (*tap_)(hk == 0 ? "vt0_q" : "vt1_q", il, t.data(), HD);
            vs2[hk] = L.vs[base8 / HD + static_cast<size_t>(hk)];
            vsm[hk] = L.vsmax[hk];
        }
        (*tap_)("attn_vs", il, vs2, KVHEADS);
        (*tap_)("attn_vsmax", il, vsm, KVHEADS);
    }
    cost_.elem += (QHEADS + KVHEADS) * 2 * ROT;
    cost_.quant += (QHEADS + 2 * KVHEADS) * (HD / 2 + 40);
    const int n = static_cast<int>(L.kc.size() / (KVHEADS * HD));
    const int nb = (n + 31) / 32;  /* blocks of 32 positions */
    std::vector<int32_t> o(QHEADS * HD);
    for (int hq = 0; hq < QHEADS; hq++) {
        const int kvh = hq * KVHEADS / QHEADS;
        const int8_t *qq = &qi[static_cast<size_t>(hq) * HD];
        /* scores: a q8dot per block of 32 dims, the key's scale then the query's, the sum of the 8 blocks over 16 */
        std::vector<int32_t> p(static_cast<size_t>(nb) * 32, 0);
        int32_t top = INT32_MIN;
        for (int t = 0; t < n; t++) {
            const int8_t *key = &L.kc[(static_cast<size_t>(t) * KVHEADS + static_cast<size_t>(kvh)) * HD];
            const int64_t kst = L.ks[static_cast<size_t>(t) * KVHEADS + static_cast<size_t>(kvh)];
            uint32_t acc = 0;
            for (int b = 0; b < HD / 32; b++) {
                int32_t inner = 0;
                for (int i = 0; i < 32; i++) inner += static_cast<int32_t>(qq[32 * b + i]) * static_cast<int32_t>(key[32 * b + i]);
                const int64_t tv = shr_round(static_cast<int64_t>(inner) * kst, 16);
                acc += static_cast<uint32_t>(shr_round(tv * qscale[hq], 16));
            }
            p[static_cast<size_t>(t)] = mulsh(static_cast<int32_t>(acc), 1, 4);
            top = std::max(top, p[static_cast<size_t>(t)]);
        }
        if (tap_ && hq == QHEADS - 1) {  /* the last head's scores: what the machine's M_P / SC_TOP hold after the loop */
            (*tap_)("attn_p", il, p.data(), static_cast<size_t>(n));
            (*tap_)("attn_top", il, &top, 1);
        }
        /* softmax: exp of the distance below the top; u8 probabilities by the reciprocal of the sum */
        std::vector<int32_t> e(static_cast<size_t>(nb) * 32, 0);
        uint32_t sum = 0;
        for (int t = 0; t < n; t++) {
            e[static_cast<size_t>(t)] = exp_neg(static_cast<int64_t>(top) - p[static_cast<size_t>(t)]);
            sum += static_cast<uint32_t>(e[static_cast<size_t>(t)]);
        }
        const uint32_t rinv = udiv(1ULL << 32, sum);
        if (tap_ && hq == QHEADS - 1) {
            (*tap_)("attn_e", il, e.data(), static_cast<size_t>(n));
            const int32_t sum32 = static_cast<int32_t>(sum);
            (*tap_)("attn_sum", il, &sum32, 1);
        }
        /* g = pr * (vs >> s): the probability with the value's scale, s from the largest vs of this kv head so far, so
         * that g has 16 bits at most (the sum over the positions then fits a lane register); quantized per 32 positions
         * as an A block, and the residual g - (gq * gs >> 8) quantized again as a second block: two q8dots a block */
        const int32_t vsmax = L.vsmax[kvh];
        const int s = bitlen32(static_cast<uint32_t>(vsmax)) > 16 ? bitlen32(static_cast<uint32_t>(vsmax)) - 16 : 0;
        std::vector<int8_t> gq(static_cast<size_t>(nb) * 32, 0), rq(static_cast<size_t>(nb) * 32, 0);
        std::vector<int32_t> gs(static_cast<size_t>(nb), 0), rs(static_cast<size_t>(nb), 0);
        for (int b = 0; b < nb; b++) {
            int32_t g[32] = {0}, r[32] = {0};
            uint32_t gbig = 0, rbig = 0;
            for (int k = 0; k < 32; k++) {
                const int t = 32 * b + k;
                if (t >= n) break;
                const uint32_t pr = mulhi(static_cast<uint32_t>(e[static_cast<size_t>(t)]) * 255u + sum / 2, rinv);
                g[k] = static_cast<int32_t>(pr * (static_cast<uint32_t>(L.vs[static_cast<size_t>(t) * KVHEADS + static_cast<size_t>(kvh)]) >> s));
                gbig = std::max(gbig, static_cast<uint32_t>(g[k]));
            }
            const int32_t gr = static_cast<int32_t>(udiv(127ULL << 24, gbig));
            const int32_t gsc = gbig ? scale_of(gbig) : 0;
            for (int k = 0; k < 32; k++) {
                const int32_t q = clamp127(shr_round(static_cast<int64_t>(g[k]) * gr, 24));
                gq[static_cast<size_t>(32 * b + k)] = static_cast<int8_t>(q);
                r[k] = static_cast<int32_t>(static_cast<uint32_t>(g[k]) - static_cast<uint32_t>(mulsh(q, gsc, 8)));
                rbig = std::max<uint32_t>(rbig, static_cast<uint32_t>(std::abs(r[k])));
            }
            const int32_t rr = static_cast<int32_t>(udiv(127ULL << 24, rbig));
            for (int k = 0; k < 32; k++) rq[static_cast<size_t>(32 * b + k)] = static_cast<int8_t>(clamp127(shr_round(static_cast<int64_t>(r[k]) * rr, 24)));
            gs[static_cast<size_t>(b)] = gsc;
            rs[static_cast<size_t>(b)] = rbig ? scale_of(rbig) : 0;
        }
        if (tap_ && hq == QHEADS - 1) {  /* the last head's probability blocks and their residuals */
            std::vector<int32_t> t(static_cast<size_t>(n));
            for (int i = 0; i < n; i++) t[static_cast<size_t>(i)] = gq[static_cast<size_t>(i)];
            (*tap_)("f_q", il, t.data(), static_cast<size_t>(n));
            (*tap_)("f_xs", il, gs.data(), static_cast<size_t>(nb));
            for (int i = 0; i < n; i++) t[static_cast<size_t>(i)] = rq[static_cast<size_t>(i)];
            (*tap_)("f2_q", il, t.data(), static_cast<size_t>(n));
            (*tap_)("f2_xs", il, rs.data(), static_cast<size_t>(nb));
        }
        /* p.V: two q8dots per block of 32 positions over V transposed (scale 1.0 as Q.24: inner << 8), the block's g
         * scale and its residual's */
        int32_t *oo = &o[static_cast<size_t>(hq) * HD];
        const int32_t *gt = &qg[static_cast<size_t>(hq) * 2 * HD + HD];
        for (int i = 0; i < HD; i++) {
            uint32_t acc = 0;
            for (int b = 0; b < nb; b++) {
                int32_t inner = 0, inner_r = 0;
                for (int k = 0; k < 32; k++) {
                    const int t = 32 * b + k;
                    if (t >= n) break;
                    const int32_t vt = L.vc[(static_cast<size_t>(t) * KVHEADS + static_cast<size_t>(kvh)) * HD + static_cast<size_t>(i)];
                    inner += vt * gq[static_cast<size_t>(t)];
                    inner_r += vt * rq[static_cast<size_t>(t)];
                }
                acc += static_cast<uint32_t>(shr_round((static_cast<int64_t>(inner) << 8) * gs[static_cast<size_t>(b)], 16));
                acc += static_cast<uint32_t>(shr_round((static_cast<int64_t>(inner_r) << 8) * rs[static_cast<size_t>(b)], 16));
            }
            /* acc is (sum of pr vs v) over 2^s times 255: Q.16 by 257 / 2^(24 - s) */
            const int32_t v = mulsh(static_cast<int32_t>(acc), 257, 24 - s);
            oo[i] = mulsh(v, sigmoid(gt[i]), 15);
        }
    }
    const uint64_t un = static_cast<uint64_t>(n);
    cost_.attn += QHEADS * (8 * un + un + 2 + HD * 2 * ((un + 31) / 32));
    cost_.elem += QHEADS * HD * 4;
    BlockAct a;
    quantize_blocks(o.data(), QHEADS * HD, a);
    if (tap_) (*tap_)("attn_gated", il, o.data(), QHEADS * HD);
    gemv(*L.wo, D, a, out);
    if (tap_) (*tap_)("attn_output", il, out, D);
}

void Fix::step(int32_t token, int32_t *logits, const FixTap *tap) {
    tap_ = tap;
    const gguf::Tensor &emb = f_.tensor("token_embd.weight");
    const uint64_t row = gguf::row_bytes(gguf::T_Q8_0, D);
    const uint8_t *raw = map_ + emb.offset + static_cast<uint64_t>(token) * row;
    int32_t x[D], h[D], y[D];
    for (int b = 0; b < D / 32; b++) {
        uint16_t hh;
        memcpy(&hh, raw + static_cast<size_t>(b) * 34, 2);
        const int8_t *q = reinterpret_cast<const int8_t *>(raw + static_cast<size_t>(b) * 34 + 2);
        const int32_t mw = static_cast<int32_t>(scale_q24_[hh]);  /* W32: the block's scale as Q.24 */
        for (int i = 0; i < 32; i++) x[b * 32 + i] = mulsh(q[i], mw, 8);
    }
    cost_.elem += 2 * D;
    if (tap_) (*tap_)("inp_embd", -1, x, D);
    BlockAct a;
    for (int il = 0; il < LAYERS; il++) {
        Layer &L = layers_[static_cast<size_t>(il)];
        norm(x, D, L.attn_norm.data(), h);
        if (tap_) (*tap_)("attn_norm", il, h, D);
        quantize_blocks(h, D, a);
        if (is_full_attention(il)) layer_full(il, a, y);
        else layer_gdn(il, a, y);
        for (int i = 0; i < D; i++) x[i] = static_cast<int32_t>(static_cast<uint32_t>(x[i]) + static_cast<uint32_t>(y[i]));
        if (tap_) (*tap_)("attn_residual", il, x, D);
        cost_.elem += 2 * D;
        norm(x, D, L.post_norm.data(), h);
        if (tap_) (*tap_)("attn_post_norm", il, h, D);
        quantize_blocks(h, D, a);
        ffn(il, a, y);
        for (int i = 0; i < D; i++) x[i] = static_cast<int32_t>(static_cast<uint32_t>(x[i]) + static_cast<uint32_t>(y[i]));
        if (tap_) (*tap_)("l_out", il, x, D);
        cost_.elem += 2 * D;
    }
    norm(x, D, out_norm_.data(), h);
    if (tap_) (*tap_)("result_norm", -1, h, D);
    quantize_blocks(h, D, a);
    gemv(emb, VOCAB, a, logits);
    if (tap_) (*tap_)("result_output", -1, logits, VOCAB);
    cost_.positions++;
    pos_++;
}

}  // namespace q2b

namespace q2b {

void fix_make_tables() { make_tables(); }

const std::vector<int32_t> &fix_table(int which) {
    make_tables();
    return which == 0 ? SILU_T : which == 1 ? SIG_T : which == 2 ? SOFTPLUS_T : EXP_T;
}

const std::vector<int64_t> &fix_scale_table() {
    static std::vector<int64_t> t;
    if (t.empty()) {
        t.resize(65536);
        for (uint32_t h = 0; h < 65536; h++) t[h] = static_cast<int64_t>(std::llrint(half_value(static_cast<uint16_t>(h)) * 16777216.0));
    }
    return t;
}

std::vector<int32_t> fix_read_q16(const gguf::File &f, const std::string &name) { return read_q16(f, name); }

}  // namespace q2b
