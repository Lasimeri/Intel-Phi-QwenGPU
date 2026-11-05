#include <sys/mman.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "elem.h"
#include "q2b.h"
#include "q35.h"

namespace q2b {

namespace {

static_assert(static_cast<int>(ROT) == static_cast<int>(q35::ROT), "the rotation table is shared with the 35B");

[[noreturn]] void die(const std::string &what) {
    fprintf(stderr, "q2b: %s\n", what.c_str());
    exit(1);
}

std::string blk(int il, const char *suffix) { return "blk." + std::to_string(il) + "." + suffix; }

void expect_tensor(const gguf::File &f, const std::string &name, uint32_t type, uint64_t d0, uint64_t d1) {
    const gguf::Tensor &t = f.tensor(name);
    if (t.type != type || t.dims[0] != d0 || t.dims[1] != d1) die("unexpected shape or type: " + name);
}

std::vector<float> read_f32(const gguf::File &f, const std::string &name) {
    const gguf::Tensor &t = f.tensor(name);
    if (t.type != gguf::T_F32) die(name + " is not F32");
    std::vector<float> v(t.nbytes / 4);
    gguf::read_at(f, t.offset, v.data(), t.nbytes);
    return v;
}

float half_to_float(uint16_t h) {
    uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16, exp = h >> 10 & 31u, man = h & 1023u, u;
    if (exp == 31) {
        u = sign | 0x7F800000u | man << 13;
    } else if (exp) {
        u = sign | (exp + 112) << 23 | man << 13;
    } else if (man) {
        int shift = 0;
        while (!(man & 0x400u)) {
            man <<= 1;
            shift++;
        }
        u = sign | static_cast<uint32_t>(113 - shift) << 23 | (man & 1023u) << 13;
    } else {
        u = sign;
    }
    float out;
    memcpy(&out, &u, 4);
    return out;
}

}  // namespace

void check_shapes(const gguf::File &f) {
    auto it = f.strs.find("general.architecture");
    if (it == f.strs.end() || it->second != "qwen35") die("not qwen35");
    if (f.get_int("qwen35.block_count") != LAYERS + 1 || f.get_int("qwen35.embedding_length") != D ||
        f.get_int("qwen35.feed_forward_length") != FF || f.get_int("qwen35.attention.head_count") != QHEADS ||
        f.get_int("qwen35.attention.head_count_kv") != KVHEADS || f.get_int("qwen35.ssm.state_size") != ST ||
        f.get_int("qwen35.ssm.group_count") != KHEADS || f.get_int("qwen35.ssm.time_step_rank") != VHEADS ||
        f.get_int("qwen35.rope.dimension_count") != ROT)
        die("the metadata is not the model this code was written for");
    if (f.has("output.weight")) die("this code expects logits tied to the embedding");
    const uint32_t Q8 = gguf::T_Q8_0, F32 = gguf::T_F32;
    expect_tensor(f, "token_embd.weight", Q8, D, VOCAB);
    expect_tensor(f, "output_norm.weight", F32, D, 1);
    for (int il = 0; il < LAYERS; il++) {
        expect_tensor(f, blk(il, "attn_norm.weight"), F32, D, 1);
        expect_tensor(f, blk(il, "post_attention_norm.weight"), F32, D, 1);
        expect_tensor(f, blk(il, "ffn_gate.weight"), Q8, D, FF);
        expect_tensor(f, blk(il, "ffn_up.weight"), Q8, D, FF);
        expect_tensor(f, blk(il, "ffn_down.weight"), Q8, FF, D);
        if (is_full_attention(il)) {
            expect_tensor(f, blk(il, "attn_q.weight"), Q8, D, 2 * QHEADS * HD);
            expect_tensor(f, blk(il, "attn_k.weight"), Q8, D, KVHEADS * HD);
            expect_tensor(f, blk(il, "attn_v.weight"), Q8, D, KVHEADS * HD);
            expect_tensor(f, blk(il, "attn_output.weight"), Q8, QHEADS * HD, D);
            expect_tensor(f, blk(il, "attn_q_norm.weight"), F32, HD, 1);
            expect_tensor(f, blk(il, "attn_k_norm.weight"), F32, HD, 1);
        } else {
            expect_tensor(f, blk(il, "attn_qkv.weight"), Q8, D, QKV);
            expect_tensor(f, blk(il, "attn_gate.weight"), Q8, D, VAL_DIM);
            expect_tensor(f, blk(il, "ssm_alpha.weight"), Q8, D, VHEADS);
            expect_tensor(f, blk(il, "ssm_beta.weight"), Q8, D, VHEADS);
            expect_tensor(f, blk(il, "ssm_out.weight"), Q8, VAL_DIM, D);
            expect_tensor(f, blk(il, "ssm_conv1d.weight"), F32, CONV, QKV);
            expect_tensor(f, blk(il, "ssm_a"), F32, VHEADS, 1);
            expect_tensor(f, blk(il, "ssm_dt.bias"), F32, VHEADS, 1);
            expect_tensor(f, blk(il, "ssm_norm.weight"), F32, ST, 1);
        }
    }
}

struct Ref::Layer {
    std::vector<float> attn_norm, post_norm;
    std::vector<float> q_norm, k_norm;
    std::vector<float> conv, a, dt, ssm_norm;
    std::vector<float> state;   /* VHEADS x ST x ST */
    std::vector<float> win;     /* the last 3 projections, oldest first, QKV each */
    std::vector<float> kc, vc;  /* attention history: position x KVHEADS x HD */
    const gguf::Tensor *wqkv = nullptr, *wgate = nullptr, *walpha = nullptr, *wbeta = nullptr, *wout = nullptr;
    const gguf::Tensor *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr;
    const gguf::Tensor *g = nullptr, *u = nullptr, *d = nullptr;
};

namespace {

/* x rounded to the given bits, sign included, with one scale over these n values. */
void round_bits(float *x, size_t n, int bits) {
    float big = 0.0f;
    for (size_t i = 0; i < n; i++) big = std::max(big, std::fabs(x[i]));
    if (big == 0.0f) return;
    const float top = static_cast<float>((1 << (bits - 1)) - 1), up = top / big, down = big / top;
    for (size_t i = 0; i < n; i++) x[i] = std::rint(x[i] * up) * down;
}

}  // namespace

void Ref::round_vec(float *x, size_t n) const {
    if (!a_.act_bits) return;
    if (!a_.per_block) {
        round_bits(x, n, a_.act_bits);
        return;
    }
    for (size_t b = 0; b < n; b += 32) round_bits(x + b, std::min<size_t>(32, n - b), a_.act_bits);
}

float Ref::grid(float x) const {
    if (!a_.fn_bits) return x;
    const float m = static_cast<float>(1 << a_.fn_bits);
    return std::rint(x * m) / m;
}

Ref::Ref(const gguf::File &f, Approx a) : f_(f), a_(a), layers_(LAYERS), scale_tab_(65536) {
    check_shapes(f);
    void *m = mmap(nullptr, f.size, PROT_READ, MAP_SHARED, f.fd, 0);
    if (m == MAP_FAILED) die("mmap failed");
    map_ = static_cast<const uint8_t *>(m);
    for (uint32_t h = 0; h < 65536; h++) scale_tab_[h] = half_to_float(static_cast<uint16_t>(h));
    out_norm_ = read_f32(f, "output_norm.weight");
    for (int il = 0; il < LAYERS; il++) {
        Layer &L = layers_[static_cast<size_t>(il)];
        L.attn_norm = read_f32(f, blk(il, "attn_norm.weight"));
        L.post_norm = read_f32(f, blk(il, "post_attention_norm.weight"));
        L.g = &f.tensor(blk(il, "ffn_gate.weight"));
        L.u = &f.tensor(blk(il, "ffn_up.weight"));
        L.d = &f.tensor(blk(il, "ffn_down.weight"));
        if (is_full_attention(il)) {
            L.q_norm = read_f32(f, blk(il, "attn_q_norm.weight"));
            L.k_norm = read_f32(f, blk(il, "attn_k_norm.weight"));
            L.wq = &f.tensor(blk(il, "attn_q.weight"));
            L.wk = &f.tensor(blk(il, "attn_k.weight"));
            L.wv = &f.tensor(blk(il, "attn_v.weight"));
            L.wo = &f.tensor(blk(il, "attn_output.weight"));
        } else {
            L.conv = read_f32(f, blk(il, "ssm_conv1d.weight"));
            L.a = read_f32(f, blk(il, "ssm_a"));
            L.dt = read_f32(f, blk(il, "ssm_dt.bias"));
            L.ssm_norm = read_f32(f, blk(il, "ssm_norm.weight"));
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

Ref::~Ref() {
    if (map_) munmap(const_cast<uint8_t *>(map_), f_.size);
}

void Ref::reset() {
    pos_ = 0;
    for (Layer &L : layers_) {
        std::fill(L.state.begin(), L.state.end(), 0.0f);
        std::fill(L.win.begin(), L.win.end(), 0.0f);
        L.kc.clear();
        L.vc.clear();
    }
}

void Ref::q8_gemv(const gguf::Tensor &t, uint64_t rows, const float *x_in, float *y) const {
    const uint64_t cols = t.dims[0], row_bytes = gguf::row_bytes(gguf::T_Q8_0, cols), blocks = cols / 32;
    const uint8_t *base = map_ + t.offset;
    std::vector<float> rounded;
    const float *x = x_in;
    if (a_.act_bits) {
        rounded.assign(x_in, x_in + cols);
        round_vec(rounded.data(), cols);
        x = rounded.data();
    }
    const float *tab = scale_tab_.data();
#pragma omp parallel for schedule(static)
    for (uint64_t r = 0; r < rows; r++) {
        const uint8_t *p = base + r * row_bytes;
        float acc = 0.0f;
        for (uint64_t b = 0; b < blocks; b++, p += 34) {
            uint16_t h;
            memcpy(&h, p, 2);
            const int8_t *q = reinterpret_cast<const int8_t *>(p + 2);
            const float *xb = x + b * 32;
            float inner = 0.0f;
            for (int i = 0; i < 32; i++) inner = Q_FMA(static_cast<float>(q[i]), xb[i], inner);
            acc = Q_FMA(tab[h], inner, acc);
        }
        y[r] = acc;
    }
}

void Ref::ffn(int il, const float *h, float *out, const Tap *tap) {
    Layer &L = layers_[static_cast<size_t>(il)];
    std::vector<float> g(FF), u(FF), x(FF);
    q8_gemv(*L.g, FF, h, g.data());
    q8_gemv(*L.u, FF, h, u.data());
    for (int i = 0; i < FF; i++) x[static_cast<size_t>(i)] = grid(q_silu(grid(g[static_cast<size_t>(i)]))) * u[static_cast<size_t>(i)];
    q8_gemv(*L.d, D, x.data(), out);
    if (tap) (*tap)("ffn_out", il, out, D);
}

void Ref::layer_gdn(int il, const float *h, float *out, const Tap *tap) {
    Layer &L = layers_[static_cast<size_t>(il)];
    std::vector<float> qkv(QKV), z(VAL_DIM);
    float alpha[VHEADS], beta[VHEADS], decay[VHEADS];
    q8_gemv(*L.wqkv, QKV, h, qkv.data());
    q8_gemv(*L.wgate, VAL_DIM, h, z.data());
    q8_gemv(*L.walpha, VHEADS, h, alpha);
    q8_gemv(*L.wbeta, VHEADS, h, beta);
    if (tap) (*tap)("linear_attn_qkv_mixed", il, qkv.data(), QKV);
    if (tap) (*tap)("z", il, z.data(), VAL_DIM);
    float gate[VHEADS];
    for (int i = 0; i < VHEADS; i++) {
        beta[i] = grid(q_sigmoid(grid(beta[i])));
        gate[i] = grid(q_softplus(grid(alpha[i] + L.dt[static_cast<size_t>(i)]))) * L.a[static_cast<size_t>(i)];
        decay[i] = grid(q_exp(grid(gate[i])));
    }
    if (tap) (*tap)("beta_sigmoid", il, beta, VHEADS);
    if (tap) (*tap)("gate", il, gate, VHEADS);
    /* the causal conv: three remembered projections and this one, oldest first */
    std::vector<float> conv(QKV);
    float *win = L.win.data();
#pragma omp parallel for schedule(static)
    for (int c = 0; c < QKV; c++) {
        const float *w = &L.conv[static_cast<size_t>(c) * CONV];
        float acc = 0.0f;
        for (int t = 0; t < 3; t++) acc = Q_FMA(w[t], win[static_cast<size_t>(t) * QKV + static_cast<size_t>(c)], acc);
        acc = Q_FMA(w[3], qkv[static_cast<size_t>(c)], acc);
        conv[static_cast<size_t>(c)] = grid(q_silu(grid(acc)));
    }
    if (tap) (*tap)("conv_output_silu", il, conv.data(), QKV);
    memmove(win, win + QKV, 2 * QKV * sizeof(float));
    memcpy(win + 2 * QKV, qkv.data(), QKV * sizeof(float));
    /* query and key, normalized per key head */
    std::vector<float> q(KEY_DIM), k(KEY_DIM);
    const float q_scale = 0.0883883461f;  /* 1 / sqrt(ST) */
    for (int kh = 0; kh < KHEADS; kh++) {
        const float *cq = &conv[static_cast<size_t>(kh) * ST];
        const float *ck = &conv[static_cast<size_t>(KEY_DIM + kh * ST)];
        float sq = q_sum_sq(cq, ST), sk = q_sum_sq(ck, ST);
        const float e2 = RMS_EPS * RMS_EPS;
        float rq = q_rsqrt(sq > e2 ? sq : e2), rk = q_rsqrt(sk > e2 ? sk : e2);
        for (int i = 0; i < ST; i++) {
            q[static_cast<size_t>(kh * ST + i)] = (cq[i] * rq) * q_scale;
            k[static_cast<size_t>(kh * ST + i)] = ck[i] * rk;
        }
        round_vec(&q[static_cast<size_t>(kh * ST)], ST);
        round_vec(&k[static_cast<size_t>(kh * ST)], ST);
    }
    for (int hd = 0; hd < VHEADS; hd++) round_vec(&conv[static_cast<size_t>(2 * KEY_DIM + hd * ST)], ST);
    const float *v = &conv[2 * KEY_DIM];
    std::vector<float> mix(VAL_DIM);
#pragma omp parallel for schedule(static)
    for (int hd = 0; hd < VHEADS; hd++) {
        float *S = &L.state[static_cast<size_t>(hd) * ST * ST];
        const float *kk = &k[static_cast<size_t>((hd % KHEADS) * ST)];
        const float *qq = &q[static_cast<size_t>((hd % KHEADS) * ST)];
        const float *vv = v + hd * ST;
        const float e = decay[hd], b = beta[hd];
        float kv[ST] = {0}, delta[ST], o[ST] = {0};
        for (int i = 0; i < ST; i++) {
            const float *row = S + i * ST;
            for (int j = 0; j < ST; j++) kv[j] = Q_FMA(row[j] * e, kk[i], kv[j]);
        }
        for (int j = 0; j < ST; j++) delta[j] = (vv[j] - kv[j]) * b;
        for (int i = 0; i < ST; i++) {
            float *row = S + i * ST;
            for (int j = 0; j < ST; j++) {
                float s = Q_FMA(kk[i], delta[j], row[j] * e);
                row[j] = s;
                o[j] = Q_FMA(s, qq[i], o[j]);
            }
        }
        if (a_.state_bits) round_bits(S, ST * ST, a_.state_bits);
        const float *zz = &z[static_cast<size_t>(hd) * ST];
        float scale = q_rms_scale(o, ST, 1.0f / ST, RMS_EPS);
        float *m = &mix[static_cast<size_t>(hd) * ST];
        for (int j = 0; j < ST; j++) m[j] = ((o[j] * scale) * L.ssm_norm[static_cast<size_t>(j)]) * grid(q_silu(grid(zz[j])));
    }
    if (tap) (*tap)("final_output", il, mix.data(), VAL_DIM);
    q8_gemv(*L.wout, D, mix.data(), out);
    if (tap) (*tap)("linear_attn_out", il, out, D);
}

void Ref::layer_full(int il, const float *h, float *out, const Tap *tap) {
    Layer &L = layers_[static_cast<size_t>(il)];
    std::vector<float> qg(2 * QHEADS * HD), kk(KVHEADS * HD), vv(KVHEADS * HD);
    q8_gemv(*L.wq, 2 * QHEADS * HD, h, qg.data());
    q8_gemv(*L.wk, KVHEADS * HD, h, kk.data());
    q8_gemv(*L.wv, KVHEADS * HD, h, vv.data());
    if (tap) (*tap)("Qcur_full", il, qg.data(), 2 * QHEADS * HD);
    float c[ROT / 2], s[ROT / 2];
    q35::rope_row(pos_, c, s);
    auto rotate = [&](float *x) {
        for (int i = 0; i < ROT / 2; i++) {
            float a = x[i], b = x[i + ROT / 2];
            x[i] = Q_FMA(a, c[i], -b * s[i]);
            x[i + ROT / 2] = Q_FMA(a, s[i], b * c[i]);
        }
    };
    std::vector<float> q(QHEADS * HD);
    for (int hq = 0; hq < QHEADS; hq++) {
        const float *src = &qg[static_cast<size_t>(hq) * 2 * HD];
        float *dst = &q[static_cast<size_t>(hq) * HD];
        float scale = q_rms_scale(src, HD, 1.0f / HD, RMS_EPS);
        for (int i = 0; i < HD; i++) dst[i] = (src[i] * scale) * L.q_norm[static_cast<size_t>(i)];
        rotate(dst);
    }
    if (tap) (*tap)("Qcur", il, q.data(), QHEADS * HD);
    for (int hk = 0; hk < KVHEADS; hk++) {
        float *x = &kk[static_cast<size_t>(hk) * HD];
        float scale = q_rms_scale(x, HD, 1.0f / HD, RMS_EPS);
        for (int i = 0; i < HD; i++) x[i] = (x[i] * scale) * L.k_norm[static_cast<size_t>(i)];
        rotate(x);
    }
    if (tap) (*tap)("Kcur", il, kk.data(), KVHEADS * HD);
    if (tap) (*tap)("Vcur", il, vv.data(), KVHEADS * HD);
    for (int hq = 0; hq < QHEADS; hq++) round_vec(&q[static_cast<size_t>(hq) * HD], HD);
    for (int hk = 0; hk < KVHEADS; hk++) {
        round_vec(&kk[static_cast<size_t>(hk) * HD], HD);
        round_vec(&vv[static_cast<size_t>(hk) * HD], HD);
    }
    L.kc.insert(L.kc.end(), kk.begin(), kk.end());
    L.vc.insert(L.vc.end(), vv.begin(), vv.end());
    const int n = static_cast<int>(L.kc.size() / (KVHEADS * HD));
    const float kq_scale = 0.0625f;  /* 1 / sqrt(HD) */
    std::vector<float> o(QHEADS * HD);
#pragma omp parallel for schedule(static)
    for (int hq = 0; hq < QHEADS; hq++) {
        const int kvh = hq * KVHEADS / QHEADS;
        const float *qq = &q[static_cast<size_t>(hq) * HD];
        std::vector<float> p(static_cast<size_t>(n));
        for (int t = 0; t < n; t++) {
            const float *key = &L.kc[(static_cast<size_t>(t) * KVHEADS + static_cast<size_t>(kvh)) * HD];
            p[static_cast<size_t>(t)] = q_dot(qq, key, HD) * kq_scale;
        }
        q_softmax(p.data(), n);
        if (a_.act_bits) {
            /* probabilities in [0, 1], unsigned, as the unit's VA holds them */
            const float top = static_cast<float>((1 << a_.act_bits) - 1);
            for (int t = 0; t < n; t++) p[static_cast<size_t>(t)] = std::rint(p[static_cast<size_t>(t)] * top) / top;
        }
        float *oo = &o[static_cast<size_t>(hq) * HD];
        for (int i = 0; i < HD; i++) oo[i] = 0.0f;
        for (int t = 0; t < n; t++) {
            const float *val = &L.vc[(static_cast<size_t>(t) * KVHEADS + static_cast<size_t>(kvh)) * HD];
            float pt = p[static_cast<size_t>(t)];
            for (int i = 0; i < HD; i++) oo[i] = Q_FMA(pt, val[i], oo[i]);
        }
    }
    if (tap) (*tap)("attn_pregate", il, o.data(), QHEADS * HD);
    for (int hq = 0; hq < QHEADS; hq++) {
        const float *gt = &qg[static_cast<size_t>(hq) * 2 * HD + HD];
        float *oo = &o[static_cast<size_t>(hq) * HD];
        for (int i = 0; i < HD; i++) oo[i] = oo[i] * grid(q_sigmoid(grid(gt[i])));
    }
    if (tap) (*tap)("attn_gated", il, o.data(), QHEADS * HD);
    q8_gemv(*L.wo, D, o.data(), out);
    if (tap) (*tap)("attn_output", il, out, D);
}

void Ref::step(int32_t token, float *logits, const Tap *tap) {
    const gguf::Tensor &emb = f_.tensor("token_embd.weight");
    const uint64_t row = gguf::row_bytes(gguf::T_Q8_0, D);
    const uint8_t *raw = map_ + emb.offset + static_cast<uint64_t>(token) * row;
    float x[D], h[D], y[D];
    for (int b = 0; b < D / 32; b++) {
        uint16_t hh;
        memcpy(&hh, raw + static_cast<size_t>(b) * 34, 2);
        const int8_t *q = reinterpret_cast<const int8_t *>(raw + static_cast<size_t>(b) * 34 + 2);
        for (int i = 0; i < 32; i++) x[b * 32 + i] = scale_tab_[hh] * static_cast<float>(q[i]);
    }
    for (int il = 0; il < LAYERS; il++) {
        Layer &L = layers_[static_cast<size_t>(il)];
        float scale = q_rms_scale(x, D, 1.0f / D, RMS_EPS);
        for (int i = 0; i < D; i++) h[i] = (x[i] * scale) * L.attn_norm[static_cast<size_t>(i)];
        if (tap) (*tap)("attn_norm", il, h, D);
        if (is_full_attention(il)) layer_full(il, h, y, tap);
        else layer_gdn(il, h, y, tap);
        for (int i = 0; i < D; i++) x[i] = x[i] + y[i];
        if (tap) (*tap)("attn_residual", il, x, D);
        scale = q_rms_scale(x, D, 1.0f / D, RMS_EPS);
        for (int i = 0; i < D; i++) h[i] = (x[i] * scale) * L.post_norm[static_cast<size_t>(i)];
        if (tap) (*tap)("attn_post_norm", il, h, D);
        ffn(il, h, y, tap);
        for (int i = 0; i < D; i++) x[i] = x[i] + y[i];
        if (tap) (*tap)("l_out", il, x, D);
    }
    float scale = q_rms_scale(x, D, 1.0f / D, RMS_EPS);
    for (int i = 0; i < D; i++) h[i] = (x[i] * scale) * out_norm_[static_cast<size_t>(i)];
    if (tap) (*tap)("result_norm", -1, h, D);
    q8_gemv(emb, VOCAB, h, logits);
    if (tap) (*tap)("result_output", -1, logits, VOCAB);
    pos_++;
}

}  // namespace q2b
