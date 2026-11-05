#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "elem.h"
#include "q35.h"
#include "q35_hash.h"

namespace q35 {

namespace {

[[noreturn]] void die(const std::string &what) {
    fprintf(stderr, "q35: %s\n", what.c_str());
    exit(1);
}

std::string blk(int il, const char *suffix) { return "blk." + std::to_string(il) + "." + suffix; }

void expect_tensor(const gguf::File &f, const std::string &name, uint32_t type, uint64_t d0, uint64_t d1, uint64_t d2 = 1) {
    const gguf::Tensor &t = f.tensor(name);
    if (t.type != type || t.dims[0] != d0 || t.dims[1] != d1 || t.dims[2] != d2) die("unexpected shape or type: " + name);
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
    if (it == f.strs.end() || it->second != "qwen35moe") die("not qwen35moe");
    if (f.get_int("qwen35moe.block_count") != LAYERS + 1 || f.get_int("qwen35moe.embedding_length") != D ||
        f.get_int("qwen35moe.expert_count") != NEXP || f.get_int("qwen35moe.expert_used_count") != USED ||
        f.get_int("qwen35moe.expert_feed_forward_length") != FF || f.get_int("qwen35moe.attention.head_count") != QHEADS ||
        f.get_int("qwen35moe.attention.head_count_kv") != KVHEADS || f.get_int("qwen35moe.ssm.state_size") != ST ||
        f.get_int("qwen35moe.ssm.group_count") != KHEADS || f.get_int("qwen35moe.ssm.time_step_rank") != VHEADS ||
        f.get_int("qwen35moe.rope.dimension_count") != ROT)
        die("the metadata is not the model this code was written for");
    const uint32_t Q8 = gguf::T_Q8_0, F32 = gguf::T_F32;
    expect_tensor(f, "token_embd.weight", Q8, D, VOCAB);
    expect_tensor(f, "output.weight", Q8, D, VOCAB);
    expect_tensor(f, "output_norm.weight", F32, D, 1);
    for (int il = 0; il < LAYERS; il++) {
        expect_tensor(f, blk(il, "attn_norm.weight"), F32, D, 1);
        expect_tensor(f, blk(il, "post_attention_norm.weight"), F32, D, 1);
        expect_tensor(f, blk(il, "ffn_gate_inp.weight"), F32, D, NEXP);
        expect_tensor(f, blk(il, "ffn_gate_inp_shexp.weight"), F32, D, 1);
        expect_tensor(f, blk(il, "ffn_gate_exps.weight"), Q8, D, FF, NEXP);
        expect_tensor(f, blk(il, "ffn_up_exps.weight"), Q8, D, FF, NEXP);
        expect_tensor(f, blk(il, "ffn_down_exps.weight"), Q8, FF, D, NEXP);
        expect_tensor(f, blk(il, "ffn_gate_shexp.weight"), Q8, D, FF);
        expect_tensor(f, blk(il, "ffn_up_shexp.weight"), Q8, D, FF);
        expect_tensor(f, blk(il, "ffn_down_shexp.weight"), Q8, FF, D);
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

void rope_row(uint32_t pos, float cos_out[ROT / 2], float sin_out[ROT / 2]) {
    for (int i = 0; i < ROT / 2; i++) {
        double freq = std::pow(10000000.0, -2.0 * i / ROT);
        double theta = pos * freq;
        cos_out[i] = static_cast<float>(std::cos(theta));
        sin_out[i] = static_cast<float>(std::sin(theta));
    }
}

struct Ref::Layer {
    std::vector<float> attn_norm, post_norm, gate_inp, gate_inp_sh;
    std::vector<float> q_norm, k_norm;
    std::vector<float> conv, a, dt, ssm_norm;
    std::vector<float> state;   /* VHEADS x ST x ST */
    std::vector<float> win;     /* the last 3 projections, oldest first, QKV each */
    std::vector<float> kc, vc;  /* attention history: position x KVHEADS x HD */
    const gguf::Tensor *wqkv = nullptr, *wgate = nullptr, *walpha = nullptr, *wbeta = nullptr, *wout = nullptr;
    const gguf::Tensor *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr;
    const gguf::Tensor *g_exps = nullptr, *u_exps = nullptr, *d_exps = nullptr, *g_sh = nullptr, *u_sh = nullptr, *d_sh = nullptr;
};

Ref::Ref(const gguf::File &f) : f_(f), layers_(LAYERS), scale_tab_(65536), routes_(LAYERS * USED) {
    check_shapes(f);
    for (uint32_t h = 0; h < 65536; h++) scale_tab_[h] = half_to_float(static_cast<uint16_t>(h));
    out_norm_ = read_f32(f, "output_norm.weight");
    for (int il = 0; il < LAYERS; il++) {
        Layer &L = layers_[static_cast<size_t>(il)];
        L.attn_norm = read_f32(f, blk(il, "attn_norm.weight"));
        L.post_norm = read_f32(f, blk(il, "post_attention_norm.weight"));
        L.gate_inp = read_f32(f, blk(il, "ffn_gate_inp.weight"));
        L.gate_inp_sh = read_f32(f, blk(il, "ffn_gate_inp_shexp.weight"));
        L.g_exps = &f.tensor(blk(il, "ffn_gate_exps.weight"));
        L.u_exps = &f.tensor(blk(il, "ffn_up_exps.weight"));
        L.d_exps = &f.tensor(blk(il, "ffn_down_exps.weight"));
        L.g_sh = &f.tensor(blk(il, "ffn_gate_shexp.weight"));
        L.u_sh = &f.tensor(blk(il, "ffn_up_shexp.weight"));
        L.d_sh = &f.tensor(blk(il, "ffn_down_shexp.weight"));
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

Ref::~Ref() = default;

void Ref::reset() {
    pos_ = 0;
    for (Layer &L : layers_) {
        std::fill(L.state.begin(), L.state.end(), 0.0f);
        std::fill(L.win.begin(), L.win.end(), 0.0f);
        L.kc.clear();
        L.vc.clear();
    }
}

void Ref::read_weights(const gguf::Tensor &, uint64_t off, uint64_t n) {
    if (buf_.size() < n) buf_.resize(n);
    gguf::read_at(f_, off, buf_.data(), n);
    bytes_read_ += n;
}

void Ref::q8_gemv(const gguf::Tensor &t, uint64_t skip_rows, uint64_t rows, const float *x, float *y) {
    const uint64_t cols = t.dims[0], row_bytes = gguf::row_bytes(gguf::T_Q8_0, cols), blocks = cols / 32;
    read_weights(t, t.offset + skip_rows * row_bytes, rows * row_bytes);
    const uint8_t *base = buf_.data();
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

void Ref::moe_ffn(int il, const float *h, float *out, const Tap *tap) {
    Layer &L = layers_[static_cast<size_t>(il)];
    float logits[NEXP], probs[NEXP];
#pragma omp parallel for schedule(static)
    for (int r = 0; r < NEXP; r++) {
        const float *w = &L.gate_inp[static_cast<size_t>(r) * D];
        float acc = 0.0f;
        for (int j = 0; j < D; j++) acc = Q_FMA(w[j], h[j], acc);
        logits[r] = acc;
    }
    if (tap) (*tap)("ffn_moe_logits", il, logits, NEXP);
    memcpy(probs, logits, sizeof probs);
    q_softmax(probs, NEXP);
    if (tap) (*tap)("ffn_moe_probs", il, probs, NEXP);
    int32_t order[NEXP];
    for (int i = 0; i < NEXP; i++) order[i] = i;
    std::stable_sort(order, order + NEXP, [&](int32_t a, int32_t b) { return probs[a] > probs[b]; });
    int32_t *sel = &routes_[static_cast<size_t>(il) * USED];
    float picked[16] = {0};
    for (int e = 0; e < USED; e++) {
        sel[e] = order[e];
        picked[e] = probs[order[e]];
    }
    if (tap) {
        float s[USED];
        for (int e = 0; e < USED; e++) s[e] = static_cast<float>(sel[e]);
        (*tap)("ffn_moe_topk", il, s, USED);
    }
    float sum = q_tree16(picked);
    float weight_scale = q_rcp(sum > 6.103515625e-5f ? sum : 6.103515625e-5f);
    float w[USED];
    for (int e = 0; e < USED; e++) w[e] = picked[e] * weight_scale;
    if (tap) (*tap)("ffn_moe_weights_norm", il, w, USED);
    float gate_s = q_sigmoid(q_dot(L.gate_inp_sh.data(), h, D));
    if (tap) (*tap)("shared_expert_gate_sigmoid", il, &gate_s, 1);
    /* the hidden side of the 8 experts and the shared one: each expert's x, weighted, ready for its
     * down matrix. Summing 9 outputs afterwards would only add a pass over memory. */
    std::vector<float> xs((USED + 1) * FF);
    float g[FF], u[FF];
    for (int e = 0; e <= USED; e++) {
        float *xe = &xs[static_cast<size_t>(e) * FF];
        const float scale = e < USED ? w[e] : gate_s;
        if (e < USED) {
            const uint64_t ex = static_cast<uint64_t>(sel[e]);
            q8_gemv(*L.g_exps, ex * FF, FF, h, g);
            q8_gemv(*L.u_exps, ex * FF, FF, h, u);
        } else {
            q8_gemv(*L.g_sh, 0, FF, h, g);
            q8_gemv(*L.u_sh, 0, FF, h, u);
        }
        for (int i = 0; i < FF; i++) xe[i] = (q_silu(g[i]) * u[i]) * scale;
    }
    /* one accumulator per output row, across all the experts in order */
    const uint64_t down_row = gguf::row_bytes(gguf::T_Q8_0, FF);
    std::vector<uint8_t> downs((USED + 1) * D * down_row);
    for (int e = 0; e <= USED; e++) {
        const gguf::Tensor &t = e < USED ? *L.d_exps : *L.d_sh;
        uint64_t off = t.offset + (e < USED ? static_cast<uint64_t>(sel[e]) * D * down_row : 0);
        gguf::read_at(f_, off, &downs[static_cast<size_t>(e) * D * down_row], D * down_row);
        bytes_read_ += D * down_row;
    }
    const float *tab = scale_tab_.data();
#pragma omp parallel for schedule(static)
    for (int r = 0; r < D; r++) {
        float acc = 0.0f;
        for (int e = 0; e <= USED; e++) {
            const uint8_t *p = &downs[(static_cast<size_t>(e) * D + static_cast<size_t>(r)) * down_row];
            const float *xe = &xs[static_cast<size_t>(e) * FF];
            for (int b = 0; b < FF / 32; b++, p += 34) {
                uint16_t hh;
                memcpy(&hh, p, 2);
                const int8_t *q = reinterpret_cast<const int8_t *>(p + 2);
                float inner = 0.0f;
                for (int i = 0; i < 32; i++) inner = Q_FMA(static_cast<float>(q[i]), xe[b * 32 + i], inner);
                acc = Q_FMA(tab[hh], inner, acc);
            }
        }
        out[r] = acc;
    }
    if (tap) (*tap)("ffn_out", il, out, D);
}

void Ref::layer_gdn(int il, const float *h, float *out, const Tap *tap) {
    Layer &L = layers_[static_cast<size_t>(il)];
    std::vector<float> qkv(QKV), z(VAL_DIM);
    float alpha[VHEADS], beta[VHEADS], decay[VHEADS];
    q8_gemv(*L.wqkv, 0, QKV, h, qkv.data());
    q8_gemv(*L.wgate, 0, VAL_DIM, h, z.data());
    q8_gemv(*L.walpha, 0, VHEADS, h, alpha);
    q8_gemv(*L.wbeta, 0, VHEADS, h, beta);
    if (tap) (*tap)("linear_attn_qkv_mixed", il, qkv.data(), QKV);
    if (tap) (*tap)("z", il, z.data(), VAL_DIM);
    float gate[VHEADS];
    for (int i = 0; i < VHEADS; i++) {
        beta[i] = q_sigmoid(beta[i]);
        gate[i] = q_softplus(alpha[i] + L.dt[static_cast<size_t>(i)]) * L.a[static_cast<size_t>(i)];
        decay[i] = q_exp(gate[i]);
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
        conv[static_cast<size_t>(c)] = q_silu(acc);
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
    }
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
        const float *zz = &z[static_cast<size_t>(hd) * ST];
        float scale = q_rms_scale(o, ST, 1.0f / ST, RMS_EPS);
        float *m = &mix[static_cast<size_t>(hd) * ST];
        for (int j = 0; j < ST; j++) m[j] = ((o[j] * scale) * L.ssm_norm[static_cast<size_t>(j)]) * q_silu(zz[j]);
    }
    if (tap) (*tap)("final_output", il, mix.data(), VAL_DIM);
    q8_gemv(*L.wout, 0, D, mix.data(), out);
    if (tap) (*tap)("linear_attn_out", il, out, D);
}

void Ref::layer_full(int il, const float *h, float *out, const Tap *tap) {
    Layer &L = layers_[static_cast<size_t>(il)];
    std::vector<float> qg(2 * QHEADS * HD), kk(KVHEADS * HD), vv(KVHEADS * HD);
    q8_gemv(*L.wq, 0, 2 * QHEADS * HD, h, qg.data());
    q8_gemv(*L.wk, 0, KVHEADS * HD, h, kk.data());
    q8_gemv(*L.wv, 0, KVHEADS * HD, h, vv.data());
    if (tap) (*tap)("Qcur_full", il, qg.data(), 2 * QHEADS * HD);
    float c[ROT / 2], s[ROT / 2];
    rope_row(pos_, c, s);
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
        for (int i = 0; i < HD; i++) oo[i] = oo[i] * q_sigmoid(gt[i]);
    }
    if (tap) (*tap)("attn_gated", il, o.data(), QHEADS * HD);
    q8_gemv(*L.wo, 0, D, o.data(), out);
    if (tap) (*tap)("attn_output", il, out, D);
}

void Ref::step(int32_t token, float *logits, const Tap *tap) {
    const gguf::Tensor &emb = f_.tensor("token_embd.weight");
    const uint64_t row = gguf::row_bytes(gguf::T_Q8_0, D);
    std::vector<uint8_t> raw(row);
    gguf::read_at(f_, emb.offset + static_cast<uint64_t>(token) * row, raw.data(), row);
    float x[D], h[D], y[D];
    for (int b = 0; b < D / 32; b++) {
        uint16_t hh;
        memcpy(&hh, &raw[static_cast<size_t>(b) * 34], 2);
        const int8_t *q = reinterpret_cast<const int8_t *>(&raw[static_cast<size_t>(b) * 34 + 2]);
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
        moe_ffn(il, h, y, tap);
        for (int i = 0; i < D; i++) x[i] = x[i] + y[i];
        if (tap) (*tap)("l_out", il, x, D);
        check_.x_hash[il] = q35_vector_hash(reinterpret_cast<const uint32_t *>(x), D);
    }
    float scale = q_rms_scale(x, D, 1.0f / D, RMS_EPS);
    for (int i = 0; i < D; i++) h[i] = (x[i] * scale) * out_norm_[static_cast<size_t>(i)];
    if (tap) (*tap)("result_norm", -1, h, D);
    check_.x_hash[LAYERS] = q35_vector_hash(reinterpret_cast<const uint32_t *>(h), D);
    q8_gemv(f_.tensor("output.weight"), 0, VOCAB, h, logits);
    if (tap) (*tap)("result_output", -1, logits, VOCAB);
    /* the same scan and the same hash the card's threads do, group by group */
    const uint32_t *bits = reinterpret_cast<const uint32_t *>(logits);
    uint64_t hash = 0;
    uint32_t tk = 0, sk = 0;
    int32_t ti = -1, si = -1;
    for (uint32_t g = 0; g < VOCAB / 16; g++) {
        hash += q35_group_hash(g, bits + g * 16);
        for (uint32_t l = 0; l < 16; l++) {
            const uint32_t k = q35_order_key(bits[g * 16 + l]);
            const int32_t idx = static_cast<int32_t>(g * 16 + l);
            if (ti < 0 || k > tk) {
                sk = tk;
                si = ti;
                tk = k;
                ti = idx;
            } else if (si < 0 || k > sk) {
                sk = k;
                si = idx;
            }
        }
    }
    check_.logits_hash = hash;
    check_.next = ti;
    check_.second = si;
    check_.top_bits = (tk >> 31) ? tk ^ 0x80000000u : ~tk;
    check_.second_bits = (sk >> 31) ? sk ^ 0x80000000u : ~sk;
    pos_++;
}

}  // namespace q35
