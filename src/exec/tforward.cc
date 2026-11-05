#include "tforward.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

#include "../text/elem.h"

namespace {

[[noreturn]] void dies(const std::string &what) {
    std::fprintf(stderr, "tforward: %s\n", what.c_str());
    std::exit(2);
}

constexpr float EPS = 1e-6f;

// A bf16 activation: the top 16 bits of a float, rounded to nearest, ties to even. A layer with bf16 activations stores what it
// computes this way, as a bf16 model's activations are; a value the layer means exactly loses only the arithmetic's last bits.
inline float bf16(float x) {
    uint32_t u = q_bits(x);
    u += 0x7FFFu + (u >> 16 & 1u);
    return q_float(u & 0xFFFF0000u);
}

inline void bf16_all(float *x, size_t n) {
    for (size_t i = 0; i < n; i++) x[i] = bf16(x[i]);
}

}  // namespace

// The tensor's nonzeros: the file is sparse, so only its data extents are read at all.
TForward::Sparse TForward::load_sparse(const std::string &name) {
    Sparse s;
    if (!f_.has(name)) return s;
    const gguf::Tensor &t = f_.tensor(name);
    if (t.type != gguf::T_F32) dies(name + " is not F32");
    const uint64_t cols = t.dims[0], rows = t.dims[1], rowb = cols * 4, end = t.offset + t.nbytes;
    s.rows = (uint32_t)rows;
    std::vector<float> page(16384);
    uint64_t at = t.offset;
    while (at < end) {
        const off_t d = lseek(f_.fd, (off_t)at, SEEK_DATA);
        if (d < 0 || (uint64_t)d >= end) break;
        off_t e = lseek(f_.fd, d, SEEK_HOLE);
        if (e < 0 || (uint64_t)e > end) e = (off_t)end;
        data_bytes_ += (uint64_t)(e - d);
        uint64_t p = (uint64_t)d;
        p -= (p - t.offset) % 4;
        for (; p < (uint64_t)e;) {
            const uint64_t n = std::min<uint64_t>((uint64_t)e - p, page.size() * 4);
            if (pread(f_.fd, page.data(), n, (off_t)p) != (ssize_t)n) dies("short read of " + name);
            for (uint64_t i = 0; i < n / 4; i++)
                if (page[i] != 0.0f) {
                    const uint64_t rel = p + i * 4 - t.offset;
                    const uint32_t r = (uint32_t)(rel / rowb), c = (uint32_t)((rel % rowb) / 4);
                    if (s.row.empty() || s.row.back() != r) {
                        s.row.push_back(r);
                        s.start.push_back((uint32_t)s.col.size());
                    }
                    s.col.push_back(c);
                    s.val.push_back(page[i]);
                }
            p += n;
        }
        at = (uint64_t)e;
    }
    s.start.push_back((uint32_t)s.col.size());
    nnz_ += s.col.size();
    return s;
}

std::vector<float> TForward::load_dense(const std::string &name, uint64_t n) {
    std::vector<float> v(n);
    if (!n) return v;
    const gguf::Tensor &t = f_.tensor(name);
    if (t.nbytes != n * 4) dies(name + ": unexpected size");
    gguf::read_at(f_, t.offset, v.data(), t.nbytes);
    return v;
}

void TForward::gemv(const Sparse &m, const float *x, float *y) const {
    std::fill(y, y + m.rows, 0.0f);
    for (size_t k = 0; k + 1 < m.start.size(); k++) {
        float acc = 0.0f;
        for (uint32_t i = m.start[k]; i < m.start[k + 1]; i++) acc = Q_FMA(m.val[i], x[m.col[i]], acc);
        y[m.row[k]] = acc;
    }
}

TForward::TForward(const std::string &path) {
    gguf::open(f_, path.c_str());
    h_ = (int)f_.get_int("qwen35moe.chip.embedding_length");
    nk_ = (int)f_.get_int("qwen35moe.chip.ssm.group_count");
    nv_ = (int)f_.get_int("qwen35moe.chip.ssm.time_step_rank");
    st_ = (int)f_.get_int("qwen35moe.chip.ssm.state_size");
    w_ = (int)f_.get_int("qwen35moe.chip.expert_shared_feed_forward_length");
    const int layers = (int)f_.get_int("qwen35moe.chip.block_count");
    ls_.resize((size_t)layers);
    const uint64_t qkv = (uint64_t)(2 * nk_ + nv_) * (uint64_t)st_, val = (uint64_t)nv_ * (uint64_t)st_;
    for (int il = 0; il < layers; il++) {
        Layer &L = ls_[(size_t)il];
        const std::string p = "chip.blk." + std::to_string(il) + ".";
        L.qkv = load_sparse(p + "attn_qkv.weight");
        L.gate = load_sparse(p + "attn_gate.weight");
        L.alpha = load_sparse(p + "ssm_alpha.weight");
        L.beta = load_sparse(p + "ssm_beta.weight");
        L.out = load_sparse(p + "ssm_out.weight");
        L.f_gate = load_sparse(p + "ffn_gate_shexp.weight");
        L.f_up = load_sparse(p + "ffn_up_shexp.weight");
        L.f_down = load_sparse(p + "ffn_down_shexp.weight");
        L.attn_norm = load_dense(p + "attn_norm.weight", (uint64_t)h_);
        L.post_norm = load_dense(p + "post_attention_norm.weight", (uint64_t)h_);
        L.conv = load_dense(p + "ssm_conv1d.weight", qkv * 4);
        L.a = load_dense(p + "ssm_a", (uint64_t)nv_);
        L.dt = load_dense(p + "ssm_dt.bias", (uint64_t)nv_);
        L.ssm_norm = load_dense(p + "ssm_norm.weight", (uint64_t)st_);
        L.gate_inp = load_dense(p + "ffn_gate_inp_shexp.weight", (uint64_t)h_);
        for (std::vector<float> &w : L.win) w.assign(qkv, 0.0f);
        for (uint32_t c = 0; c < qkv; c++) {
            bool any = false;
            for (int t = 0; t < 4; t++) any = any || L.conv[c * 4ull + (uint64_t)t] != 0.0f;
            if (any) L.convrows.push_back(c);
        }
        // the heads with anything to read or write: the rest keep an empty state for ever
        std::vector<char> used((size_t)nv_, 0);
        const auto mark = [&](const Sparse &m, uint32_t base, uint32_t per) {
            for (uint32_t r : m.row)
                if (r >= base) used[(r - base) / per] = 1;
        };
        mark(L.gate, 0, (uint32_t)st_);
        for (uint32_t r : L.qkv.row)
            if (r >= 2ull * (uint64_t)nk_ * (uint64_t)st_) used[(r - 2u * (uint32_t)nk_ * (uint32_t)st_) / (uint32_t)st_] = 1;
        const gguf::Tensor &sti = f_.tensor(p + "ssm_state");
        for (int hd = 0; hd < nv_; hd++) {
            std::vector<float> st((size_t)st_ * (size_t)st_);
            gguf::read_at(f_, sti.offset + (uint64_t)hd * st.size() * 4, st.data(), st.size() * 4);
            bool any = used[(size_t)hd] != 0;
            for (float v : st) any = any || v != 0.0f;
            if (!any) continue;
            VState vs;
            vs.S = std::move(st);
            vs.hd = hd;
            L.st.push_back(std::move(vs));
        }
    }
    // the memories: chip.mem.K holds kind K's bytes (its true length in chip.meminfo); read-only kinds stay mapped
    {
        const int nm = (int)f_.get_int("qwen35moe.chip.mems");
        if (nm != MB_COUNT) dies("the file's memory kinds are not this forward's");
        const std::vector<float> info = load_dense("chip.meminfo", (uint64_t)MB_COUNT);
        for (int k = 0; k < MB_COUNT; k++) {
            mem_n_[k] = (uint64_t)info[(size_t)k];
            if (!mem_n_[k]) continue;
            const gguf::Tensor &m = f_.tensor("chip.mem." + std::to_string(k));
            if (m.nbytes < mem_n_[k]) dies("chip.mem." + std::to_string(k) + " is shorter than its length");
            if (k == MB_WBLK || k == MB_ROM || k == MB_F16) {
                const long pg = sysconf(_SC_PAGESIZE);
                const uint64_t at = m.offset & ~((uint64_t)pg - 1);
                void *mp = mmap(nullptr, m.nbytes + (m.offset - at), PROT_READ, MAP_SHARED, f_.fd, (off_t)at);
                if (mp == MAP_FAILED) dies("cannot map a memory");
                mem_ro_[k] = static_cast<const uint8_t *>(mp) + (m.offset - at);
            } else {
                mem_[k].resize(mem_n_[k]);
                gguf::read_at(f_, m.offset, mem_[k].data(), mem_n_[k]);
            }
        }
    }
    const int nb = (int)f_.get_int("qwen35moe.chip.banks");
    for (int K = 0; K < nb; K++) {
        const std::string p = "chip.bank." + std::to_string(K) + ".";
        const std::vector<float> info = load_dense(p + "info", 6);
        Bank b;
        b.kind = (int)info[0];
        if (b.kind < 0 || b.kind >= MB_COUNT) dies(p + "info: unknown memory kind");
        const int nr = (int)info[2], nw = (int)info[3];
        const std::vector<float> ri = load_dense(p + "rinfo", (uint64_t)nr * 4), wi = load_dense(p + "winfo", (uint64_t)nw * 9);
        int nra = 0, nwa = 0, nd = 0, nout = 0;
        for (int i = 0; i < nr; i++) {
            BRead r{(int)ri[(size_t)i * 4], (int)ri[(size_t)i * 4 + 1], (int)ri[(size_t)i * 4 + 3], (int)ri[(size_t)i * 4 + 2]};  // rinfo: bits, width, out, addr
            nra = std::max(nra, r.aoff + r.nbits);
            nout = std::max(nout, r.ooff + r.width);
            b.reads.push_back(r);
        }
        for (int i = 0; i < nw; i++) {
            const float *w = &wi[(size_t)i * 9];
            BWrite x{(int)w[0], (int)w[1], (int)w[2], (int)w[3], (int)w[4], {w[5], w[6], w[7], w[8]}};
            nwa = std::max(nwa, x.aoff + x.nabits);
            nd = std::max(nd, x.doff + x.ndbits);
            b.writes.push_back(x);
        }
        b.ra = load_sparse(p + "ra");
        b.wa = load_sparse(p + "wa");
        b.wd = load_sparse(p + "wd");
        b.we = load_sparse(p + "we");
        b.taps = load_dense(p + "taps", (uint64_t)nd * 4);
        const std::vector<float> o = load_dense(p + "out", (uint64_t)nout);
        for (float v : o) b.out.push_back((int)v);
        b.vra.assign((size_t)std::max(nra, 1), 0.0f);
        b.vwa.assign((size_t)std::max(nwa, 1), 0.0f);
        b.vd.assign((size_t)std::max(nd, 1), 0.0f);
        b.vwe.assign((size_t)std::max(nw, 1), 0.0f);
        for (std::vector<float> &h : b.hwa) h.assign((size_t)std::max(nwa, 1), 0.0f);
        for (std::vector<float> &h : b.hd) h.assign((size_t)std::max(nd, 1), 0.0f);
        // the rows' counts as the file has them: a gemv writes rows < .rows
        b.ra.rows = std::max<uint32_t>(b.ra.rows, (uint32_t)nra);
        b.wa.rows = std::max<uint32_t>(b.wa.rows, (uint32_t)nwa);
        b.wd.rows = std::max<uint32_t>(b.wd.rows, (uint32_t)nd);
        b.we.rows = std::max<uint32_t>(b.we.rows, (uint32_t)nw);
        if ((size_t)info[1] >= ls_.size()) dies(p + "info: the layer is out of range");
        ls_[(size_t)info[1]].banks.push_back(std::move(b));
    }
    x_.resize((size_t)h_);
    hn_.resize((size_t)h_);
    y_.resize((size_t)h_);
    qkv_.resize(qkv);
    conv_.resize(qkv);
    z_.resize(val);
    mix_.resize(val);
    g_.resize((size_t)w_);
    u_.resize((size_t)w_);
    xs_.resize((size_t)w_);
    q_.resize((size_t)nk_ * (size_t)st_);
    k_.resize((size_t)nk_ * (size_t)st_);
    alpha_.resize((size_t)nv_);
    beta_.resize((size_t)nv_);
}

TForward::~TForward() { gguf::close(f_); }

std::vector<uint8_t> &TForward::memory(int kind) {
    if (kind < 0 || kind >= MB_COUNT || mem_ro_[kind]) dies("not a writable memory");
    return mem_[kind];
}

void TForward::one_layer(Layer &L) {
    const int ST = st_;
    const float over_h = 1.0f / (float)h_, over_st = 1.0f / (float)ST;
    float scale = q_rms_scale(x_.data(), h_, over_h, EPS);
    for (int i = 0; i < h_; i++) hn_[(size_t)i] = (x_[(size_t)i] * scale) * L.attn_norm[(size_t)i];
    bf16_all(hn_.data(), hn_.size());
    const int il = (int)(&L - ls_.data());
    if (probe_) probe_->on_hn(il, 0, hn_.data(), x_.data());
    gemv(L.qkv, hn_.data(), qkv_.data());
    gemv(L.gate, hn_.data(), z_.data());
    gemv(L.alpha, hn_.data(), alpha_.data());
    gemv(L.beta, hn_.data(), beta_.data());
    std::fill(conv_.begin(), conv_.end(), 0.0f);
    for (uint32_t c : L.convrows) {
        const float *w = &L.conv[c * 4ull];
        float acc = 0.0f;
        for (int t = 0; t < 3; t++) acc = Q_FMA(w[t], L.win[t][c], acc);
        acc = Q_FMA(w[3], qkv_[c], acc);
        conv_[c] = q_silu(bf16(acc));
    }
    for (uint32_t c : L.convrows) {
        L.win[0][c] = L.win[1][c];
        L.win[1][c] = L.win[2][c];
        L.win[2][c] = qkv_[c];
    }
    const float q_scale = 0.0883883461f, e2 = EPS * EPS;
    const int key = nk_ * ST;
    for (int kh = 0; kh < nk_; kh++) {
        const float *cq = &conv_[(size_t)(kh * ST)], *ck = &conv_[(size_t)(key + kh * ST)];
        const float sq = q_sum_sq(cq, ST), sk = q_sum_sq(ck, ST);
        const float rq = q_rsqrt(sq > e2 ? sq : e2), rk = q_rsqrt(sk > e2 ? sk : e2);
        for (int i = 0; i < ST; i++) {
            q_[(size_t)(kh * ST + i)] = (cq[i] * rq) * q_scale;
            k_[(size_t)(kh * ST + i)] = ck[i] * rk;
        }
    }
    std::fill(mix_.begin(), mix_.end(), 0.0f);
    const float *v = &conv_[(size_t)(2 * key)];
    const int group = nv_ / nk_;
    for (VState &vs : L.st) {
        const int hd = vs.hd, kh = hd / group;
        float *S = vs.S.data();
        const float *kk = &k_[(size_t)(kh * ST)], *qq = &q_[(size_t)(kh * ST)], *vv = v + hd * ST;
        const float e = q_exp(q_softplus(alpha_[(size_t)hd] + L.dt[(size_t)hd]) * L.a[(size_t)hd]);
        const float b = q_sigmoid(beta_[(size_t)hd]);
        // a decay of 1 and a key of zeros leave the state as it is
        if (e != 1.0f) {
            for (int i = 0; i < ST * ST; i++) S[i] = S[i] * e;
        }
        std::vector<float> o((size_t)ST, 0.0f);
        float kv[128] = {0};
        bool keyed = false;
        for (int i = 0; i < ST; i++)
            if (kk[i] != 0.0f) {
                keyed = true;
                const float *row = S + i * ST;
                for (int j = 0; j < ST; j++) kv[j] = Q_FMA(row[j], kk[i], kv[j]);
            }
        if (keyed) {
            for (int i = 0; i < ST; i++)
                if (kk[i] != 0.0f) {
                    float *row = S + i * ST;
                    for (int j = 0; j < ST; j++) row[j] = Q_FMA(kk[i], (vv[j] - kv[j]) * b, row[j]);
                }
        }
        for (int i = 0; i < ST; i++)
            if (qq[i] != 0.0f) {
                const float *row = S + i * ST;
                for (int j = 0; j < ST; j++) o[(size_t)j] = Q_FMA(row[j], qq[i], o[(size_t)j]);
            }
        const float *zz = &z_[(size_t)(hd * ST)];
        const float sc = q_rms_scale(o.data(), ST, over_st, EPS);
        float *m = &mix_[(size_t)(hd * ST)];
        for (int j = 0; j < ST; j++) m[j] = bf16(bf16((o[(size_t)j] * sc) * L.ssm_norm[(size_t)j]) * q_silu(zz[j]));
    }
    if (probe_) probe_->on_mix(il, mix_.data(), (int)mix_.size());
    gemv(L.out, mix_.data(), y_.data());
    bf16_all(y_.data(), y_.size());
    for (Bank &b : L.banks) run_bank(b);
    if (probe_) probe_->on_delta(il, 0, y_.data());
    for (int i = 0; i < h_; i++) x_[(size_t)i] = x_[(size_t)i] + y_[(size_t)i];

    scale = q_rms_scale(x_.data(), h_, over_h, EPS);
    for (int i = 0; i < h_; i++) hn_[(size_t)i] = (x_[(size_t)i] * scale) * L.post_norm[(size_t)i];
    bf16_all(hn_.data(), hn_.size());
    if (probe_) probe_->on_hn(il, 1, hn_.data(), x_.data());
    float dot = 0.0f;
    for (int i = 0; i < h_; i++)
        if (L.gate_inp[(size_t)i] != 0.0f) dot = Q_FMA(L.gate_inp[(size_t)i], hn_[(size_t)i], dot);
    const float gate_s = q_sigmoid(dot);
    gemv(L.f_gate, hn_.data(), g_.data());
    gemv(L.f_up, hn_.data(), u_.data());
    // a unit whose gate row is empty has silu(0) = 0: its value is 0, never a stale one from another layer
    std::fill(xs_.begin(), xs_.end(), 0.0f);
    for (size_t k = 0; k + 1 < L.f_gate.start.size(); k++) {
        const uint32_t r = L.f_gate.row[k];
        xs_[r] = bf16((q_silu(g_[r]) * u_[r]) * gate_s);
    }
    if (probe_) probe_->on_units(il, xs_.data(), (int)xs_.size());
    gemv(L.f_down, xs_.data(), y_.data());
    bf16_all(y_.data(), y_.size());
    if (probe_) probe_->on_delta(il, 1, y_.data());
    for (int i = 0; i < h_; i++) x_[(size_t)i] = x_[(size_t)i] + y_[(size_t)i];
}

// A memory held in one piece: the heads' semantics, writes and then reads in the same token. A bit is a row's value being
// nonzero; a write's bits come through their taps over this and the last three passes, so a value computed in one pass
// may land in the next. Every tap window advances every pass, whether the write fires or not.
void TForward::run_bank(Bank &b) {
    uint8_t *mem = mem_[(size_t)b.kind].data();
    const uint8_t *rmem = mem_ro_[(size_t)b.kind] ? mem_ro_[(size_t)b.kind] : mem;
    const uint64_t n = mem_n_[(size_t)b.kind];
    gemv(b.ra, hn_.data(), b.vra.data());
    if (!b.writes.empty()) {
        gemv(b.wa, hn_.data(), b.vwa.data());
        gemv(b.wd, hn_.data(), b.vd.data());
        gemv(b.we, hn_.data(), b.vwe.data());
    }
    for (size_t wi = 0; wi < b.writes.size(); wi++) {
        const BWrite &w = b.writes[wi];
        uint32_t addr = 0, data = 0;
        for (int i = 0; i < w.nabits; i++) {
            const size_t k = (size_t)(w.aoff + i);
            const float cur = b.vwa[k];
            const float v = w.ataps[0] * cur + w.ataps[1] * b.hwa[0][k] + w.ataps[2] * b.hwa[1][k] + w.ataps[3] * b.hwa[2][k];
            if (v != 0.0f && i < 32) addr |= 1u << i;
        }
        for (int i = 0; i < w.ndbits; i++) {
            const size_t k = (size_t)(w.doff + i);
            const float *t = &b.taps[k * 4];
            const float cur = b.vd[k];
            const float v = t[0] * cur + t[1] * b.hd[0][k] + t[2] * b.hd[1][k] + t[3] * b.hd[2][k];
            if (v != 0.0f && i < 32) data |= 1u << i;
        }
        if (b.vwe[wi] != 0.0f) {
            if (!mem) dies("a write to a read-only memory");
            membits_write(b.kind, w.width, addr, data, mem, n);
        }
    }
    if (!b.writes.empty()) {
        for (int h = 2; h > 0; h--) {
            b.hwa[(size_t)h].swap(b.hwa[(size_t)h - 1]);
            b.hd[(size_t)h].swap(b.hd[(size_t)h - 1]);
        }
        b.hwa[0] = b.vwa;
        b.hd[0] = b.vd;
    }
    uint8_t bits[MBW_ROMBLK];
    for (const BRead &r : b.reads) {
        uint32_t addr = 0;
        for (int i = 0; i < r.nbits && i < 32; i++)
            if (b.vra[(size_t)(r.aoff + i)] != 0.0f) addr |= 1u << i;
        const uint32_t got = membits_read(b.kind, r.width, addr, rmem, n, bits);
        for (int i = 0; i < r.width; i++) y_[(size_t)b.out[(size_t)(r.ooff + i)]] += (uint32_t)i < got ? (float)bits[i] : 0.0f;
    }
}

const std::vector<float> &TForward::step(uint32_t token) {
    const gguf::Tensor &emb = f_.tensor("chip.token_embd.weight");
    gguf::read_at(f_, emb.offset + (uint64_t)token * (uint64_t)h_ * 4, x_.data(), (uint64_t)h_ * 4);
    for (size_t i = 0; i < ls_.size(); i++) {
        if ((int)i == keep_at_) before_ = x_;
        one_layer(ls_[i]);
    }
    if (keep_at_ == (int)ls_.size()) before_ = x_;
    return x_;
}

// The output: tied to the embedding, logit = row . x. With few dims left in x the rows are read on those dims only. A row whose dot
// reaches the sum of the largest |row[d]| * |x[d]| has the largest logit there can be, so the argmax needs no scan of the others; the
// table only finds that row by its signs. It holds for any tensors, and a miss falls back to the full scan.
uint32_t TForward::next_token() {
    const gguf::Tensor &emb = f_.tensor("chip.token_embd.weight");
    const uint64_t vocab = emb.dims[1], rowb = (uint64_t)h_ * 4;
    // Dims no row of the embedding touches add exactly 0 to every logit: found once, by one pass over the rows.
    if (row_used_.empty()) {
        row_used_.assign((size_t)h_, 0);
        std::vector<float> rv((size_t)h_);
        for (uint64_t t = 0; t < vocab; t++) {
            gguf::read_at(f_, emb.offset + t * rowb, rv.data(), rowb);
            for (int i = 0; i < h_; i++) row_used_[(size_t)i] |= rv[(size_t)i] != 0.0f;
        }
    }
    std::vector<int> dims;
    for (int i = 0; i < h_; i++)
        if (x_[(size_t)i] != 0.0f) {
            if (row_used_[(size_t)i]) dims.push_back(i);
            else if (strays_seen_.insert(i).second && strays_seen_.size() <= 64) std::fprintf(stderr, "tforward: residual dim %d = %.9g reaches the output and no row reads it\n", i, (double)x_[(size_t)i]);
        }
    if (dims.empty() || dims.size() > 64) dies("the output is not a code");
    if (dims != out_dims_all_) {
        // the first sight of this support: the union grows, the table is rebuilt only if a new dim appears
        std::vector<int> u = out_dims_;
        for (int d : dims)
            if (std::find(u.begin(), u.end(), d) == u.end()) u.push_back(d);
        std::sort(u.begin(), u.end());
        if (u != out_dims_) {
            out_dims_ = u;
            out_table_.clear();
            out_max_.assign(u.size(), 0.0f);
            const int lo = u.front(), hi = u.back();
            std::vector<float> part((size_t)(hi - lo + 1));
            for (uint64_t t = 0; t < vocab; t++) {
                gguf::read_at(f_, emb.offset + t * rowb + (uint64_t)lo * 4, part.data(), part.size() * 4);
                uint64_t key = 0;
                for (size_t k = 0; k < u.size(); k++) {
                    const float v = part[(size_t)(u[k] - lo)];
                    key = key << 1 | (uint64_t)(v > 0.0f);
                    out_max_[k] = std::max(out_max_[k], std::fabs(v));
                }
                out_table_.emplace(key, (uint32_t)t);
            }
        }
        out_dims_all_ = dims;
    }
    uint64_t key = 0;
    double bound = 0;
    for (size_t k = 0; k < out_dims_.size(); k++) {
        const float v = x_[(size_t)out_dims_[k]];
        key = key << 1 | (uint64_t)(v > 0.0f);
        bound += (double)out_max_[k] * std::fabs((double)v);
    }
    const auto it = out_table_.find(key);
    if (it != out_table_.end()) {
        const int lo = out_dims_.front();
        std::vector<float> part((size_t)(out_dims_.back() - lo + 1));
        gguf::read_at(f_, emb.offset + (uint64_t)it->second * rowb + (uint64_t)lo * 4, part.data(), part.size() * 4);
        double dot = 0;
        for (int d : out_dims_) dot += (double)part[(size_t)(d - lo)] * (double)x_[(size_t)d];
        if (dot == bound) return it->second;
    }
    // no row reaches the bound: the argmax itself
    uint32_t best = 0;
    double top = -1e300;
    std::vector<float> rowv((size_t)h_);
    for (uint64_t t = 0; t < vocab; t++) {
        gguf::read_at(f_, emb.offset + t * rowb, rowv.data(), rowb);
        double dot = 0;
        for (int d : dims) dot += (double)rowv[(size_t)d] * (double)x_[(size_t)d];
        if (dot > top) {
            top = dot;
            best = (uint32_t)t;
        }
    }
    full_scans_++;
    return best;
}
