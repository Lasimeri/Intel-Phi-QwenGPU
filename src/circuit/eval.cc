// See eval.h.
#include "eval.h"

#include "../exec/membits.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

// The worker pool: the unit nodes of one level, handed out in small runs; the main thread works too and waits for all.
struct Evaluator::Pool {
    std::vector<std::thread> workers;
    std::atomic<uint64_t> gen{0};
    std::atomic<size_t> next{0};
    std::atomic<int> done{0};
    std::atomic<bool> quit{false};
    const std::vector<int> *work = nullptr;
    Evaluator *ev = nullptr;
    int n = 0;
    void start(int nthreads) {
        n = nthreads;
        for (int i = 0; i < n; i++) workers.emplace_back([this] { loop(); });
    }
    void loop() {
        uint64_t seen = 0;
        for (;;) {
            uint64_t g;
            int spins = 0;
            while ((g = gen.load(std::memory_order_acquire)) == seen) {
                if (quit.load(std::memory_order_acquire)) return;
                if (++spins > 4000) {
                    std::this_thread::yield();
                    spins = 0;
                }
            }
            seen = g;
            drain();
            done.fetch_add(1, std::memory_order_release);
        }
    }
    void drain() {
        const std::vector<int> &w = *work;
        for (;;) {
            const size_t i = next.fetch_add(8, std::memory_order_relaxed);
            if (i >= w.size()) return;
            const size_t e = std::min(i + 8, w.size());
            for (size_t k = i; k < e; k++) ev->eval_ffn(ev->nodes_[(size_t)w[k]]);
        }
    }
    void run(const std::vector<int> &w) {
        work = &w;
        next.store(0, std::memory_order_relaxed);
        done.store(0, std::memory_order_relaxed);
        gen.fetch_add(1, std::memory_order_release);
        drain();
        while (done.load(std::memory_order_acquire) < n) {
        }
    }
    void stop() {
        quit.store(true, std::memory_order_release);
        for (std::thread &t : workers) t.join();
    }
};

namespace {

constexpr double RMS_DATA_LIMIT = 1.0 / (1 << 26);  // data energy absorbed next to the ballast
constexpr double RESIDUAL_MAX = 268435456.0;        // 2^28: x^2 is lost in fp32 next to m * 2^80

double bf16(double x) {
    float f = (float)x;
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7FFF + ((u >> 16) & 1);
    u &= 0xFFFF0000u;
    std::memcpy(&f, &u, 4);
    return (double)f;
}

bool is_bf16(double x) { return std::isfinite(x) && bf16(x) == x; }

bool silu_ok(double x) { return x == 0 || x >= 18 || x <= -104; }

[[noreturn]] void exact_fail(uint64_t pos, const char *node, const char *what, double v) {
    std::fprintf(stderr, "ExactnessError at pass %llu, node %s: %s: bad value %.17g\n", (unsigned long long)pos, node, what, v);
    std::exit(4);
}

}  // namespace

LinF Evaluator::flat(const Expr &e) {
    LinF f;
    f.off = (uint32_t)fi_.size();
    f.n = (uint32_t)e.t.size();
    for (const auto &p : e.t) {
        fi_.push_back((uint32_t)p.first);
        fc_.push_back(p.second.to_double());
    }
    return f;
}

double Evaluator::ev(LinF f) const {
    double s = 0;
    for (uint32_t i = f.off; i < f.off + f.n; i++) s += m_[fi_[i]] * fc_[i];
    return s;
}

double Evaluator::bit(LinF f, const char *node, const char *what) const {
    const double v = ev(f);
    if (check_ && v != 0.0 && v != 1.0) exact_fail(pos_, node, what, v);
    return v;
}

Evaluator::~Evaluator() {
    if (pool_) {
        pool_->stop();
        delete pool_;
    }
}

Evaluator::Evaluator(const Circ &c, bool check, int threads) : c_(c), check_(check) {
    m_.assign(c.dims.size(), 0.0);
    nodes_.reserve(c.nodes.size());
    auto chan_of = [&](int col, const Chan &ch) {
        EChan e;
        e.col = col;
        e.f = flat(ch.expr);
        e.taps = ch.taps;
        e.lagged = ch.taps[1] != 0 || ch.taps[2] != 0 || ch.taps[3] != 0;
        e.h[0] = e.h[1] = e.h[2] = 0;
        return e;
    };
    auto lag_of = [&](const Expr &e, Taps taps) {
        ELag l;
        l.f = flat(e);
        l.taps = taps;
        l.h[0] = l.h[1] = l.h[2] = 0;
        return l;
    };
    for (const Node &n : c.nodes) {
        ENode en;
        en.kind = n.is_bank ? 2 : (n.is_ffn ? 0 : 1);
        en.name = n.name.c_str();
        en.out = n.out;
        en.into = n.into;
        en.tiny = n.tiny;
        en.mem = n.bank.mem;
        if (n.is_bank) {
            for (const BankRead &r : n.bank.reads) {
                ERead er;
                for (const Expr &a : r.addr) er.addr.push_back(flat(a));
                er.out = r.out;
                er.width = r.width;
                en.reads.push_back(std::move(er));
            }
            for (const BankWrite &w : n.bank.writes) {
                EWrite ew;
                for (const Expr &a : w.addr) ew.addr.push_back(lag_of(a, w.addr_taps));
                for (size_t i = 0; i < w.data.size(); i++) ew.data.push_back(lag_of(w.data[i], i < w.data_taps.size() ? w.data_taps[i] : LAG0));
                ew.we = flat(w.we);
                ew.width = w.width;
                en.writes.push_back(std::move(ew));
            }
        } else if (n.is_ffn) {
            for (const Term &t : n.terms) en.units.push_back(EUnit{t.c.to_double(), flat(t.a), flat(t.b)});
            units_ += n.terms.size();
        } else {
            for (const auto &p : n.q) en.q.push_back(chan_of(p.first, p.second));
            for (const auto &p : n.k) en.k.push_back(chan_of(p.first, p.second));
            for (const VHead &vh : n.vh) {
                EVHead ev_;
                for (const auto &p : vh.v) ev_.v.push_back(chan_of(p.first, p.second));
                ev_.a = flat(vh.a);
                ev_.has_beta = vh.beta.has_value();
                if (vh.beta) ev_.beta = flat(*vh.beta);
                for (const auto &p : vh.z) ev_.z.emplace_back(p.first, flat(p.second));
                ev_.out = vh.out;
                ev_.into = vh.into;
                ev_.S.assign((size_t)HD * HD, 0.0);
                ev_.live.assign(HD, 0);
                en.vh.push_back(std::move(ev_));
            }
        }
        nodes_.push_back(std::move(en));
    }
    for (int i = 0; i < NFAM && i < (int)c.out_fam.size(); i++) out_fam_[i] = flat(c.out_fam[(size_t)i]);
    for (int i = 0; i < NPAY && i < (int)c.out_pay.size(); i++) out_pay_[i] = flat(c.out_pay[(size_t)i]);
    order_.resize(c.nodes.size());
    for (size_t i = 0; i < order_.size(); i++) order_[i] = (int)i;
    std::stable_sort(order_.begin(), order_.end(), [&](int a, int b) { return c.nodes[(size_t)a].level < c.nodes[(size_t)b].level; });
    // the levels: unit nodes without `into` in parallel, the rest in creation order
    int cur = -1;
    for (int idx : order_) {
        const Node &n = c.nodes[(size_t)idx];
        if (n.level != cur) {
            cur = n.level;
            levels_.emplace_back();
            levels_ser_.emplace_back();
        }
        if (n.is_ffn && !n.is_bank && !n.into) levels_.back().push_back(idx);
        else levels_ser_.back().push_back(idx);
    }
    for (auto &m : mem_) m.clear();
    if (threads <= 0) {
        const int hw = (int)std::thread::hardware_concurrency();
        threads = std::min(14, std::max(1, hw - 2));
    }
    if (threads > 1) {
        pool_ = new Pool;
        pool_->ev = this;
        pool_->start(threads - 1);
    }
}

void Evaluator::run_ffn_level(const std::vector<int> &nodes) {
    if (pool_ && nodes.size() >= 24) {
        pool_->run(nodes);
        return;
    }
    for (int idx : nodes) eval_ffn(nodes_[(size_t)idx]);
}

double Evaluator::conv(EChan &c, const char *node) {
    double x = ev(c.f);
    if (check_ && !is_bf16(x)) exact_fail(pos_, node, "projection", x);
    double out = c.taps[0] * x;
    if (c.lagged) {
        out += c.taps[1] * c.h[0] + c.taps[2] * c.h[1] + c.taps[3] * c.h[2];
        c.h[2] = c.h[1];
        c.h[1] = c.h[0];
        c.h[0] = x;
    }
    if (check_ && !(is_bf16(out) && silu_ok(out))) exact_fail(pos_, node, "conv output (silu input)", out);
    return out > 0 ? out : 0.0;
}

double Evaluator::lagged(ELag &l) const {
    double x = ev(l.f);
    double out = l.taps[0] * x + l.taps[1] * l.h[0] + l.taps[2] * l.h[1] + l.taps[3] * l.h[2];
    l.h[2] = l.h[1];
    l.h[1] = l.h[0];
    l.h[0] = x;
    return out;
}

void Evaluator::eval_ffn(ENode &n) {
    double ideal = 0, hw = 0;
    for (const EUnit &u : n.units) {
        double bv = ev(u.b), av = ev(u.a);
        ideal += u.c * av * (bv > 0 ? bv : 0.0);
        if (check_) {
            double gate = bf16(K_GATE * bv);
            if (!silu_ok(gate)) exact_fail(pos_, n.name, "gate outside silu's exact range", gate);
            double up = bf16(av / K_GATE);
            hw += u.c * bf16((gate > 0 ? gate : 0.0) * up);
        }
    }
    if (check_) {
        hw = bf16(hw);
        if (hw != ideal) exact_fail(pos_, n.name, "bf16 output differs from exact", hw - ideal);
    }
    if (n.into) {
        ideal = m_[(size_t)n.out] + ideal;
        if (check_ && !is_bf16(ideal)) exact_fail(pos_, n.name, "residual add", ideal);
    }
    m_[(size_t)n.out] = ideal;
}

void Evaluator::eval_head(ENode &n) {
    double kmax = 0;
    int qslot[HD], nq = 0, key_row = -1, nkey = 0;
    double q0 = 0;
    for (EChan &ch : n.q) {
        double v = conv(ch, n.name);
        if (v > 0) {
            if (check_ && nq && v != q0) exact_fail(pos_, n.name, "query must be multi-hot", v);
            q0 = v;
            qslot[nq++] = ch.col;
        }
    }
    for (EChan &ch : n.k) {
        double v = conv(ch, n.name);
        if (v > 0) {
            nkey++;
            if (v > kmax) {
                kmax = v;
                key_row = ch.col;
            }
        }
    }
    if (check_ && nkey > 1) exact_fail(pos_, n.name, "key must be one-hot", nkey);
    for (EVHead &vh : n.vh) {
        double v[HD];
        std::fill(v, v + HD, 0.0);
        for (EChan &ch : vh.v) v[ch.col] = conv(ch, n.name);
        double a = ev(vh.a);
        double beta = vh.has_beta ? ev(vh.beta) : 1.0;
        bool writes = key_row >= 0 && beta > 0;
        double decay;
        if (a == 0.0) decay = 1.0;
        else if (a == 110.0) decay = 0.5;
        else if (a == 238.0) decay = 0.0;
        else exact_fail(pos_, n.name, "decay selector a", a);
        if (check_) {
            if (!(beta == 1.0 || (beta <= 0 && beta == std::round(beta)))) exact_fail(pos_, n.name, "write gate beta", beta);
            if (writes) {
                for (int col = 2; col < HD; col++)
                    if (v[col] != 0.0 && v[col] != (double)A_BIT) exact_fail(pos_, n.name, "state row data column", v[col]);
                for (int col = 0; col < 2; col++)
                    if (v[col] != 0.0 && v[col] != (double)A_REF) exact_fail(pos_, n.name, "state row ballast column", v[col]);
            }
        }
        if (decay == 0.0) {
            std::fill(vh.S.begin(), vh.S.end(), 0.0);
            std::fill(vh.live.begin(), vh.live.end(), 0);
        } else if (decay == 0.5) {
            for (int r = 0; r < HD; r++)
                if (vh.live[(size_t)r])
                    for (int col = 0; col < HD; col++) vh.S[(size_t)(r * HD + col)] *= 0.5;
        }
        if (writes) {
            std::memcpy(&vh.S[(size_t)(key_row * HD)], v, sizeof v);
            vh.live[(size_t)key_row] = 1;
        }
        double out[HD];
        std::fill(out, out + HD, 0.0);
        if (nq) {
            double o[HD];
            std::fill(o, o + HD, 0.0);
            for (int i = 0; i < nq; i++) {
                const double *row = &vh.S[(size_t)(qslot[i] * HD)];
                for (int col = 0; col < HD; col++) o[col] += row[col];
            }
            double ref = o[0] * o[0] + o[1] * o[1];
            if (ref == 0) {
                if (check_)
                    for (int col = 0; col < HD; col++)
                        if (o[col] != 0) exact_fail(pos_, n.name, "read without ballast", o[col]);
            } else {
                if (check_) {
                    double data = 0;
                    for (int col = 2; col < HD; col++) data += o[col] * o[col];
                    if (data > ref * RMS_DATA_LIMIT) exact_fail(pos_, n.name, "data not absorbed by ballast", data / ref);
                }
                double d = std::sqrt(ref / HD);
                for (int col = 0; col < HD; col++) out[col] = o[col] / d;
            }
        }
        double zv[HD];
        std::fill(zv, zv + HD, 0.0);
        for (const auto &p : vh.z) {
            double zz = ev(p.second);
            if (check_ && !(is_bf16(zz) && silu_ok(zz))) exact_fail(pos_, n.name, "gate z", zz);
            zv[p.first] = zz;
        }
        for (const Out &x : vh.out) {
            double zz = zv[x.col];
            double val = out[x.col] * (zz > 0 ? zz : 0.0);
            if (check_) {
                bool small = std::fabs(out[x.col]) < n.tiny;
                if (!small && !is_bf16(out[x.col])) exact_fail(pos_, n.name, "normed column", out[x.col]);
                if (!small && !is_bf16(val * x.w)) exact_fail(pos_, n.name, "head output column", val * x.w);
            }
            if (vh.into) {
                m_[(size_t)x.dim] += val * x.w;
                if (check_ && !is_bf16(m_[(size_t)x.dim])) exact_fail(pos_, n.name, "head output sum", m_[(size_t)x.dim]);
            } else {
                m_[(size_t)x.dim] = val * x.w;
            }
        }
    }
}

// --- banks -----------------------------------------------------------------------------------------------

static inline uint32_t bits_to_u32_unused(const uint8_t *b, int n) __attribute__((unused));
static inline uint32_t bits_to_u32_unused(const uint8_t *b, int n) {
    uint32_t v = 0;
    for (int i = 0; i < n; i++) v |= (uint32_t)(b[i] & 1) << i;
    return v;
}

// the element forms are the file's (src/exec/membits.h): the forward over the file selects the same bytes
static_assert((int)MEM_M == MB_WORDS && (int)MEM_A == MB_ABLK && (int)MEM_S == MB_HALF && (int)MEM_C == MB_BYTE && (int)MEM_W == MB_WBLK && (int)MEM_ROM == MB_ROM && (int)MEM_REG == MB_REG && (int)MEM_UREG == MB_UREG &&
                  (int)MEM_F16 == MB_F16 && (int)MEM_SCR == MB_SCR && (int)MEM_COUNT == MB_COUNT,
              "the circuit's memory kinds are the file's");
uint64_t Evaluator::read_bits(int mem, int width, uint32_t addr, uint8_t *bits) const {
    if (mem == MEM_W) return membits_read(MB_WBLK, width, addr, w_, (uint64_t)w_blocks_ * 34, bits);
    const std::vector<uint8_t> &M = mem_[(size_t)mem];
    return membits_read(mem, width, addr, M.data(), M.size(), bits);
}
void Evaluator::write_bits(int mem, int width, uint32_t addr, uint32_t value) {
    std::vector<uint8_t> &M = mem_[(size_t)mem];
    if (membits_write(mem, width, addr, value, M.data(), M.size())) exact_fail(pos_, "bank", "a write to a read-only memory", (double)mem);
}

void Evaluator::eval_bank(ENode &n) {
    // writes land before the reads
    for (EWrite &w : n.writes) {
        const double we = bit(w.we, n.name, "write enable");
        uint32_t addr = 0, data = 0;
        for (size_t i = 0; i < w.addr.size(); i++) {
            const double v = lagged(w.addr[i]);
            if (check_ && v != 0.0 && v != 1.0) exact_fail(pos_, n.name, "write address bit", v);
            if (v != 0.0 && i < 32) addr |= 1u << i;
        }
        for (size_t i = 0; i < w.data.size(); i++) {
            const double v = lagged(w.data[i]);
            if (check_ && v != 0.0 && v != 1.0) exact_fail(pos_, n.name, "write data bit", v);
            if (v != 0.0 && i < 32) data |= 1u << i;
        }
        if (we != 0.0) {
            if (n.mem == MEM_M && (int64_t)addr == watch_) std::printf("circuit: pass %llu store M[%u] = %u (%s, width %d)\n", (unsigned long long)pos_, addr, data, n.name, w.width);
            write_bits(n.mem, w.width, addr, data);
        }
    }
    uint8_t bits[PW_ROMBLK];
    for (ERead &r : n.reads) {
        uint32_t addr = 0;
        for (size_t i = 0; i < r.addr.size(); i++) {
            const double v = bit(r.addr[i], n.name, "read address bit");
            if (v != 0.0 && i < 32) addr |= 1u << i;
        }
        if (n.mem == MEM_F16 && f16_ && f16_[addr & 0xFFFF] != (int64_t)(int32_t)f16_[addr & 0xFFFF]) exact_fail(pos_, n.name, "an f16 scale outside int32 Q.24", (double)f16_[addr & 0xFFFF]);
        const uint64_t got = read_bits(n.mem, r.width, addr, bits);
        for (size_t i = 0; i < r.out.size(); i++) m_[(size_t)r.out[i]] = i < got ? (double)bits[i] : 0.0;
    }
}

void Evaluator::step(uint32_t token) {
    std::fill(m_.begin(), m_.end(), 0.0);
    m_[ONE] = 1.0;
    for (size_t i = 0; i < c_.feat_dim.size(); i++) {
        double v = c_.feat_fn[i](token);
        if (check_ && !is_bf16(v)) exact_fail(pos_, c_.dims[(size_t)c_.feat_dim[i]].name.c_str(), "token feature", v);
        m_[(size_t)c_.feat_dim[i]] = v;
    }
    for (size_t lv = 0; lv < levels_.size(); lv++) {
        run_ffn_level(levels_[lv]);
        for (int idx : levels_ser_[lv]) {
            ENode &n = nodes_[(size_t)idx];
            if (n.kind == 0) eval_ffn(n);
            else if (n.kind == 1) eval_head(n);
            else eval_bank(n);
        }
    }
    if (check_) {
        for (size_t d = 1; d < m_.size(); d++)
            if (std::fabs(m_[d]) > RESIDUAL_MAX) exact_fail(pos_, c_.dims[d].name.c_str(), "residual value would move the norms", m_[d]);
    }
    pos_++;
}

int64_t Evaluator::predict() const {
    int fid = 0;
    uint32_t p = 0;
    for (int i = 0; i < NFAM; i++) fid |= (int)std::rint(ev(out_fam_[i])) << i;
    for (int i = 0; i < NPAY; i++) p |= (uint32_t)std::rint(ev(out_pay_[i])) << i;
    if (fid < 0 || fid >= F_COUNT) return -1;
    if (p >= (1u << PAY_BITS[fid]) && PAY_BITS[fid] > 0) return -1;
    if (PAY_BITS[fid] == 0) p = 0;
    return (int64_t)tok_of(fid, p);
}

// --- the starting state --------------------------------------------------------------------------------------

void Evaluator::load(const SmImage &img) {
    for (ENode &n : nodes_) {
        for (EVHead &vh : n.vh) {
            std::fill(vh.S.begin(), vh.S.end(), 0.0);
            std::fill(vh.live.begin(), vh.live.end(), 0);
            for (EChan &ch : vh.v) ch.h[0] = ch.h[1] = ch.h[2] = 0;
        }
        for (EChan &ch : n.q) ch.h[0] = ch.h[1] = ch.h[2] = 0;
        for (EChan &ch : n.k) ch.h[0] = ch.h[1] = ch.h[2] = 0;
        for (EWrite &w : n.writes) {
            for (ELag &l : w.addr) l.h[0] = l.h[1] = l.h[2] = 0;
            for (ELag &l : w.data) l.h[0] = l.h[1] = l.h[2] = 0;
        }
    }
    pos_ = 0;
    mem_[MEM_M].assign((size_t)img.m_words * 4, 0);
    if (img.m_words) std::memcpy(mem_[MEM_M].data(), img.m, (size_t)img.m_words * 4);
    mem_[MEM_A].assign((size_t)img.a_blocks * 36, 0);
    if (img.a_blocks) std::memcpy(mem_[MEM_A].data(), img.a, (size_t)img.a_blocks * 36);
    mem_[MEM_S].assign((size_t)img.s_elems * 2, 0);
    if (img.s_elems) std::memcpy(mem_[MEM_S].data(), img.s, (size_t)img.s_elems * 2);
    mem_[MEM_C].assign((size_t)img.c_bytes, 0);
    if (img.c_bytes) std::memcpy(mem_[MEM_C].data(), img.c, (size_t)img.c_bytes);
    w_ = img.w;
    w_blocks_ = img.w_blocks;
    // the f16 table as 65536 words of Q.24; an entry outside int32 (an f16 of 128 or more, inf, NaN) is stored truncated and a
    // read of it is a fault here: no Q8_0 block of the 2B has such a scale, so the file never reads one
    f16_ = img.f16_q24;
    mem_[MEM_F16].assign(65536 * 4, 0);
    if (img.f16_q24)
        for (uint32_t i = 0; i < 65536; i++) {
            const int64_t v = img.f16_q24[i];
            const int32_t w = (int32_t)(uint32_t)(uint64_t)v;
            std::memcpy(&mem_[MEM_F16][(size_t)i * 4], &w, 4);
        }
    mem_[MEM_ROM].assign((size_t)img.nblk * 68, 0);
    for (uint32_t b = 0; b < img.nblk; b++) {
        uint8_t *dst = &mem_[MEM_ROM][(size_t)b * 68];
        for (int s = 0; s < SM_STAGES; s++) std::memcpy(dst + 10 * s, img.rom[b].rows[s], 10);
        std::memcpy(dst + 60, &img.rom[b].live, 8);
    }
    mem_[MEM_REG].assign((size_t)SM_WPB_MAX * SM_LANES * SM_NREG * 4, 0);
    mem_[MEM_UREG].assign((size_t)SM_WPB_MAX * SM_NUREG * 4, 0);
    mem_[MEM_SCR].assign(128 * 4, 0);
}

std::vector<float> Evaluator::export_state() const {
    std::vector<float> out;
    for (const ENode &n : nodes_)
        for (const EVHead &vh : n.vh)
            for (double d : vh.S) {
                float f = (float)d;
                if ((double)f != d) Dy::die("a state value is not exact in F32");
                out.push_back(f);
            }
    return out;
}
