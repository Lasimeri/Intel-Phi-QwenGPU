// See gate.h.
#include "gate.h"

#include <algorithm>
#include <cmath>

// --- tokens ---------------------------------------------------------------------------------------

Fam famof(uint32_t tok) {
    if (tok < SM_VHI0) return Fam{F_VLO, tok};
    if (tok < SM_VEND0) return Fam{F_VHI, tok - SM_VHI0};
    if (tok < SM_VAL_MAX) return Fam{F_VEND, tok - SM_VEND0};
    if (tok < SM_NEXT0) return Fam{F_EXEC, tok - SM_EXEC0};
    if (tok < SM_EMIT) return Fam{F_NEXT, tok - SM_NEXT0};
    if (tok == SM_EMIT) return Fam{F_EMIT, 0};
    if (tok == SM_HALT) return Fam{F_HALT, 0};
    if (tok == SM_GO) return Fam{F_GO, 0};
    if (tok == SM_LSTEP) return Fam{F_LSTEP, 0};
    if (tok == SM_TEXT) return Fam{F_TEXT, 0};
    if (tok >= SM_WARP0 && tok < SM_CHIP_VOCAB) return Fam{F_WARP, tok - SM_WARP0};
    return Fam{-1, 0};
}

uint32_t tok_of(int fam, uint32_t p) {
    switch (fam) {
    case F_VLO: return SM_VLO0 + p;
    case F_VHI: return SM_VHI0 + p;
    case F_VEND: return SM_VEND0 + p;
    case F_EXEC: return SM_EXEC0 + p;
    case F_NEXT: return SM_NEXT0 + p;
    case F_WARP: return SM_WARP0 + p;
    case F_EMIT: return SM_EMIT;
    case F_HALT: return SM_HALT;
    case F_GO: return SM_GO;
    case F_LSTEP: return SM_LSTEP;
    default: return SM_TEXT;
    }
}

static const char *const FAM_NAMES[F_COUNT] = {"VLO", "VHI", "VEND", "EXEC", "NEXT", "WARP", "EMIT", "HALT", "GO", "LSTEP", "TEXT"};

Circ::Circ(Target t) : Graph(t) {
    auto f = [&](const std::string &n, std::function<double(uint32_t)> fn) { return token_feature(n, std::move(fn)); };
    for (int i = 0; i < F_COUNT; i++) fam[i] = f(std::string("is") + FAM_NAMES[i], [i](uint32_t tk) { return (double)(famof(tk).id == i); });
    pay.resize(F_COUNT);
    for (int i = 0; i < F_COUNT; i++)
        for (int b = 0; b < PAY_BITS[i]; b++)
            pay[(size_t)i].push_back(f(std::string(FAM_NAMES[i]) + ".b" + std::to_string(b), [i, b](uint32_t tk) {
                const Fam fm = famof(tk);
                return fm.id == i ? (double)(fm.p >> b & 1) : 0.0;
            }));
    for (int i = 0; i < NFAM + NPAY; i++)
        code.push_back(f("code" + std::to_string(i), [i](uint32_t tk) {
            const Fam fm = famof(tk);
            const int bit = i < NFAM ? (fm.id < 0 ? 0 : fm.id >> i & 1) : (int)(fm.p >> (i - NFAM) & 1);
            return (double)(CODE_AMP * (2 * bit - 1));
        }));
}

// --- shared pieces ----------------------------------------------------------------------------------

L lits_of(const L &bits, long long v) {
    L r;
    r.reserve(bits.size());
    for (size_t i = 0; i < bits.size(); i++) r.push_back((v >> i & 1) ? bits[i] : Dy(1) - bits[i]);
    return r;
}

Expr andpre(const L &lits) { return sum(lits) - Dy((long long)lits.size() - 1); }

L nots(const L &bits) {
    L r;
    r.reserve(bits.size());
    for (const Expr &b : bits) r.push_back(Dy(1) - b);
    return r;
}

Expr nib(const L &bits, int k, bool inv) {
    Expr x = from_bits(sl(bits, 4 * k, 4 * k + 4));
    return inv ? Dy(15) - x : x;
}

std::vector<int> col_offsets(const std::vector<std::pair<int, int>> &bounds, int top) {
    std::vector<std::pair<int, int>> reach{{0, -1}};
    std::vector<std::vector<std::pair<int, int>>> hist;
    for (const auto &bd : bounds) {
        int pos = bd.first, neg = bd.second;
        std::vector<std::pair<int, int>> nxt;
        for (const auto &r : reach) {
            int gp = r.first;
            for (int g = 0; g < 64; g++) {
                int v = 16 * g - gp;
                if (neg <= v && v <= top - pos) {
                    bool seen = false;
                    for (const auto &x : nxt) seen = seen || x.first == g;
                    if (!seen) nxt.push_back({g, gp});
                }
            }
        }
        if (nxt.empty()) Dy::die("no column offsets");
        hist.push_back(nxt);
        reach = nxt;
    }
    int g = reach[0].first;
    for (const auto &x : reach) g = std::min(g, x.first);
    std::vector<int> offs;
    for (int i = (int)bounds.size() - 1; i >= 0; i--) {
        int gp = 0;
        for (const auto &x : hist[(size_t)i])
            if (x.first == g) gp = x.second;
        offs.push_back(16 * g - gp);
        g = gp;
    }
    std::reverse(offs.begin(), offs.end());
    return offs;
}

L select_bits(Circ &c, const std::vector<SelCase> &cases, int n, const std::string &name) {
    L out;
    for (int i = 0; i < n; i++) {
        Terms terms;
        for (const SelCase &cs : cases) {
            if (cs.is_int) {
                if (cs.ival >> i & 1) terms += c.t_and(cs.lits);
            } else if (i < (int)cs.bits.size()) {
                const Expr &b = cs.bits[(size_t)i];
                if (!b.empty()) terms += c.t_and(cs.lits + L{b});
            }
        }
        out.push_back(c.node(terms, name + ".b" + std::to_string(i)));
    }
    return out;
}

L mux_bits(Circ &c, const Expr &sel, const L &when1, const L &when0, const std::string &name) {
    L out;
    const size_t n = std::max(when1.size(), when0.size());
    for (size_t i = 0; i < n; i++) {
        Terms t;
        if (i < when1.size() && !when1[i].empty()) t += c.t_and(L{sel, when1[i]});
        if (i < when0.size() && !when0[i].empty()) t += c.t_and(L{Dy(1) - sel, when0[i]});
        out.push_back(c.node(t, name + ".b" + std::to_string(i)));
    }
    return out;
}

L and_bits(Circ &c, const L &bits, const Expr &lit, const std::string &name) {
    L r;
    for (size_t i = 0; i < bits.size(); i++) r.push_back(bits[i].empty() ? Expr() : c.and_(L{bits[i], lit}, name + ".b" + std::to_string(i)));
    return r;
}

// --- adders -----------------------------------------------------------------------------------------

Stage1 stage1_digits(Circ &c, const SumForms &sums, const std::string &name, long long hi) {
    Stage1 out;
    for (size_t k = 0; k < sums.size(); k++) {
        Terms gt, pt;
        std::vector<Terms> dt(4);
        for (const auto &form : sums[k]) {
            const Expr &s = form.first;
            const L &lits = form.second;
            gt += c.t_ge(s, 16, 0, hi, 1, lits);
            pt += c.t_eq(s, 15, 0, hi, 1, lits);
            for (int b = 0; b < 4; b++)
                for (long long theta = 1; theta <= hi; theta++) {
                    long long dd = (fmod_py(theta, 16) >> b & 1) - (fmod_py(theta - 1, 16) >> b & 1);
                    if (dd) dt[(size_t)b] += c.t_ge(s, theta, 0, hi, Dy(dd), lits);
                }
        }
        std::string ks = std::to_string(k);
        out.g.push_back(c.node(gt, name + ".g" + ks));
        out.p.push_back(c.node(pt, name + ".p" + ks));
        L d;
        for (int b = 0; b < 4; b++) d.push_back(c.node(dt[(size_t)b], name + ".d" + ks + "." + std::to_string(b)));
        out.d.push_back(std::move(d));
    }
    return out;
}

Stage1 stage1_cols(Circ &c, const std::vector<std::pair<Expr, long long>> &cols, const L &lits, const std::string &name) {
    Stage1 out;
    for (size_t k = 0; k < cols.size(); k++) {
        const Expr &s = cols[k].first;
        long long hi = cols[k].second;
        std::string ks = std::to_string(k);
        out.g.push_back(c.node(c.t_ge(s, 16, 0, hi, 1, lits), name + ".g" + ks));
        out.p.push_back(c.node(c.t_eq(s, 15, 0, hi, 1, lits), name + ".p" + ks));
        std::vector<Terms> dt(4);
        for (int b = 0; b < 4; b++)
            for (long long theta = 1; theta <= hi; theta++) {
                long long dd = (fmod_py(theta, 16) >> b & 1) - (fmod_py(theta - 1, 16) >> b & 1);
                if (dd) dt[(size_t)b] += c.t_ge(s, theta, 0, hi, Dy(dd), lits);
            }
        L d;
        for (int b = 0; b < 4; b++) d.push_back(c.node(dt[(size_t)b], name + ".d" + ks + "." + std::to_string(b)));
        out.d.push_back(std::move(d));
    }
    return out;
}

Adder adder_stage2(Circ &c, const L &g, const L &p_, const std::vector<L> &d, int n_nib, const std::string &name) {
    const Expr &one = c.one;
    auto carry_sources = [&](int k) {
        std::vector<L> v;
        for (int j = 0; j < k; j++) v.push_back(L{g[(size_t)j]} + sl(p_, j + 1, k));
        return v;
    };
    Adder a;
    for (int k = 0; k < n_nib; k++) {
        for (int b = 0; b < 4; b++) {
            const L &dk = d[(size_t)k];
            Terms terms = c.t_and(L{dk[(size_t)b]});
            for (const L &src : carry_sources(k)) {
                L low = sl(dk, 0, b);
                terms += c.t_and(src + low + L{one - dk[(size_t)b]});
                terms += c.t_and(src + low + L{dk[(size_t)b]}, -1);
            }
            a.r.push_back(c.node(terms, name + ".r" + std::to_string(4 * k + b)));
        }
    }
    Terms ct;
    for (const L &src : carry_sources(n_nib)) ct += c.t_and(src);
    a.cout = c.node(ct, name + ".cout");
    return a;
}

L add_nibbles(Circ &c, const L &a_bits, const L &b_bits, int n_nib, const L &lits, const std::string &name) {
    SumForms sums;
    for (int k = 0; k < n_nib; k++)
        sums.push_back({{from_bits(sl(a_bits, 4 * k, 4 * k + 4)) + from_bits(sl(b_bits, 4 * k, 4 * k + 4)), lits}});
    Stage1 s = stage1_digits(c, sums, name, 30);
    return adder_stage2(c, s.g, s.p, s.d, n_nib, name).r;
}

Adder add_full(Circ &c, const L &a_bits, const L &b_bits, const Expr &cin, bool inv_b, int n_nib, const std::string &name) {
    SumForms sums;
    for (int k = 0; k < n_nib; k++) {
        Expr s = nib(a_bits, k) + nib(b_bits, k, inv_b);
        if (k == 0 && !cin.empty()) s += cin;
        sums.push_back({{s, L{}}});
    }
    Stage1 s = stage1_digits(c, sums, name, 31);
    return adder_stage2(c, s.g, s.p, s.d, n_nib, name);
}

// --- the Circ's own ----------------------------------------------------------------------------------

std::vector<Terms> Circ::fn_bits(const Expr &z, long long lo, long long hi, const std::function<long long(long long)> &fn, int n, const L &lits) {
    std::vector<Terms> terms((size_t)n);
    for (int b = 0; b < n; b++) {
        if (fn(lo) >> b & 1) terms[(size_t)b] += lits.empty() ? t_const(1) : t_and(lits);
        for (long long theta = lo + 1; theta <= hi; theta++) {
            long long dd = (fn(theta) >> b & 1) - (fn(theta - 1) >> b & 1);
            if (dd) terms[(size_t)b] += t_ge(z, theta, lo, hi, Dy(dd), lits);
        }
    }
    return terms;
}

L Circ::zeros(int n, const std::string &name) {
    L r;
    r.reserve((size_t)n);
    for (int i = 0; i < n; i++) r.push_back(token_feature(name + "[" + std::to_string(i) + "]", [](uint32_t) { return 0.0; }));
    return r;
}

L Circ::table(const std::string &name, const std::map<int, Chan> &queries, const std::map<int, Chan> &keys, const std::vector<VSpec> &vheads,
              char levels, double tiny, int min_level) {
    std::map<int, Chan> q, k;
    for (const auto &p : queries) q[p.first] = sel(*this, p.second);
    for (const auto &p : keys) k[p.first] = sel(*this, p.second);
    std::vector<VSpec> rest(vheads.begin() + 1, vheads.end());
    L outs = head(name, std::move(q), std::move(k), vheads[0], rest, min_level, tiny);
    nodes.back().levels = levels;
    tag_table(name);
    return outs;
}

L Circ::int_bits(const Expr &z, long long hi, const std::string &name, int n) {
    L out;
    for (int j = 0; j < n; j++) {
        Terms terms;
        for (long long theta = 1; theta <= hi; theta++) {
            long long d = (theta >> j & 1) - ((theta - 1) >> j & 1);
            if (d) terms += t_ge(z, theta, 0, hi, Dy(d));
        }
        out.push_back(node(terms, name + ".b" + std::to_string(j)));
    }
    return out;
}

Expr Circ::counter(const std::string &name, const Expr &reset, const Expr &step) {
    TSpec s;
    s.data[2] = Chan(reset);
    s.ref = one - reset;
    s.a = Dy(110) * step + Dy(238) * reset;
    auto outs = thead(*this, name, {{0, Chan(reset)}, {1, Chan(one - reset)}}, {{0, Chan(one - reset)}, {1, Chan(one - reset)}}, {s}, std::ldexp(1.0, -90));
    tag_table(name);
    nodes.back().levels = 'd';
    return outs[0][0];
}

L Circ::thermo(const Expr &x, const Expr &reset, int n, const std::string &name) {
    L r;
    for (int m = 0; m < n; m++) {
        Expr scaled = x * Dy::pow2(m + 3);
        r.push_back(node(Terms{Term{Dy(1), one - reset, Dy(7) * one - scaled}, Term{Dy(-1), one - reset, Dy(6) * one - scaled}}, name + ">" + std::to_string(m)));
    }
    return r;
}

Expr Circ::split(const Expr &z, long long lo, long long hi, const std::string &name) {
    long long q = fdiv_py(lo, 16);
    Terms terms = q ? t_const(Dy(q)) : Terms{};
    for (long long t = q + 1; t <= fdiv_py(hi, 16); t++) terms += t_ge(z, 16 * t, lo, hi);
    return node(terms, name);
}

// A register: a one-row table written only when `we` or `clear` is 1 (they must be exclusive), with the bits (masked by
// clear) as its row; otherwise the row, and so the output, keeps its value. A head writes before it reads within a pass,
// so a register never reads its own value: its data must come from the tokens or from state below it.
L Circ::reg(const std::string &name, const L &bits, const Expr &we, const Expr &clear, int n) {
    L outs = zeros(n, name);
    TSpec s;
    for (int i = 0; i < n; i++) {
        const Expr &b = i < (int)bits.size() ? bits[(size_t)i] : Expr();
        if (b.empty()) continue;
        s.data[2 + i] = Chan(clear.empty() ? b : and_(L{one - clear, b}, name + ".d" + std::to_string(i)));
    }
    s.ref = one;
    s.beta = clear.empty() ? we : we + clear;
    s.into = outs;
    thead(*this, name, {{0, Chan(one)}}, {{0, Chan(one)}}, {s});
    tag_table(name);
    return outs;
}

// The previous pass's value of each bit: a one-row table written every pass from a lag-1 channel of the bit. The bit's
// form must be computed below this table (a channel samples its form at the table's level).
L Circ::lag1(const L &bits, const std::string &name) {
    L outs = zeros((int)bits.size(), name);
    TSpec s;
    for (size_t i = 0; i < bits.size(); i++) s.data[2 + (int)i] = Chan(bits[i], {0, 1, 0, 0});
    s.ref = one;
    s.beta = one;
    s.into = outs;
    thead(*this, name, {{0, Chan(one)}}, {{0, Chan(one)}}, {s});
    tag_table(name);
    return outs;
}

// --- nibble columns ----------------------------------------------------------------------------------

L bits_of(const L &th, int n_bits, int top) {
    L out;
    for (int i = 0; i < n_bits; i++) {
        Expr e;
        for (int v = 1; v <= top; v++) {
            int d = (v >> i & 1) - ((v - 1) >> i & 1);
            if (d) e = e + Dy(d) * th[(size_t)v - 1];
        }
        out.push_back(e);
    }
    return out;
}

Expr count_fn(Circ &c, const Counter &x, const IntF &f, int top, const std::string &name, const Expr *a_in, const L &lits_) {
    const Expr &one = c.one;
    Expr a = a_in ? *a_in : one;
    Expr rs = Dy(16) * x.reset + Dy(16) * (Dy((long long)lits_.size()) - sum(lits_));
    Terms terms;
    if (f(0)) {
        if (!lits_.empty()) terms = c.t_and(lits_);
        else terms.push_back(Term{Dy(1), a, one});
    }
    for (int v = 1; v <= top; v++) {
        int d = f(v) - f(v - 1);
        if (d) {
            int m = v - 1;
            Expr sx = x.x * Dy::pow2(m + 3);
            terms.push_back(Term{Dy(d), a, Dy(7) * one - sx - rs});
            terms.push_back(Term{Dy(-d), a, Dy(6) * one - sx - rs});
        }
    }
    return c.node(terms, name);
}

Expr count_bit(Circ &c, const Counter &x, int b, int top, const std::string &name, const Expr *base, int shift) {
    IntF f = [b, shift](int k) { return k >= shift ? (k - shift) >> b & 1 : 0; };
    if (!base) return count_fn(c, x, f, top, name);
    Expr na = c.one - *base;
    Expr node = count_fn(c, x, f, top, name, &na);
    return node + *base;
}

L sum_columns(Circ &c, const Cols &cols_in, int n_nib, const std::string &name) {
    Cols cols = cols_in;
    cols.resize((size_t)n_nib);
    auto bounds_of = [&](const std::vector<CT> &cl) {
        long long p = 0, n = 0;
        for (const CT &t : cl) {
            p += std::max(t.hi, 0LL);
            n += std::max(-t.lo, 0LL);
        }
        return std::pair<long long, long long>(p, n);
    };
    int rnd = 0;
    for (;;) {
        bool fits = true;
        for (const auto &cl : cols) {
            auto b = bounds_of(cl);
            fits = fits && b.first + b.second + 15 <= 255;
        }
        if (fits) break;
        Cols nw((size_t)n_nib);
        for (int k = 0; k < n_nib; k++) {
            const auto &cl = cols[(size_t)k];
            auto bd = bounds_of(cl);
            if (bd.first + bd.second + 15 <= 255) {
                nw[(size_t)k] += cl;
                continue;
            }
            struct Group {
                std::vector<CT> ts;
                long long p, n;
            };
            std::vector<Group> groups;
            Group cur{{}, 0, 0};
            for (const CT &t : cl) {
                long long tp = std::max(t.hi, 0LL), tn = std::max(-t.lo, 0LL);
                if (!cur.ts.empty() && cur.p + tp + cur.n + tn > 240) {
                    groups.push_back(cur);
                    cur = Group{{}, 0, 0};
                }
                cur.ts.push_back(t);
                cur.p += tp;
                cur.n += tn;
            }
            groups.push_back(cur);
            for (size_t gi = 0; gi < groups.size(); gi++) {
                const Group &g = groups[gi];
                long long off = -fdiv_py(-g.n, 16) * 16;
                Expr e;
                {
                    L v;
                    for (const CT &t : g.ts) v.push_back(t.e);
                    e = sum(v) + Dy(off);
                }
                long long hi = g.p + off;
                Expr car = c.split(e, 0, hi, name + ".r" + std::to_string(rnd) + ".c" + std::to_string(k) + "." + std::to_string(gi));
                nw[(size_t)k].push_back(CT{e - Dy(16) * car, 0, 15});
                if (k + 1 < n_nib) nw[(size_t)k + 1].push_back(CT{car - Dy(fdiv_py(off, 16)), -fdiv_py(off, 16), fdiv_py(hi, 16) - fdiv_py(off, 16)});
            }
        }
        cols = nw;
        rnd++;
    }
    std::vector<std::pair<int, int>> bounds;
    for (const auto &cl : cols) {
        auto b = bounds_of(cl);
        bounds.push_back({(int)b.first, (int)b.second});
    }
    std::vector<int> offs = col_offsets(bounds);
    L car, dig;
    std::vector<long long> cmax;
    for (int k = 0; k < n_nib; k++) {
        L v;
        for (const CT &t : cols[(size_t)k]) v.push_back(t.e);
        Expr e = sum(v) + Dy(offs[(size_t)k]);
        long long lo = offs[(size_t)k] - bounds[(size_t)k].second, hi = offs[(size_t)k] + bounds[(size_t)k].first;
        Expr cc = c.split(e, lo, hi, name + ".f" + std::to_string(k));
        car.push_back(cc);
        dig.push_back(e - Dy(16) * cc);
        cmax.push_back(fdiv_py(hi, 16));
    }
    std::vector<std::pair<Expr, long long>> fin;
    for (int k = 0; k < n_nib; k++) fin.push_back({dig[(size_t)k] + (k ? car[(size_t)k - 1] : Expr()), 15 + (k ? cmax[(size_t)k - 1] : 0)});
    Stage1 s = stage1_cols(c, fin, {}, name);
    return adder_stage2(c, s.g, s.p, s.d, n_nib, name).r;
}

void product_cols(Circ &c, Cols &cols, const L &x, int nx, bool sx, const L &y, const std::vector<std::pair<long long, long long>> &yr, const std::string &name) {
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < (int)y.size() && i + j < (int)cols.size(); j++) {
            const bool sg = sx && i == nx - 1;
            Terms t;
            for (int b = 0; b < 4; b++) t += c.t_and(L{x[(size_t)(4 * i + b)]}, Dy((sg && b == 3) ? -8 : 1 << b), &y[(size_t)j]);
            Expr pt = c.node(t, name + ".m" + std::to_string(i) + "." + std::to_string(j));
            const long long xlo = sg ? -8 : 0, xhi = sg ? 7 : 15;
            const long long v[4] = {xlo * yr[(size_t)j].first, xlo * yr[(size_t)j].second, xhi * yr[(size_t)j].first, xhi * yr[(size_t)j].second};
            cols[(size_t)(i + j)].push_back(CT{pt, *std::min_element(v, v + 4), *std::max_element(v, v + 4)});
        }
}

L mul_round(Circ &c, const L &x, int nx, const L &y, const std::string &name) {
    Cols cols(12);
    product_cols(c, cols, x, nx, true, y, std::vector<std::pair<long long, long long>>(8, {0, 15}), name);
    cols[3].push_back(CT{Dy(8) * c.one, 8, 8});
    L r = sum_columns(c, cols, 12, name);
    return sl(r, 16, 48);
}

std::pair<L, std::vector<std::pair<long long, long long>>> signed_nibs(const L &bits, int n) {
    L out;
    std::vector<std::pair<long long, long long>> r;
    for (int k = 0; k < n; k++) {
        if (k + 1 < n) {
            out.push_back(nib(bits, k));
            r.push_back({0, 15});
        } else {
            const size_t top = (size_t)(4 * k + 3);
            out.push_back(from_bits(sl(bits, 4 * k, 4 * k + 3)) - Dy(8) * bits[top]);
            r.push_back({-8, 7});
        }
    }
    return {out, r};
}

L barrel(Circ &c, const L &x48, const L &u, int n, const std::string &name) {
    const auto xw = [&](int i) -> Expr {
        if (i < 33) return Expr();
        return x48[(size_t)std::min(i - 33, 47)];
    };
    std::map<int, Expr> ta;
    for (int i = 0; i < n + 8; i++) {
        Terms t;
        for (int q = 0; q < 8; q++) {
            const Expr bit = xw(i + 8 * q);
            if (!bit.empty()) t += c.t_and(lits_of(sl(u, 3, 6), q) + L{bit});
        }
        ta[i] = c.node(t, name + ".a" + std::to_string(i));
    }
    L out;
    for (int i = 0; i < n; i++) {
        Terms t;
        for (int r = 0; r < 8; r++) t += c.t_and(lits_of(sl(u, 0, 3), r) + L{ta[i + r]});
        out.push_back(c.node(t, name + ".b" + std::to_string(i)));
    }
    return out;
}

void plane_cols(Circ &c, Cols &cols, long long &cst, const std::vector<L> &qs, const L &ws, long long wmin, long long wmax, int nibk, const std::string &name) {
    const int gsz = 16;
    const long long lo = wmin * gsz, hi = wmax * gsz;
    for (int b = 0; b < 8; b++) {
        const long long sg = b == 7 ? -1 : 1;
        for (int g = 0; g < (int)qs.size(); g += gsz) {
            Terms t;
            for (int i = g; i < g + gsz; i++) t += c.t_and(L{qs[(size_t)i][(size_t)b]}, 1, &ws[(size_t)i]);
            const std::string bs = std::to_string(b) + "." + std::to_string(g);
            Expr n_ = c.node(t, name + ".n" + bs);
            Expr car = c.split(n_ - Dy(lo), 0, hi - lo, name + ".c" + bs);
            Expr dig = n_ - Dy(lo) - Dy(16) * car;
            const int w = b + 4 * nibk;
            const struct {
                Expr e;
                long long top;
                int w;
            } parts[2] = {{dig, 15, w}, {car, fdiv_py(hi - lo, 16), w + 4}};
            for (const auto &p : parts) {
                const size_t k = (size_t)fdiv_py(p.w, 4);
                if (k >= cols.size()) continue;
                const long long f = sg * (1LL << fmod_py(p.w, 4));
                cols[k].push_back(CT{Dy(f) * p.e, std::min(0LL, f * p.top), std::max(0LL, f * p.top)});
            }
            cst += sg * lo * (1LL << w);
        }
    }
}

std::pair<Expr, Expr> cmp_bits(Circ &c, const L &a, const L &b, int n_nib, const std::string &name) {
    L gt, eq;
    for (int k = 0; k < n_nib; k++) gt.push_back(c.node(c.t_ge(nib(b, k) - nib(a, k), 1, -15, 15), name + ".lt" + std::to_string(k)));
    for (int k = 0; k < n_nib; k++) eq.push_back(c.node(c.t_eq(nib(a, k) - nib(b, k), 0, -15, 15), name + ".eq" + std::to_string(k)));
    Terms lt_t;
    for (int k = 0; k < n_nib; k++) lt_t += c.t_and(L{gt[(size_t)k]} + sl(eq, k + 1, n_nib));
    Expr lt = c.node(lt_t, name + ".lt");
    Expr eqa = c.and_(eq, name + ".eq");
    return {lt, eqa};
}
