// See graph.h.
#include "graph.h"

#include <algorithm>
#include <set>

Graph::Graph(Target t) : target(t) {
    dims.push_back(Dim{"one", 'o', 0});
    one = Expr::dim(ONE, Dy(1));
}

int Graph::new_dim(const std::string &name, char kind, int level) {
    dims.push_back(Dim{name, kind, level});
    return (int)dims.size() - 1;
}

Expr Graph::token_feature(const std::string &name, std::function<double(uint32_t)> fn) {
    int d = new_dim(name, 't', 0);
    feat_dim.push_back(d);
    feat_fn.push_back(std::move(fn));
    return Expr::dim(d);
}

int Graph::level_of(const Expr &e) const {
    int m = 0;
    for (const auto &p : e.t) m = std::max(m, dims[(size_t)p.first].level);
    return m;
}

int Graph::place(int lvl, bool odd, int need, std::map<int, int> &used, int cap) {
    int sub = lvl + 1;
    for (;;) {
        auto it = used.find(sub);
        int u = it == used.end() ? 0 : it->second;
        if ((sub % 2 == 1) == odd && u + need <= cap) break;
        sub++;
    }
    used[sub] += need;
    return sub;
}

Expr Graph::ffn(const Terms &terms_in, const std::string &name, const Expr *into, int min_level) {
    min_level = std::max(min_level, floor_);
    Terms terms;
    terms.reserve(terms_in.size());
    for (const Term &t : terms_in)
        if (!t.c.zero()) terms.push_back(t);
    int lvl = min_level;
    for (const Term &t : terms) lvl = std::max({lvl, level_of(t.a), level_of(t.b)});
    int sub = place(lvl, false, (int)terms.size(), units, target.intermediate);
    int d;
    if (into) {
        d = into->only_dim();
        dims[(size_t)d].level = std::max(dims[(size_t)d].level, sub);
    } else {
        d = new_dim(name, 'f', sub);
    }
    Node n;
    n.is_ffn = true;
    n.name = name;
    n.terms = std::move(terms);
    n.out = d;
    n.level = sub;
    n.into = into != nullptr;
    n.min_level = min_level;
    nodes.push_back(std::move(n));
    return Expr::dim(d);
}

L Graph::head(const std::string &name, std::map<int, Chan> q, std::map<int, Chan> k, const VSpec &first,
              const std::vector<VSpec> &rest, int min_level, double tiny) {
    if (!rest.empty() && (int)rest.size() >= target.value_heads / target.key_heads)
        Dy::die("more value heads than the key head has");
    int lvl = min_level;
    auto bump = [&](const Expr &e) { lvl = std::max(lvl, level_of(e)); };
    for (const auto &p : q) bump(p.second.expr);
    for (const auto &p : k) bump(p.second.expr);
    auto scan = [&](const VSpec &x) {
        for (const auto &p : x.v) bump(p.second.expr);
        bump(x.a);
        for (const auto &p : x.z) bump(p.second);
        if (x.beta) bump(*x.beta);
    };
    scan(first);
    for (const VSpec &x : rest) scan(x);
    int sub = place(lvl, true, 1, heads, target.key_heads);

    L ret;
    auto make = [&](const VSpec &x, const std::string &tag) {
        VHead vh;
        vh.v = x.v;
        vh.a = x.a;
        vh.z = x.z;
        vh.beta = x.beta;
        vh.into = x.into.has_value();
        for (size_t i = 0; i < x.outs.size(); i++) {
            int d;
            if (x.into) {
                d = (*x.into)[i].only_dim();
                dims[(size_t)d].level = std::max(dims[(size_t)d].level, sub);
            } else {
                d = new_dim(name + tag + "[" + std::to_string(x.outs[i].first) + "]", 'h', sub);
            }
            vh.out.push_back(Out{x.outs[i].first, d, x.outs[i].second});
            ret.push_back(Expr::dim(d));
        }
        return vh;
    };
    Node n;
    n.is_ffn = false;
    n.name = name;
    n.level = sub;
    n.min_level = min_level;
    n.tiny = tiny;
    n.q = std::move(q);
    n.k = std::move(k);
    n.vh.push_back(make(first, ""));
    for (size_t i = 0; i < rest.size(); i++) n.vh.push_back(make(rest[i], ".v" + std::to_string(i + 1)));
    nodes.push_back(std::move(n));
    return ret;
}

void Graph::bank(const std::string &name, const Bank &spec, int min_level) {
    int lvl = min_level;
    auto bump = [&](const Expr &e) { lvl = std::max(lvl, level_of(e)); };
    for (const BankRead &r : spec.reads)
        for (const Expr &e : r.addr) bump(e);
    for (const BankWrite &w : spec.writes) {
        for (const Expr &e : w.addr) bump(e);
        for (const Expr &e : w.data) bump(e);
        bump(w.we);
    }
    int sub = place(lvl, true, 0, heads, target.key_heads);
    for (const BankRead &r : spec.reads)
        for (int d : r.out) dims[(size_t)d].level = std::max(dims[(size_t)d].level, sub);
    Node n;
    n.is_ffn = false;
    n.is_bank = true;
    n.name = name;
    n.level = sub;
    n.min_level = min_level;
    n.bank = spec;
    nodes.push_back(std::move(n));
}

int Graph::n_layers() const {
    int top = 0;
    for (const Node &n : nodes) top = std::max(top, n.level);
    return (top + 1) / 2;
}

// --- prims.Circ ------------------------------------------------------------------------------

Terms Graph::t_const(const Dy &c) const { return Terms{Term{c, one, one}}; }

Terms Graph::t_and(const L &lits, const Dy &coef, const Expr *a) const {
    // The gate is 4 * (sum - n + 1/2): +2 when all are set, <= -2 otherwise.
    Expr g = Dy(4) * sum(lits) - Dy((long long)(4 * lits.size()) - 2);
    return Terms{Term{coef.half(), a ? *a : one, g}};
}

Terms Graph::t_ge(const Expr &z, long long theta, long long lo, long long hi, const Dy &coef,
                  const L &lits) const {
    if (theta > hi) return {};
    long long n = (long long)lits.size();
    if (theta <= lo && !n) return t_const(coef);
    if (hi - theta + 1 <= 256) {
        long long m = 1LL << bit_length(hi - lo);
        Expr zz = n ? z + Dy(m) * (sum(lits) - Dy(n)) : z;
        Expr g0 = zz - Dy(theta);
        return Terms{Term{coef, one, g0 + Dy(1)}, Term{-coef, one, g0}};
    }
    if (theta - lo > 256) Dy::die("t_ge: threshold out of the exact range");
    Terms base = n ? t_and(lits, coef) : t_const(coef);
    return base + t_lt(z, theta, lo, hi, -coef, lits);
}

Terms Graph::t_lt(const Expr &z, long long theta, long long lo, long long hi, const Dy &coef,
                  const L &lits) const {
    if (theta <= lo) return {};
    long long n = (long long)lits.size();
    if (theta > hi && !n) return t_const(coef);
    if (theta - lo <= 256) {
        long long m = 1LL << bit_length(hi - lo);
        Expr zz = n ? z - Dy(m) * (sum(lits) - Dy(n)) : z;
        Expr g0 = Dy(theta) - zz;
        return Terms{Term{coef, one, g0}, Term{-coef, one, g0 - Dy(1)}};
    }
    if (hi - theta + 1 > 256) Dy::die("t_lt: threshold out of the exact range");
    Terms base = n ? t_and(lits, coef) : t_const(coef);
    return base + t_ge(z, theta, lo, hi, -coef, lits);
}

Terms Graph::t_in(const Expr &z, long long a, long long b, long long lo, long long hi, const Dy &coef,
                  const L &lits) const {
    a = std::max(a, lo);
    b = std::min(b, hi + 1);
    if (a >= b) return {};
    if (a == lo) return t_lt(z, b, lo, hi, coef, lits);
    if (b == hi + 1) return t_ge(z, a, lo, hi, coef, lits);
    if (!lits.empty()) return t_ge(z, a, lo, hi, coef, lits) + t_ge(z, b, lo, hi, -coef, lits);
    if (std::max(b - lo, a - lo) <= 256) return t_lt(z, b, lo, hi, coef) + t_lt(z, a, lo, hi, -coef);
    return t_ge(z, a, lo, hi, coef) + t_ge(z, b, lo, hi, -coef);
}

Terms Graph::t_eq(const Expr &z, long long v, long long lo, long long hi, const Dy &coef,
                  const L &lits) const {
    return t_in(z, v, v + 1, lo, hi, coef, lits);
}

Expr Graph::node(const Terms &terms, const std::string &name) {
    if (terms.empty()) return Expr();
    return ffn(terms, name);
}

Expr Graph::and_(const L &lits, const std::string &name) { return node(t_and(lits), name); }

Expr from_bits(const L &bits) {
    Expr e;
    for (size_t i = 0; i < bits.size(); i++) e += bits[i] * Dy(1LL << i);
    return e;
}

Chan sel(const Graph &c, const Chan &e) {
    return Chan(e.expr * Dy(2 * K_GATE) - c.one * Dy(K_GATE), e.taps);
}

Chan chan(const Chan &e, const Dy &scale) { return Chan(e.expr * scale, e.taps); }

std::vector<L> thead(Graph &c, const std::string &name, const std::map<int, Chan> &keys,
                     const std::map<int, Chan> &queries, const std::vector<TSpec> &vheads, double tiny) {
    std::vector<VSpec> specs;
    for (const TSpec &x : vheads) {
        VSpec s;
        for (const auto &p : x.data) s.v[p.first] = chan(p.second, Dy(A_BIT));
        s.v[REF_COL0] = chan(Chan(x.ref), Dy((long long)A_REF));
        s.v[REF_COL1] = chan(Chan(x.ref), Dy((long long)A_REF));
        Expr gate = x.gate ? *x.gate : c.one;
        for (const auto &p : x.data) {  // sorted(x["data"]): std::map iterates in column order
            s.z[p.first] = Dy(K_GATE) * gate;
            s.outs.emplace_back(p.first, OUT_W);
        }
        s.a = x.a;
        s.beta = x.beta;
        s.into = x.into;
        specs.push_back(std::move(s));
    }
    std::map<int, Chan> q, k;
    for (const auto &p : queries) q[p.first] = sel(c, p.second);
    for (const auto &p : keys) k[p.first] = sel(c, p.second);
    std::vector<VSpec> rest(specs.begin() + 1, specs.end());
    L outs = c.head(name, std::move(q), std::move(k), specs[0], rest, 0, tiny);
    std::vector<L> res;
    size_t i = 0;
    for (const VSpec &s : specs) {
        size_t n = s.outs.size();
        res.push_back(L(outs.begin() + (long)i, outs.begin() + (long)(i + n)));
        i += n;
    }
    return res;
}

// --- sched.py --------------------------------------------------------------------------------

std::vector<int> node_outputs(const Node &n) {
    std::vector<int> o;
    if (n.is_bank) {
        for (const BankRead &r : n.bank.reads)
            for (int d : r.out) o.push_back(d);
        return o;
    }
    if (n.is_ffn) {
        o.push_back(n.out);
    } else {
        for (const VHead &vh : n.vh)
            for (const Out &x : vh.out) o.push_back(x.dim);
    }
    return o;
}

std::vector<int> node_reads(const Node &n) {
    std::vector<int> r;
    auto add = [&](const Expr &e) {
        for (const auto &p : e.t) r.push_back(p.first);
    };
    if (n.is_bank) {
        for (const BankRead &rd : n.bank.reads)
            for (const Expr &e : rd.addr) add(e);
        for (const BankWrite &w : n.bank.writes) {
            for (const Expr &e : w.addr) add(e);
            for (const Expr &e : w.data) add(e);
            add(w.we);
        }
        return r;
    }
    if (n.is_ffn) {
        for (const Term &t : n.terms) {
            add(t.a);
            add(t.b);
        }
        return r;
    }
    for (const auto &p : n.q) add(p.second.expr);
    for (const auto &p : n.k) add(p.second.expr);
    for (const VHead &vh : n.vh) {
        for (const auto &p : vh.v) add(p.second.expr);
        add(vh.a);
        for (const auto &p : vh.z) add(p.second);
        if (vh.beta) add(*vh.beta);
    }
    return r;
}

static std::vector<int> uniq(std::vector<int> v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}

void reschedule(Graph &g, int unit_cap, int head_cap, const char *last) {
    std::vector<Node> &nodes = g.nodes;
    int N = (int)nodes.size();
    size_t D = g.dims.size();
    std::vector<std::vector<int>> writers(D), readers(D);
    std::vector<std::set<std::pair<int, int>>> deps((size_t)N);  // (pred, gap): node >= pred + gap
    for (int i = 0; i < N; i++) {
        for (int d : uniq(node_reads(nodes[(size_t)i]))) {
            for (int j : writers[(size_t)d]) deps[(size_t)i].insert({j, 1});
            readers[(size_t)d].push_back(i);
        }
        for (int d : node_outputs(nodes[(size_t)i])) {
            for (int r : readers[(size_t)d])
                if (r != i) deps[(size_t)i].insert({r, 0});
            writers[(size_t)d].push_back(i);
        }
    }
    std::vector<std::vector<std::pair<int, int>>> succ((size_t)N);
    for (int i = 0; i < N; i++)
        for (const auto &p : deps[(size_t)i]) succ[(size_t)p.first].push_back({i, p.second});
    std::vector<int> height((size_t)N, 0);
    for (int i = N - 1; i >= 0; i--) {
        int h = 0;
        for (const auto &p : succ[(size_t)i]) {
            int step = p.second == 0 ? 0 : (nodes[(size_t)i].is_ffn == nodes[(size_t)p.first].is_ffn ? 2 : 1);
            h = std::max(h, height[(size_t)p.first] + step);
        }
        height[(size_t)i] = h;
    }
    std::vector<int> level((size_t)N, -1), n_left((size_t)N), earliest((size_t)N);
    std::vector<char> is_last((size_t)N);
    std::string pre(last);
    int n_body = 0;
    for (int i = 0; i < N; i++) {
        n_left[(size_t)i] = (int)deps[(size_t)i].size();
        is_last[(size_t)i] = nodes[(size_t)i].name.compare(0, pre.size(), pre) == 0;
        earliest[(size_t)i] = is_last[(size_t)i] ? 1 : nodes[(size_t)i].min_level + 1;
        n_body += !is_last[(size_t)i];
    }
    std::vector<int> ready;
    for (int i = 0; i < N; i++)
        if (!n_left[(size_t)i]) ready.push_back(i);
    int placed = 0, placed_body = 0, sub = 1, top_body = 0;
    std::map<int, int> units, heads;
    while (placed < N) {
        bool want_ffn = sub % 2 == 0, body_done = placed_body == n_body;
        int floor_lvl = body_done ? top_body + 1 : 0;
        std::vector<int> cand;
        for (int i : ready)
            if (nodes[(size_t)i].is_ffn == want_ffn && earliest[(size_t)i] <= sub
                && (!is_last[(size_t)i] || (body_done && sub >= floor_lvl)))
                cand.push_back(i);
        std::sort(cand.begin(), cand.end(), [&](int a, int b) {
            if (height[(size_t)a] != height[(size_t)b]) return height[(size_t)a] > height[(size_t)b];
            return a < b;
        });
        std::vector<int> done;
        for (int i : cand) {
            if (want_ffn) {
                int need = (int)nodes[(size_t)i].terms.size();
                if (units[sub] + need > unit_cap) continue;
                units[sub] += need;
            } else if (!nodes[(size_t)i].is_bank) {
                if (heads[sub] + 1 > head_cap) continue;
                heads[sub] += 1;
            }
            level[(size_t)i] = sub;
            done.push_back(i);
        }
        if (!done.empty()) {
            std::set<int> ds(done.begin(), done.end());
            ready.erase(std::remove_if(ready.begin(), ready.end(), [&](int i) { return ds.count(i) != 0; }),
                        ready.end());
            placed += (int)done.size();
            for (int i : done) {
                if (!is_last[(size_t)i]) {
                    placed_body++;
                    top_body = std::max(top_body, level[(size_t)i]);
                }
                for (const auto &p : succ[(size_t)i]) {
                    int j = p.first;
                    earliest[(size_t)j] = std::max(earliest[(size_t)j], sub + p.second);
                    if (--n_left[(size_t)j] == 0) ready.push_back(j);
                }
            }
            continue;
        }
        sub++;
        if (sub >= 100000) Dy::die("reschedule: no progress");
    }
    for (int i = 0; i < N; i++) nodes[(size_t)i].level = level[(size_t)i];
    for (size_t d = 0; d < D; d++) {
        if (writers[d].empty()) continue;
        int m = 0;
        for (int i : writers[d]) m = std::max(m, level[(size_t)i]);
        g.dims[d].level = m;
    }
    g.units = units;
    g.heads = heads;
}

void sched_check(const Graph &g) {
    size_t D = g.dims.size();
    std::vector<std::vector<int>> writers(D);
    for (size_t i = 0; i < g.nodes.size(); i++) {
        const Node &n = g.nodes[i];
        if ((n.level % 2 == 0) != n.is_ffn) Dy::die("sched_check: node on the wrong parity");
        for (int d : node_outputs(n)) writers[(size_t)d].push_back((int)i);
    }
    for (size_t i = 0; i < g.nodes.size(); i++) {
        const Node &n = g.nodes[i];
        for (int d : uniq(node_reads(n)))
            for (int j : writers[(size_t)d]) {
                int lj = g.nodes[(size_t)j].level;
                if ((j < (int)i && lj >= n.level) || (j > (int)i && lj < n.level))
                    Dy::die("sched_check: a read is not respected");
            }
    }
    std::map<int, int> units, heads;
    for (const Node &n : g.nodes) {
        if (n.is_ffn) units[n.level] += (int)n.terms.size();
        else if (!n.is_bank) heads[n.level] += 1;
    }
    for (const auto &p : units)
        if (p.second > g.target.intermediate) Dy::die("sched_check: too many units in a sublayer");
    for (const auto &p : heads)
        if (p.second > g.target.key_heads) Dy::die("sched_check: too many heads in a sublayer");
}

// --- emit.py allocate() --------------------------------------------------------------------------

Alloc allocate(const Graph &g, const std::vector<int> &keep_list, int unit_cap) {
    const size_t D = g.dims.size();
    Alloc a;
    a.slot.assign(D, -1);
    std::vector<char> keep(D, 0);
    for (int d : keep_list) keep[(size_t)d] = 1;
    int top = 0;
    for (const Node &n : g.nodes) top = std::max(top, n.level);
    top += 1;
    std::vector<int> last(D, 0);
    for (size_t d = 1; d < D; d++) last[d] = g.dims[d].level;
    for (const Node &n : g.nodes)
        for (int d : node_reads(n))
            if (d != ONE) last[(size_t)d] = std::max(last[(size_t)d], n.level);
    for (int d : keep_list) last[(size_t)d] = top;
    std::map<int, int> units = g.units;
    std::vector<int> &owner = a.final_owner;

    // a slot is handed over by subtracting its old value: an FFN node defining the new dim does it with one extra unit at
    // its own level; a head's dim is cleared by a unit at the FFN level before the head's
    auto take = [&](int d, int level, bool by_ffn) {
        for (size_t s = 0; s < owner.size(); s++) {
            const int o = owner[s];
            if (keep[(size_t)o]) continue;
            const int born = g.dims[(size_t)o].level;  // the old value must already be in the sublayer's input
            if (by_ffn && last[(size_t)o] <= level && born < level && units[level] < unit_cap) {
                units[level] += 1;
                a.subtract[d] = o;
            } else if (!by_ffn && last[(size_t)o] <= level - 1 && born < level - 1 && level >= 3 && units[level - 1] < unit_cap) {
                units[level - 1] += 1;
                a.clears[level - 1].push_back(o);
            } else {
                continue;
            }
            owner[s] = d;
            a.slot[(size_t)d] = (int)s;
            return;
        }
        a.slot[(size_t)d] = (int)owner.size();
        owner.push_back(d);
    };

    std::vector<int> feats = g.feat_dim;
    std::sort(feats.begin(), feats.end());
    for (int d : feats) take(d, 0, false);
    std::map<int, std::vector<std::pair<int, bool>>> defined;
    for (const Node &n : g.nodes) {
        if (n.is_bank) continue;
        if (n.is_ffn) {
            if (!n.into) defined[n.level].push_back({n.out, true});
        } else {
            for (const VHead &vh : n.vh)
                if (!vh.into)
                    for (const Out &x : vh.out) defined[n.level].push_back({x.dim, false});
        }
    }
    for (const auto &lv : defined)
        for (const auto &p : lv.second) take(p.first, lv.first, p.second);
    a.n_slots = (int)owner.size();
    return a;
}

int allocate_slots(const Graph &g, const std::vector<int> &keep_list, int unit_cap) { return allocate(g, keep_list, unit_cap).n_slots; }
