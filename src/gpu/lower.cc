// See lower.h. The structure follows lower.py's _lower: symbolic values per PTX register, machine instructions over
// virtual registers grouped by control flow, liveness over the group CFG, register allocation (two files: lane and
// uniform), then block formation and terminators.
#include "lower.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <sstream>

namespace lower {

namespace {

constexpr int EXIT_P = 7, BRANCH_P = 6, SEL_P = 3;
constexpr uint32_t M32 = 0xFFFFFFFFu;

struct Val {
    enum K { NONE, CONST, PTR, SREG, REG, SCALED, ADDR, PRED, KNOWN, WIDE, GATHER } k = NONE;
    int64_t v = 0;  // CONST: the 32-bit value or a byte offset; PTR / ADDR: the word; SCALED: k (words per unit)
    int r = 0;      // REG, SCALED, ADDR
    bool ptr = false;
    std::string name;  // SREG
    int cmp = 0;       // PRED
    bool sgn = false;
    int ra = 0;
    bool rhs_const = false;
    int64_t rhs_v = 0;
    int rhs_r = 0;
    bool known = false;  // KNOWN
    bool puni = false;   // PRED: the same on every lane by the uniformity analysis (a field test of the thread index), whatever its operands' kinds
    std::shared_ptr<Val> inner;  // WIDE
    std::vector<int> bits;       // GATHER: t-bit indices, low first
    bool linear = false;         // GATHER: t itself
    static Val Const(int64_t x) { Val o; o.k = CONST; o.v = x & M32; return o; }
    static Val RawConst(int64_t x) { Val o; o.k = CONST; o.v = x; return o; }  // a byte count that may exceed 32 bits
    static Val Ptr(int64_t w) { Val o; o.k = PTR; o.v = w; return o; }
    static Val Sreg(const std::string &n) { Val o; o.k = SREG; o.name = n; return o; }
    static Val Reg(int r, bool ptr = false) { Val o; o.k = REG; o.r = r; o.ptr = ptr; return o; }
    static Val Scaled(int r, int64_t k) { Val o; o.k = SCALED; o.r = r; o.v = k; return o; }
    static Val Addr(int r, int64_t w) { Val o; o.k = ADDR; o.r = r; o.v = w; return o; }                 // a 32-bit register + constant
    static Val PAddr(int r, int64_t w) { Val o; o.k = ADDR; o.r = r; o.v = w; o.ptr = true; return o; }  // a word index register + word
    static Val Known(bool b) { Val o; o.k = KNOWN; o.known = b; return o; }
    static Val Wide(const Val &x) { Val o; o.k = WIDE; o.inner = std::make_shared<Val>(x); return o; }
    static Val Gather(std::vector<int> b) { Val o; o.k = GATHER; o.bits = std::move(b); return o; }
    static Val Linear() { Val o; o.k = GATHER; o.linear = true; return o; }
    bool is(K kk) const { return k == kk; }
    bool operator==(const Val &o) const {
        return k == o.k && v == o.v && r == o.r && ptr == o.ptr && name == o.name && cmp == o.cmp && sgn == o.sgn && ra == o.ra
               && rhs_const == o.rhs_const && rhs_v == o.rhs_v && rhs_r == o.rhs_r && known == o.known && puni == o.puni && bits == o.bits && linear == o.linear
               && ((!inner && !o.inner) || (inner && o.inner && *inner == *o.inner));
    }
};

const char *CMPS[6] = {"eq", "ne", "lt", "le", "gt", "ge"};
int cmp_index(const std::string &c) {
    for (int i = 0; i < 6; i++)
        if (c == CMPS[i]) return i;
    throw Unsupported("compare " + c);
}
int mirror(int c) {
    static const int m[6] = {SM_CMP_EQ, SM_CMP_NE, SM_CMP_GT, SM_CMP_GE, SM_CMP_LT, SM_CMP_LE};
    return m[c];
}
bool compare_const(int cmp, uint32_t a, uint32_t b, bool sgn) {
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
bool pow2(int64_t v) { return v > 0 && (v & (v - 1)) == 0; }
int bitlen(int64_t v) { return v ? 64 - __builtin_clzll((uint64_t)v) : 0; }

bool is_sreg_uniform(const std::string &n) {
    return n == "%ctaid.x" || n == "%ntid.x" || n == "%nctaid.x" || n == "%ctaid.y" || n == "%ntid.y" || n == "%nctaid.y";
}

// the register a PTX instruction defines, or ""
std::string defines(const ptx::Instr &ins) {
    if (!ins.args.empty() && ins.args[0].kind == ptx::Operand::REG) {
        const std::string &b = ins.base();
        if (b == "st" || b == "bra" || b == "scfg" || b == "stva8" || b == "stva32" || b == "stvs" || b == "stvc") return "";
        return ins.args[0].name;
    }
    return "";
}

// PTX registers equal on every lane of a warp (fixpoint)
std::set<std::string> uniform_regs(const ptx::Kernel &k) {
    std::map<std::string, std::vector<const ptx::Instr *>> defs;
    for (const ptx::Instr &ins : k.instrs) {
        std::string d = defines(ins);
        if (!d.empty()) defs[d].push_back(&ins);
    }
    std::set<std::string> uni;
    for (const auto &p : defs) uni.insert(p.first);
    // registers that are exactly %tid.x (mov), so that the field patterns below see through the copy
    std::set<std::string> tidx;
    for (const ptx::Instr &ins : k.instrs)
        if (ins.base() == "mov" && ins.args.size() == 2 && ins.args[1].kind == ptx::Operand::SREG && ins.args[1].name == "%tid.x" && defs[ins.args[0].name].size() == 1)
            tidx.insert(ins.args[0].name);
    auto is_tidx = [&](const ptx::Operand &o) {
        return (o.kind == ptx::Operand::SREG && o.name == "%tid.x") || (o.kind == ptx::Operand::REG && tidx.count(o.name));
    };
    auto ok = [&](const ptx::Instr &ins) {
        const std::string &b = ins.base();
        if (b == "tid" || b == "wsum" || b == "wmax" || b == "udiv" || b == "usqrt") return b != "tid";  // reductions: uniform results
        // fields of %tid.x above the lane bits are the same on every lane: tid.x >> k (k >= 5), tid.x < 32 m, tid.x & ~31
        if (ins.args.size() == 3 && is_tidx(ins.args[1]) && ins.args[2].kind == ptx::Operand::IMM) {
            int64_t c = ins.args[2].value;
            if ((b == "shr") && c >= 5) return true;
            if (b == "setp" && ins.parts.size() == 3 && (ins.parts[1] == "lt" || ins.parts[1] == "ge" || ins.parts[1] == "gt" || ins.parts[1] == "le") && c % 32 == 0) return true;
            if (b == "and" && (c & 31) == 0) return true;
        }
        for (size_t i = 1; i < ins.args.size(); i++) {
            const ptx::Operand &a = ins.args[i];
            if (a.kind == ptx::Operand::REG && !uni.count(a.name)) return false;
            if (a.kind == ptx::Operand::SREG && !is_sreg_uniform(a.name)) return false;
            if (a.kind == ptx::Operand::MEM && a.base->kind == ptx::Operand::REG && !uni.count(a.base->name)) return false;
        }
        return true;
    };
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto it = uni.begin(); it != uni.end();) {
            bool all = true;
            for (const ptx::Instr *ins : defs[*it])
                if (!ok(*ins)) all = false;
            if (!all) {
                it = uni.erase(it);
                changed = true;
            } else {
                ++it;
            }
        }
    }
    return uni;
}

std::vector<std::pair<int, int>> basic_blocks(const ptx::Kernel &k);

// Reorder basic blocks so each loop is contiguous, header first, and the only branches backwards are back-edges
// (compilers rotate loops). Fall-throughs that no longer hold become bra.uni; a conditional branch whose target is now
// next is inverted. A structured kernel comes back unchanged. Port of lower.relayout.
ptx::Kernel relayout(const ptx::Kernel &k) {
    std::vector<std::pair<int, int>> bbs = basic_blocks(k);
    const int n = (int)bbs.size();
    std::map<int, int> at;
    for (int i = 0; i < n; i++) at[bbs[(size_t)i].first] = i;
    auto edges = [&](int i) {  // (taken target or -1, fall-through or -1)
        const ptx::Instr &last = k.instrs[(size_t)bbs[(size_t)i].second - 1];
        int fall = i + 1 < n ? i + 1 : -1;
        if (last.base() == "ret") return std::make_pair(-1, -1);
        if (last.base() == "bra") {
            int t = at.at(k.labels.at(last.args[0].name));
            return std::make_pair(t, last.guard.empty() ? -1 : fall);
        }
        return std::make_pair(-1, fall);
    };
    std::vector<std::vector<int>> succ((size_t)n), pred((size_t)n);
    for (int i = 0; i < n; i++) {
        auto [t, f] = edges(i);
        if (t >= 0) succ[(size_t)i].push_back(t);
        if (f >= 0) succ[(size_t)i].push_back(f);
    }
    for (int i = 0; i < n; i++)
        for (int s : succ[(size_t)i]) pred[(size_t)s].push_back(i);
    std::vector<std::set<int>> dom((size_t)n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) dom[(size_t)i].insert(j);
    dom[0] = {0};
    bool changed = true;
    while (changed) {
        changed = false;
        for (int i = 1; i < n; i++) {
            std::set<int> d;
            bool first = true;
            for (int p : pred[(size_t)i]) {
                if (first) {
                    d = dom[(size_t)p];
                    first = false;
                } else {
                    std::set<int> x;
                    for (int v : d)
                        if (dom[(size_t)p].count(v)) x.insert(v);
                    d = x;
                }
            }
            d.insert(i);
            if (d != dom[(size_t)i]) {
                dom[(size_t)i] = d;
                changed = true;
            }
        }
    }
    std::map<int, std::set<int>> loops;  // header -> natural loop body
    for (int u = 0; u < n; u++)
        for (int h : succ[(size_t)u])
            if (dom[(size_t)u].count(h)) {
                std::set<int> &body = loops[h];
                body.insert(h);
                std::vector<int> work = {u};
                while (!work.empty()) {
                    int x = work.back();
                    work.pop_back();
                    if (!body.count(x)) {
                        body.insert(x);
                        for (int p : pred[(size_t)x]) work.push_back(p);
                    }
                }
            }
    std::function<std::vector<int>(const std::set<int> &, int)> order = [&](const std::set<int> &region, int head) {
        std::vector<int> inner;
        for (const auto &lp : loops) {
            int h = lp.first;
            bool inside = region.count(h) && h != head;
            for (int x : lp.second)
                if (!region.count(x)) inside = false;
            if (inside) inner.push_back(h);
        }
        std::vector<int> top;
        for (int h : inner) {
            bool nested = false;
            for (int g : inner)
                if (h != g && loops[g].count(h) && loops[h].size() < loops[g].size()) nested = true;
            if (!nested) top.push_back(h);
        }
        std::map<int, int> rep_of;
        for (int x : region) rep_of[x] = x;
        for (int h : top)
            for (int x : loops[h]) rep_of[x] = h;
        std::set<int> nodes;
        for (const auto &p : rep_of) nodes.insert(p.second);
        std::map<int, int> indeg;
        std::map<int, std::set<int>> out;
        for (int x : nodes) indeg[x] = 0;
        for (int x : region)
            for (int y : succ[(size_t)x])
                if (region.count(y) && y != head && rep_of[y] != rep_of[x] && !out[rep_of[x]].count(rep_of[y])) {
                    out[rep_of[x]].insert(rep_of[y]);
                    indeg[rep_of[y]]++;
                }
        std::vector<int> seq, ready;
        for (int x : nodes)
            if (indeg[x] == 0) ready.push_back(x);
        std::sort(ready.begin(), ready.end());
        while (!ready.empty()) {
            int x = ready.front();
            ready.erase(ready.begin());
            if (std::find(top.begin(), top.end(), x) != top.end()) {
                std::vector<int> sub = order(loops[x], x);
                seq.insert(seq.end(), sub.begin(), sub.end());
            } else {
                seq.push_back(x);
            }
            for (int y : out[x])
                if (--indeg[y] == 0) {
                    ready.push_back(y);
                    std::sort(ready.begin(), ready.end());
                }
        }
        if (std::set<int>(seq.begin(), seq.end()).size() != region.size()) throw Unsupported("an irreducible CFG");
        return seq;
    };
    std::set<int> all;
    for (int i = 0; i < n; i++) all.insert(i);
    std::vector<int> seq = order(all, -1);
    bool identity = true;
    for (int i = 0; i < n; i++)
        if (seq[(size_t)i] != i) identity = false;
    if (identity) return k;
    std::map<int, std::string> name;
    for (const auto &p : k.labels)
        if (at.count(p.second)) name[at.at(p.second)] = p.first;
    for (int i = 0; i < n; i++)
        if (!name.count(i)) name[i] = "$L__qgpu" + std::to_string(i);
    ptx::Kernel out = k;
    out.instrs.clear();
    out.labels.clear();
    for (size_t pos = 0; pos < seq.size(); pos++) {
        int i = seq[pos];
        int nxt = pos + 1 < seq.size() ? seq[pos + 1] : -1;
        out.labels[name[i]] = (int)out.instrs.size();
        std::vector<ptx::Instr> body(k.instrs.begin() + bbs[(size_t)i].first, k.instrs.begin() + bbs[(size_t)i].second);
        auto [t, fall] = edges(i);
        ptx::Instr &last = body.back();
        if (last.base() == "bra" && !last.guard.empty() && fall != nxt && t == nxt) {
            last.neg = !last.neg;
            ptx::Operand lab;
            lab.kind = ptx::Operand::LABEL;
            lab.name = name[fall];
            last.args = {lab};
            fall = -1;
        }
        if (fall >= 0 && fall != nxt) {
            ptx::Instr j;
            j.op = "bra.uni";
            j.parts = {"bra", "uni"};
            ptx::Operand lab;
            lab.kind = ptx::Operand::LABEL;
            lab.name = name[fall];
            j.args = {lab};
            body.push_back(j);
        }
        out.instrs.insert(out.instrs.end(), body.begin(), body.end());
    }
    return out;
}

struct Lowerer {
    ptx::Kernel k;
    const Options &opt;
    std::vector<ParamSpec> params;
    int gx, gy, bx, by;
    bool one_d;
    int lbx, lby, lgx;
    std::map<std::string, Val> vals;
    std::map<std::string, Val> known;  // loop variables whose value here is a known constant
    std::map<std::string, int> mreg;   // loop variable -> home register
    std::map<std::string, int> ndefs;
    std::set<std::string> uniform;
    std::map<std::string, uint32_t> shared;  // .shared name -> M word
    std::vector<char> vuni;  // per virtual register: uniform
    int nreg = 0;
    std::vector<std::vector<KInstr>> code;  // groups
    struct Term {
        int kind = SM_BR_NEXT;  // SM_BR_*; jump / cond carry a target group
        Val pred;
        bool neg = false;
        int target = 0;
        bool is_pred = false;
    };
    std::vector<Term> terms;
    std::vector<int> group_of;  // PTX block -> group
    std::vector<std::pair<int, int>> bbs;
    std::map<int, int> bb_at;
    std::set<int> ret_blocks, targets;
    std::vector<Val> exit_pred;
    std::map<int, int> exit_at;  // group -> position of its latest exit setp
    std::map<std::vector<int>, int> gathered;
    std::map<uint32_t, int> const_reg;  // constants materialized in the current group
    int linear_reg = -1;
    bool branched = false;
    int wpb = 1;

    Lowerer(const ptx::Kernel &kk, const Options &o) : k(relayout(kk)), opt(o) {}

    int fresh(bool uni) {
        vuni.push_back((char)uni);
        return nreg++;
    }
    bool isuni(int r) const {
        if (r == VZERO || r == VARGS) return true;
        return vuni[(size_t)r] != 0;
    }
    bool val_uni(const Val &v) const {
        switch (v.k) {
        case Val::CONST: case Val::PTR: case Val::KNOWN: return true;
        case Val::REG: case Val::SCALED: case Val::ADDR: return isuni(v.r);
        case Val::WIDE: return val_uni(*v.inner);
        case Val::GATHER: return !v.linear && !v.bits.empty() && v.bits[0] >= 5;  // bits above the lane's: the same on every lane
        case Val::SREG: return is_sreg_uniform(v.name);
        case Val::PRED: return v.puni || (isuni(v.ra) && (v.rhs_const || isuni(v.rhs_r)));
        default: return false;
        }
    }
    int home(const std::string &name) {
        auto it = mreg.find(name);
        if (it != mreg.end()) return it->second;
        int r = fresh(uniform.count(name) != 0);
        mreg[name] = r;
        return r;
    }
    bool multi(const std::string &name) const {
        auto it = ndefs.find(name);
        return it != ndefs.end() && it->second > 1;
    }

    std::vector<int> sreg_bits(const std::string &name) const {
        // t = ((ctaid.y * gx + ctaid.x) * by + tid.y) * bx + tid.x: with power-of-two sizes each is a bit field
        int lo, width, size;
        bool open = false;
        if (name == "%tid.x") lo = 0, width = lbx, size = bx;
        else if (name == "%tid.y") lo = lbx, width = lby, size = by;
        else if (name == "%ctaid.x") lo = lbx + lby, width = lgx, size = gx;
        else if (name == "%ctaid.y") lo = lbx + lby + lgx, width = 0, size = gy, open = true;
        else throw Unsupported("special register " + name);
        std::vector<int> out;
        if (size == 1) return out;
        if (!pow2(size) && !(name == "%ctaid.y" || (name == "%ctaid.x" && gy == 1))) throw Unsupported(name + " of a non-power-of-two launch size");
        if ((name == "%tid.y" || name == "%ctaid.x") && !pow2((int64_t)bx * (name == "%ctaid.x" ? by : 1)))
            throw Unsupported(name + " with a non-power-of-two block");
        int top = (open || (name == "%ctaid.x" && gy == 1)) ? SM_TBITS : lo + width;
        for (int i = lo; i < top; i++) out.push_back(i);
        return out;
    }
    Val sreg_value(const std::string &name) {
        if (name == "%ntid.x") return Val::Const(bx);
        if (name == "%ntid.y") return Val::Const(by);
        if (name == "%nctaid.x") return Val::Const(gx);
        if (name == "%nctaid.y") return Val::Const(gy);
        std::vector<int> bits = sreg_bits(name);
        return bits.empty() ? Val::Const(0) : Val::Gather(bits);
    }

    Val val(const ptx::Operand &a, bool fold = false, bool raw = false) {
        Val v;
        if (a.kind == ptx::Operand::REG) {
            if (fold && known.count(a.name)) v = known[a.name];
            else if (vals.count(a.name)) v = vals[a.name];
            else throw Unsupported(a.name + " read before it is defined");
        } else if (a.kind == ptx::Operand::IMM) {
            v = Val::Const(a.value);
        } else if (a.kind == ptx::Operand::SREG) {
            v = Val::Sreg(a.name);
        } else if (a.kind == ptx::Operand::SYM && shared.count(a.name)) {
            v = Val::Ptr(shared[a.name]);
        } else {
            throw Unsupported("operand " + a.name);
        }
        if (raw || v.k != Val::SREG) return v;
        return sreg_value(v.name);
    }

    // mad.lo(ctaid.a, ntid.a, tid.a): the global index along axis a
    bool global_index(const Val &x, const Val &y, const Val &z, Val &out) {
        for (const char *ax : {"x", "y"}) {
            std::string c = std::string("%ctaid.") + ax, n = std::string("%ntid.") + ax, t = std::string("%tid.") + ax;
            bool pair = (x.is(Val::SREG) && y.is(Val::SREG)) && ((x.name == c && y.name == n) || (x.name == n && y.name == c));
            if (pair && z.is(Val::SREG) && z.name == t) {
                if (ax[0] == 'x' && one_d) {
                    out = Val::Linear();
                    return true;
                }
                int size = ax[0] == 'x' ? bx : by;
                if (!pow2(size)) throw Unsupported(std::string("global ") + ax + " index with a non-power-of-two block");
                std::vector<int> bits = sreg_bits(t);
                std::vector<int> more = sreg_bits(c);
                bits.insert(bits.end(), more.begin(), more.end());
                out = Val::Gather(bits);
                return true;
            }
        }
        return false;
    }

    KInstr mk(int op) {
        KInstr i;
        i.op = op;
        return i;
    }
    KInstr add_imm(int rd, int ra, uint32_t imm) {
        KInstr i = mk(SM_ADD);
        i.rd = rd; i.ra = ra; i.imm = imm; i.useimm = true;
        return i;
    }
    KInstr add_rr(int rd, int ra, int rb) {
        KInstr i = mk(SM_ADD);
        i.rd = rd; i.ra = ra; i.rb = rb;
        return i;
    }
    KInstr mad_imm(int rd, int ra, uint32_t imm, int rc) {
        KInstr i = mk(SM_MAD);
        i.rd = rd; i.ra = ra; i.imm = imm; i.useimm = true; i.rc = rc;
        return i;
    }
    bool regs_uni(std::initializer_list<int> rs) const {
        for (int r : rs)
            if (!isuni(r)) return false;
        return true;
    }

    // a machine register holding v
    int reg_of(const Val &v, std::vector<KInstr> &out) {
        switch (v.k) {
        case Val::REG: return v.r;
        case Val::CONST: {
            // one uniform register per constant per group (an unrolled loop materializes the same base many times)
            auto it = const_reg.find((uint32_t)v.v);
            if (it != const_reg.end()) return it->second;
            int r = fresh(true);
            out.push_back(add_imm(r, VZERO, (uint32_t)v.v));
            const_reg[(uint32_t)v.v] = r;
            return r;
        }
        case Val::PTR: {
            int r = fresh(true);
            out.push_back(add_imm(r, VZERO, (uint32_t)v.v));
            return r;
        }
        case Val::ADDR: {
            int r = fresh(isuni(v.r));
            out.push_back(add_imm(r, v.r, (uint32_t)v.v));
            return r;
        }
        case Val::SCALED: {  // used as a number after all: 4 k r
            int r = fresh(isuni(v.r));
            out.push_back(mad_imm(r, v.r, (uint32_t)(4 * v.v), VZERO));
            return r;
        }
        case Val::GATHER: {
            if (v.linear) {
                if (linear_reg >= 0) return linear_reg;
                int r = fresh(false);
                KInstr i = mk(SM_TID);
                i.rd = r;
                out.push_back(i);
                linear_reg = r;
                return r;
            }
            auto it = gathered.find(v.bits);
            if (it != gathered.end()) return it->second;
            // contiguous bits lo..hi: (t >> lo) & mask
            for (size_t i = 1; i < v.bits.size(); i++)
                if (v.bits[i] != v.bits[i - 1] + 1) throw Unsupported("a gather of non-contiguous thread-index bits");
            int r = fresh(v.bits[0] >= 5);
            KInstr i = mk(SM_SREG);
            i.rd = r;
            i.imm = (uint32_t)v.bits[0] | (uint32_t)v.bits.size() << 8;
            out.push_back(i);
            gathered[v.bits] = r;
            return r;
        }
        case Val::WIDE: return reg_of(*v.inner, out);
        default: throw Unsupported("cannot materialize a value of kind " + std::to_string(v.k));
        }
    }

    int dest(const std::string &name) {
        if (!multi(name)) return -2;  // decided by the caller: a fresh register of the right kind
        return home(name);
    }
    // a fresh destination for a single-definition value, or the home of a loop variable; uni says what the value is
    int dest_for(const std::string &name, bool uni) {
        if (multi(name)) {
            int h = home(name);
            if (!isuni(h) || uni) return h;
            throw Unsupported(name + ": a uniform loop variable assigned a per-lane value");
        }
        return fresh(uni);
    }

    void define(const std::string &name, const Val &v, std::vector<KInstr> &out, bool ptr = false) {
        if (!multi(name)) {
            vals[name] = v;
            return;
        }
        int r = home(name);
        known.erase(name);
        if (v.is(Val::CONST)) known[name] = v;
        if (isuni(r) && !val_uni(v)) throw Unsupported(name + ": a uniform loop variable assigned a per-lane value");
        if (v.is(Val::REG)) {
            if (v.r != r) out.push_back(add_imm(r, v.r, 0));
        } else if (v.is(Val::CONST)) {
            out.push_back(add_imm(r, VZERO, (uint32_t)v.v));
        } else if (v.is(Val::PTR)) {
            out.push_back(add_imm(r, VZERO, (uint32_t)v.v));
        } else if (v.is(Val::ADDR)) {
            out.push_back(add_imm(r, v.r, (uint32_t)v.v));
        } else if (v.is(Val::WIDE)) {
            out.push_back(add_imm(r, reg_of(*v.inner, out), 0));
        } else if (v.is(Val::GATHER)) {
            out.push_back(add_imm(r, reg_of(v, out), 0));
        } else {
            throw Unsupported(name + ": a loop variable holding a value of kind " + std::to_string(v.k));
        }
        bool is_ptr = ptr || v.is(Val::PTR) || (v.is(Val::ADDR) && v.ptr) || (v.is(Val::REG) && v.ptr);
        vals[name] = Val::Reg(r, is_ptr);
    }

    void emit_setp(const Val &p, int pd, std::vector<KInstr> &out) {
        KInstr i = mk(SM_SETP);
        i.ra = p.ra;
        i.pd = pd;
        i.cmp = p.cmp;
        i.sgn = p.sgn;
        if (p.rhs_const) {
            i.useimm = true;
            i.imm = (uint32_t)p.rhs_v;
        } else {
            i.rb = p.rhs_r;
        }
        out.push_back(i);
    }

    std::pair<int, int64_t> word_addr(const ptx::Operand &a) {
        const ptx::Operand &base_op = *a.base;
        int64_t off = a.value;
        if (off % 4) throw Unsupported("a byte offset that is not a whole word");
        Val v;
        if (base_op.kind == ptx::Operand::REG) {
            if (!vals.count(base_op.name)) throw Unsupported("address register " + base_op.name + " undefined");
            v = vals[base_op.name];
        } else if (shared.count(base_op.name)) {
            v = Val::Ptr(shared[base_op.name]);
        } else {
            throw Unsupported("address " + base_op.name);
        }
        if (v.is(Val::ADDR) && v.ptr) return {v.r, v.v + off / 4};
        if (v.is(Val::REG) && v.ptr) return {v.r, off / 4};
        if (v.is(Val::PTR)) return {VZERO, v.v + off / 4};
        throw Unsupported("address of kind " + std::to_string(v.k) + (v.is(Val::ADDR) ? " (a 32-bit sum, not a pointer)" : ""));
    }

    Val scaled(int r, int64_t kk, std::vector<KInstr> &out) {
        // a loop variable's register may be rewritten before the use: compute the word offset now
        bool is_home = false;
        for (const auto &p : mreg)
            if (p.second == r) is_home = true;
        if (!is_home) return Val::Scaled(r, kk);
        int w = fresh(isuni(r));
        if (kk == 1) out.push_back(add_imm(w, r, 0));
        else out.push_back(mad_imm(w, r, (uint32_t)kk, VZERO));
        return Val::Scaled(w, 1);
    }

    int words_reg(const Val &sc, std::vector<KInstr> &out, int plus = -1) {
        if (sc.v == 1 && plus < 0) return sc.r;
        int t = fresh(isuni(sc.r) && (plus < 0 || isuni(plus)));
        if (sc.v == 1) out.push_back(add_rr(t, plus, sc.r));
        else out.push_back(mad_imm(t, sc.r, (uint32_t)sc.v, plus < 0 ? VZERO : plus));
        return t;
    }

    // a value as (register, immediate) for an instruction that adds the two: a constant alone rides on the zero register
    std::pair<int, int64_t> addr_of(const Val &v, std::vector<KInstr> &out) {
        if (v.is(Val::CONST)) return {VZERO, (int64_t)(uint32_t)v.v};
        if (v.is(Val::ADDR)) return {v.r, v.v};
        if (v.is(Val::PTR)) return {VZERO, v.v};
        return {reg_of(v, out), 0};
    }

    Val as_wide(const Val &x) {
        if (x.is(Val::REG) && !x.ptr) return Val::Wide(x);
        return x;
    }
    int64_t whole_words(const Val &c) {
        if (c.v % 4) throw Unsupported("pointer + bytes that are not a whole word");
        return c.v / 4;
    }

    void barrier(int &gi, std::vector<KInstr> *&out) {
        terms[(size_t)gi].kind = SM_BR_BAR;
        code.emplace_back();
        terms.emplace_back();
        gathered.clear();
        const_reg.clear();
        linear_reg = -1;
        gi = (int)code.size() - 1;
        out = &code.back();
    }

    int mul_hi(const Val &x, const Val &y, bool sgn, const std::string &name, std::vector<KInstr> &out) {
        // the machine's mulhi is unsigned; signed: hi_s = hi_u - (x < 0 ? y : 0) - (y < 0 ? x : 0)
        int ra = reg_of(x, out);
        int rb = y.is(Val::CONST) ? VZERO : reg_of(y, out);
        bool uni = isuni(ra) && (y.is(Val::CONST) || isuni(rb));
        int r = sgn ? fresh(uni) : dest_for(name, uni);
        KInstr i = mk(SM_MULHI);
        i.rd = r; i.ra = ra; i.rb = rb; i.useimm = y.is(Val::CONST); i.imm = y.is(Val::CONST) ? (uint32_t)y.v : 0;
        out.push_back(i);
        if (!sgn) return r;
        // x's sign: subtract y; y's sign: subtract x
        for (int pass = 0; pass < 2; pass++) {
            if (pass == 1 && y.is(Val::CONST)) {
                if (y.v >> 31 & 1) {
                    int t = fresh(uni);
                    KInstr s = mk(SM_SUB);
                    s.rd = t; s.ra = r; s.rb = ra;
                    out.push_back(s);
                    r = t;
                }
                continue;
            }
            int neg_of = pass == 0 ? ra : rb;
            Val other = pass == 0 ? y : Val::Reg(ra);
            Val p;
            p.k = Val::PRED; p.cmp = SM_CMP_LT; p.sgn = true; p.ra = neg_of; p.rhs_const = true; p.rhs_v = 0;
            emit_setp(p, SEL_P, out);
            int corr = fresh(uni), t = fresh(uni);
            KInstr sel = mk(SM_SELP);
            sel.rd = corr; sel.ra = reg_of(other, out); sel.useimm = true; sel.imm = 0; sel.guard = SEL_P;
            out.push_back(sel);
            KInstr s = mk(SM_SUB);
            s.rd = t; s.ra = r; s.rb = corr;
            out.push_back(s);
            r = t;
        }
        if (multi(name)) {
            define(name, Val::Reg(r), out);
            return home(name);
        }
        return r;
    }

    void lower_bits(const std::string &base_op, const std::string &ty, const std::string &name, std::vector<Val> xs, std::vector<KInstr> &out) {
        if (base_op == "and" || base_op == "or" || base_op == "xor" || base_op == "not") {
            Val x = xs[0], y = base_op == "not" ? Val::Const(M32) : xs[1];
            if (x.is(Val::CONST)) std::swap(x, y);
            int ra = reg_of(x, out);
            int rb = y.is(Val::CONST) ? VZERO : reg_of(y, out);
            bool uni = isuni(ra) && (y.is(Val::CONST) || isuni(rb));
            int r = dest_for(name, uni);
            KInstr i = mk(base_op == "and" ? SM_AND : base_op == "or" ? SM_OR : SM_XOR);
            i.rd = r; i.ra = ra; i.rb = rb; i.useimm = y.is(Val::CONST); i.imm = y.is(Val::CONST) ? (uint32_t)y.v : 0;
            out.push_back(i);
            vals[name] = Val::Reg(r);
            return;
        }
        Val x = xs[0], kv = xs[1];
        if (!kv.is(Val::CONST)) {
            // a run-time amount: through a uniform register (the sh field 63)
            int rc = reg_of(kv, out);
            if (!isuni(rc)) throw Unsupported(base_op + " by a per-lane amount");
            int ra = reg_of(x, out);
            int r = dest_for(name, isuni(ra));
            KInstr i = mk(base_op == "shl" ? SM_SHL : SM_SHR);
            i.rd = r; i.ra = ra; i.rc = rc; i.sh = 63; i.sgn = ty == "s32";
            out.push_back(i);
            vals[name] = Val::Reg(r);
            return;
        }
        int64_t kk = kv.v;
        if (kk == 0) {
            define(name, x, out);
            return;
        }
        if (base_op == "shl") {
            if (kk >= 32) {
                define(name, Val::Const(0), out);
                return;
            }
            if (kk >= 2 && !multi(name)) {  // a byte offset if it meets a pointer, else a number where it is used
                vals[name] = scaled(reg_of(x, out), (int64_t)1 << (kk - 2), out);
                return;
            }
            int ra = reg_of(x, out);
            int r = dest_for(name, isuni(ra));
            KInstr i = mk(SM_SHL);
            i.rd = r; i.ra = ra; i.sh = (int)kk;
            out.push_back(i);
            vals[name] = Val::Reg(r);
            return;
        }
        // a field of the thread index shifted down is a smaller field: no instruction, and uniform above the lane bits
        if (base_op == "shr" && x.is(Val::GATHER) && !x.linear && !multi(name)) {
            std::vector<int> bits(x.bits.begin() + std::min<size_t>((size_t)kk, x.bits.size()), x.bits.end());
            vals[name] = bits.empty() ? Val::Const(0) : Val::Gather(bits);
            return;
        }
        // shr: logical for u32 / b32, arithmetic for s32
        int ra = reg_of(x, out);
        int r = dest_for(name, isuni(ra));
        KInstr i = mk(SM_SHR);
        i.rd = r; i.ra = ra; i.sh = (int)std::min<int64_t>(kk, 63); i.sgn = ty == "s32";
        out.push_back(i);
        vals[name] = Val::Reg(r);
    }

    int shr_reg(int r, int kk, std::vector<KInstr> &out) {
        if (kk == 0) return r;
        int q = fresh(isuni(r));
        KInstr i = mk(SM_SHR);
        i.rd = q; i.ra = r; i.sh = kk;
        out.push_back(i);
        return q;
    }

    // n / d for a register n and a constant d > 0, unsigned: Granlund and Montgomery
    int udiv_const(int ra, uint32_t d, std::vector<KInstr> &out) {
        if (d == 1) return ra;
        int l = bitlen((int64_t)d - 1);
        if (d == (1u << l)) return shr_reg(ra, l, out);
        uint64_t m = (((uint64_t)1 << 32) * (((uint64_t)1 << l) - d)) / d + 1;
        bool uni = isuni(ra);
        int t = fresh(uni), u = fresh(uni), v = fresh(uni);
        KInstr hi = mk(SM_MULHI);
        hi.rd = t; hi.ra = ra; hi.useimm = true; hi.imm = (uint32_t)m;
        out.push_back(hi);
        KInstr s = mk(SM_SUB);
        s.rd = u; s.ra = ra; s.rb = t;
        out.push_back(s);
        out.push_back(add_rr(v, shr_reg(u, 1, out), t));
        return shr_reg(v, l - 1, out);
    }

    void div_rem(const std::string &op, const std::string &name, const Val &x, const Val &y, std::vector<KInstr> &out) {
        bool sgn = op.size() >= 3 && op.substr(op.size() - 3) == "s32";
        bool is_div = op.compare(0, 3, "div") == 0;
        if (!y.is(Val::CONST)) throw Unsupported(op + " by a value not known at build time (use udiv)");
        int64_t d = sgn && (y.v >> 31 & 1) ? y.v - ((int64_t)1 << 32) : y.v;
        if (d == 0) throw Unsupported(op + " by zero");
        if (x.is(Val::CONST)) {
            int64_t n = sgn && (x.v >> 31 & 1) ? x.v - ((int64_t)1 << 32) : x.v;
            int64_t q = (std::llabs(n) / std::llabs(d)) * ((n < 0) == (d < 0) ? 1 : -1);
            define(name, Val::Const(is_div ? q : n - q * d), out);
            return;
        }
        int rn = reg_of(x, out);
        bool uni = isuni(rn);
        int q;
        if (!sgn) {
            q = udiv_const(rn, (uint32_t)d, out);
        } else {
            int neg = fresh(uni), a = fresh(uni), qn = fresh(uni);
            q = fresh(uni);
            KInstr s = mk(SM_SUB);
            s.rd = neg; s.ra = VZERO; s.rb = rn;
            out.push_back(s);
            Val p;
            p.k = Val::PRED; p.cmp = SM_CMP_LT; p.sgn = true; p.ra = rn; p.rhs_const = true; p.rhs_v = 0;
            emit_setp(p, SEL_P, out);
            KInstr sel = mk(SM_SELP);
            sel.rd = a; sel.ra = neg; sel.rb = rn; sel.guard = SEL_P;
            out.push_back(sel);
            int qa = udiv_const(a, (uint32_t)std::llabs(d), out);
            KInstr s2 = mk(SM_SUB);
            s2.rd = qn; s2.ra = VZERO; s2.rb = qa;
            out.push_back(s2);
            emit_setp(p, SEL_P, out);
            KInstr sel2 = mk(SM_SELP);
            sel2.rd = q; sel2.ra = d > 0 ? qn : qa; sel2.rb = d > 0 ? qa : qn; sel2.guard = SEL_P;
            out.push_back(sel2);
        }
        int r = dest_for(name, uni);
        if (is_div) out.push_back(add_imm(r, q, 0));
        else out.push_back(mad_imm(r, q, (uint32_t)(-d), rn));
        vals[name] = Val::Reg(r);
    }

    // the guard of a side effect: the early exit's predicate, recomputed right before it. Predicates never cross blocks,
    // and a block may end anywhere (an stv forbids a later ld in its block, live-outs run out): a setp adjacent to its
    // user always lands with it, since a block ending between them would carry a predicate out and the former never
    // allows that split.
    void side_guard(KInstr &i, int gi, std::vector<KInstr> &out) {
        if (exit_pred.empty()) return;
        emit_setp(exit_pred[0], EXIT_P, out);
        exit_at[gi] = (int)out.size() - 1;
        i.guard = EXIT_P;
        i.gneg = true;
    }

    Launch run();
};

std::vector<std::pair<int, int>> basic_blocks(const ptx::Kernel &k) {
    std::set<int> starts = {0};
    for (const auto &p : k.labels) starts.insert(p.second);
    for (size_t i = 0; i < k.instrs.size(); i++) {
        const std::string &b = k.instrs[i].base();
        if (b == "bra" || b == "ret") starts.insert((int)i + 1);
    }
    std::vector<int> s;
    for (int x : starts)
        if (x <= (int)k.instrs.size()) s.push_back(x);
    std::vector<std::pair<int, int>> out;
    for (size_t i = 0; i < s.size(); i++) {
        int a = s[i], b = i + 1 < s.size() ? s[i + 1] : (int)k.instrs.size();
        if (a < b) out.emplace_back(a, b);
    }
    return out;
}

Launch Lowerer::run() {
    gy = std::max(gy, 1);
    by = std::max(by, 1);
    if ((int64_t)bx * by % SM_LANES) throw Unsupported("threads per block must be a multiple of 32");
    int64_t most = (int64_t)SM_LANES * (1 << SM_WBITS);
    if ((int64_t)gx * gy * bx * by > most) throw Unsupported("more than 256 warps");
    one_d = gy == 1 && by == 1;
    lbx = bitlen(bx) - 1;
    lby = bitlen(by) - 1;
    lgx = bitlen(gx) - 1;
    // .shared arrays follow at the shared base
    uint32_t top = opt.shared_base;
    for (const ptx::Shared &sh : k.shared) {
        if (sh.size % 4) throw Unsupported("shared " + sh.name + " is not whole words");
        shared[sh.name] = top;
        top += (uint32_t)sh.size / 4;
    }
    bool barriers = false;
    for (const ptx::Instr &ins : k.instrs)
        if (ins.base() == "bar" || ins.base() == "barrier") barriers = true;
        else if (ins.base() == "shfl" || ins.base() == "atom") throw Unsupported(ins.base() + " (use wsum / wmax and stv)");
    wpb = barriers ? bx * by / SM_LANES : 1;
    if (wpb != 1 && wpb != 2 && wpb != 4 && wpb != 8) throw Unsupported("a barrier with " + std::to_string(wpb) + " warps per thread block");
    if (params.size() != k.params.size()) throw Unsupported("parameter count");
    for (const ptx::Instr &ins : k.instrs) {
        std::string d = defines(ins);
        if (!d.empty()) ndefs[d]++;
    }
    uniform = uniform_regs(k);
    bbs = basic_blocks(k);
    for (size_t i = 0; i < bbs.size(); i++) bb_at[bbs[i].first] = (int)i;
    for (size_t i = 0; i < bbs.size(); i++)
        if (k.instrs[(size_t)bbs[i].first].base() == "ret") ret_blocks.insert((int)i);
    for (const ptx::Instr &ins : k.instrs)
        if (ins.base() == "bra") targets.insert(bb_at.at(k.labels.at(ins.args[0].name)));
    // divergent control flow: not lowered here (the 2B's kernels branch uniformly)
    {
        bool seen = false;
        for (const ptx::Instr &ins : k.instrs) {
            if (ins.base() != "bra") continue;
            int t = bb_at.at(k.labels.at(ins.args[0].name));
            bool exit_like = !ins.guard.empty() && ret_blocks.count(t) && !seen && !ins.neg;
            if (!ins.guard.empty() && !uniform.count(ins.guard) && !exit_like) throw Unsupported("divergent branch on " + ins.guard);
            seen = seen || !exit_like;
        }
    }
    // loop variables get their register before any block reads them
    for (const auto &p : ndefs)
        if (p.second > 1) vals[p.first] = Val::Reg(home(p.first));

    for (size_t bi = 0; bi < bbs.size(); bi++) {
        int start = bbs[bi].first, end = bbs[bi].second;
        bool new_group = bi == 0 || targets.count((int)bi) || terms.back().kind != SM_BR_NEXT;
        if (new_group) {
            code.emplace_back();
            terms.emplace_back();
            gathered.clear();
            const_reg.clear();
            linear_reg = -1;
            known.clear();
        }
        group_of.push_back((int)code.size() - 1);
        std::vector<KInstr> *out = &code.back();
        int gi = (int)code.size() - 1;
        for (int ii = start; ii < end; ii++) {
            const ptx::Instr &ins = k.instrs[(size_t)ii];
            const std::string &op = ins.op, &base_op = ins.base();
            const std::vector<ptx::Operand> &a = ins.args;
            if (!ins.guard.empty() && base_op != "bra") throw Unsupported("guarded " + op);
            if (base_op == "ret") {
                terms[(size_t)gi].kind = SM_BR_END;
            } else if (op.compare(0, 8, "ld.param") == 0) {
                // which parameter: by name
                const std::string &pname = a[1].base->name;
                size_t pi = 0;
                while (pi < k.params.size() && k.params[pi].name != pname) pi++;
                if (pi == k.params.size()) throw Unsupported("parameter " + pname);
                const ParamSpec &ps = params[pi];
                if (ps.kind == ParamSpec::CONST) define(a[0].name, Val::Const(ps.value), *out);
                else if (ps.kind == ParamSpec::PTR) define(a[0].name, Val::Ptr(ps.value), *out);
                else {
                    int r = fresh(true);
                    KInstr l = mk(SM_LD);
                    l.rd = r; l.ra = VARGS; l.imm = (uint32_t)ps.value; l.bank = SM_BANK_M;
                    out->push_back(l);
                    define(a[0].name, Val::Reg(r, k.params[pi].size == 8), *out, k.params[pi].size == 8);
                }
            } else if (op == "cvta.to.global.u64") {
                define(a[0].name, val(a[1]), *out, true);
            } else if (base_op == "mov") {
                define(a[0].name, val(a[1], false, true), *out);
            } else if (op == "mad.lo.s32") {
                Val gidx;
                Val rx = val(a[1], false, true), ry = val(a[2], false, true), rz = val(a[3], false, true);
                if (global_index(rx, ry, rz, gidx)) {
                    int r = reg_of(gidx, *out);
                    if (multi(a[0].name)) define(a[0].name, Val::Reg(r), *out);
                    else vals[a[0].name] = Val::Reg(r);
                    continue;
                }
                Val x = val(a[1]), y = val(a[2]), z = val(a[3]);
                if (x.is(Val::CONST) && y.is(Val::CONST) && z.is(Val::CONST)) {
                    define(a[0].name, Val::Const(x.v * y.v + z.v), *out);
                    continue;
                }
                if (x.is(Val::CONST)) std::swap(x, y);
                int ra = reg_of(x, *out), rc = reg_of(z, *out);
                int rb = y.is(Val::CONST) ? VZERO : reg_of(y, *out);
                bool uni = isuni(ra) && isuni(rc) && (y.is(Val::CONST) || isuni(rb));
                int r = dest_for(a[0].name, uni);
                KInstr i = mk(SM_MAD);
                i.rd = r; i.ra = ra; i.rb = rb; i.rc = rc; i.useimm = y.is(Val::CONST); i.imm = y.is(Val::CONST) ? (uint32_t)y.v : 0;
                out->push_back(i);
                vals[a[0].name] = Val::Reg(r);
            } else if (op == "mul.lo.s32") {
                Val x = val(a[1]), y = val(a[2]);
                if (x.is(Val::CONST) && y.is(Val::CONST)) {
                    define(a[0].name, Val::Const(x.v * y.v), *out);
                    continue;
                }
                if (x.is(Val::CONST)) std::swap(x, y);
                int ra = reg_of(x, *out);
                int rb = y.is(Val::CONST) ? VZERO : reg_of(y, *out);
                bool uni = isuni(ra) && (y.is(Val::CONST) || isuni(rb));
                int r = dest_for(a[0].name, uni);
                KInstr i = mk(SM_MAD);
                i.rd = r; i.ra = ra; i.rb = rb; i.rc = VZERO; i.useimm = y.is(Val::CONST); i.imm = y.is(Val::CONST) ? (uint32_t)y.v : 0;
                out->push_back(i);
                vals[a[0].name] = Val::Reg(r);
            } else if (op != "or.pred" && op != "and.pred" && (base_op == "and" || base_op == "or" || base_op == "xor" || base_op == "shl" || base_op == "shr" || base_op == "not")
                       && ins.parts.size() > 1 && (ins.parts[1] == "b32" || ins.parts[1] == "u32" || ins.parts[1] == "s32")) {
                std::vector<Val> xs;
                for (size_t q = 1; q < a.size(); q++) xs.push_back(val(a[q]));
                bool allc = true;
                for (const Val &x : xs) allc = allc && x.is(Val::CONST);
                if (!allc) {
                    lower_bits(base_op, ins.parts[1], a[0].name, xs, *out);
                    continue;
                }
                int64_t v;
                if (base_op == "and") v = xs[0].v & xs[1].v;
                else if (base_op == "or") v = xs[0].v | xs[1].v;
                else if (base_op == "xor") v = xs[0].v ^ xs[1].v;
                else if (base_op == "not") v = ~xs[0].v;
                else if (base_op == "shl") v = xs[1].v < 32 ? xs[0].v << xs[1].v : 0;
                else {
                    int64_t sv = ins.parts[1] == "s32" && (xs[0].v >> 31 & 1) ? xs[0].v - ((int64_t)1 << 32) : xs[0].v;
                    v = sv >> std::min<int64_t>(xs[1].v, 32);
                }
                define(a[0].name, Val::Const(v), *out);
            } else if ((base_op == "max" || base_op == "min" || base_op == "sub") && ins.parts.size() > 1 && (ins.parts[1] == "s32" || ins.parts[1] == "u32")) {
                Val x = val(a[1]), y = val(a[2]);
                bool sgn = ins.parts[1] == "s32";
                if (x.is(Val::CONST) && y.is(Val::CONST)) {
                    int64_t v;
                    if (base_op == "sub") v = x.v - y.v;
                    else {
                        bool lt = compare_const(SM_CMP_LT, (uint32_t)x.v, (uint32_t)y.v, sgn);
                        v = base_op == "max" ? (lt ? y.v : x.v) : (lt ? x.v : y.v);
                    }
                    define(a[0].name, Val::Const(v), *out);
                    continue;
                }
                if (base_op == "sub" && y.is(Val::CONST)) {
                    int ra = reg_of(x, *out);
                    int r = dest_for(a[0].name, isuni(ra));
                    out->push_back(add_imm(r, ra, (uint32_t)(-y.v)));
                    vals[a[0].name] = Val::Reg(r);
                    continue;
                }
                if (base_op != "sub" && x.is(Val::CONST)) std::swap(x, y);
                int ra = reg_of(x, *out);
                int rb = y.is(Val::CONST) ? VZERO : reg_of(y, *out);
                bool uni = isuni(ra) && (y.is(Val::CONST) || isuni(rb));
                int r = dest_for(a[0].name, uni);
                KInstr i = mk(base_op == "max" ? SM_MAX : base_op == "min" ? SM_MIN : SM_SUB);
                i.rd = r; i.ra = ra; i.rb = rb; i.useimm = y.is(Val::CONST); i.imm = y.is(Val::CONST) ? (uint32_t)y.v : 0; i.sgn = sgn;
                out->push_back(i);
                vals[a[0].name] = Val::Reg(r);
            } else if (op == "div.u32" || op == "div.s32" || op == "rem.u32" || op == "rem.s32") {
                div_rem(op, a[0].name, val(a[1]), val(a[2]), *out);
            } else if (op == "neg.s32") {
                Val x = val(a[1]);
                if (x.is(Val::CONST)) {
                    define(a[0].name, Val::Const(-x.v), *out);
                    continue;
                }
                int rb = reg_of(x, *out);
                int r = dest_for(a[0].name, isuni(rb));
                KInstr i = mk(SM_SUB);
                i.rd = r; i.ra = VZERO; i.rb = rb;
                out->push_back(i);
                vals[a[0].name] = Val::Reg(r);
            } else if (op == "abs.s32") {
                Val x = val(a[1]);
                if (x.is(Val::CONST)) {
                    int32_t v = (int32_t)(uint32_t)x.v;
                    define(a[0].name, Val::Const(v < 0 ? -(int64_t)v : v), *out);
                    continue;
                }
                int ra = reg_of(x, *out);
                int neg = fresh(isuni(ra));
                KInstr s = mk(SM_SUB);
                s.rd = neg; s.ra = VZERO; s.rb = ra;
                out->push_back(s);
                int r = dest_for(a[0].name, isuni(ra));
                KInstr mx = mk(SM_MAX);
                mx.rd = r; mx.ra = ra; mx.rb = neg; mx.sgn = true;
                out->push_back(mx);
                vals[a[0].name] = Val::Reg(r);
            } else if (op == "mul.hi.u32" || op == "mul.hi.s32") {
                Val x = val(a[1]), y = val(a[2]);
                bool sgn = op == "mul.hi.s32";
                if (x.is(Val::CONST) && y.is(Val::CONST)) {
                    int64_t sx = sgn && (x.v >> 31 & 1) ? x.v - ((int64_t)1 << 32) : x.v, sy = sgn && (y.v >> 31 & 1) ? y.v - ((int64_t)1 << 32) : y.v;
                    define(a[0].name, Val::Const((sx * sy) >> 32), *out);
                    continue;
                }
                if (x.is(Val::CONST)) std::swap(x, y);
                vals[a[0].name] = Val::Reg(mul_hi(x, y, sgn, a[0].name, *out));
            } else if (op == "selp.b32" || op == "selp.s32" || op == "selp.u32") {
                Val x = val(a[1]), y = val(a[2]);
                if (!vals.count(a[3].name)) throw Unsupported("selp on an undefined predicate");
                Val p = vals[a[3].name];
                if (p.is(Val::KNOWN)) {
                    define(a[0].name, p.known ? x : y, *out);
                    continue;
                }
                if (!p.is(Val::PRED)) throw Unsupported("selp on a predicate of kind " + std::to_string(p.k));
                emit_setp(p, SEL_P, *out);
                int ra = reg_of(x, *out);
                int rb = y.is(Val::CONST) ? VZERO : reg_of(y, *out);
                bool uni = isuni(ra) && (y.is(Val::CONST) || isuni(rb)) && val_uni(p);
                int r = dest_for(a[0].name, uni);
                KInstr i = mk(SM_SELP);
                i.rd = r; i.ra = ra; i.rb = rb; i.useimm = y.is(Val::CONST); i.imm = y.is(Val::CONST) ? (uint32_t)y.v : 0; i.guard = SEL_P;
                out->push_back(i);
                vals[a[0].name] = Val::Reg(r);
            } else if (op == "or.pred" || op == "and.pred") {
                Val p = vals.at(a[1].name), q = vals.at(a[2].name);
                if (p.is(Val::KNOWN) || q.is(Val::KNOWN)) {
                    Val kp = p.is(Val::KNOWN) ? p : q, other = p.is(Val::KNOWN) ? q : p;
                    bool absorb = op == "or.pred" ? kp.known : !kp.known;
                    vals[a[0].name] = absorb ? Val::Known(kp.known) : other;
                    continue;
                }
                if (!(p.is(Val::PRED) && q.is(Val::PRED))) throw Unsupported(op + " of non-predicates");
                bool same = p.cmp == q.cmp && p.sgn == q.sgn && p.rhs_const == q.rhs_const && p.rhs_v == q.rhs_v && p.rhs_r == q.rhs_r
                            && p.cmp != SM_CMP_EQ && p.cmp != SM_CMP_NE;
                if (!same) {
                    // each as a 0 / 1 register, and / or, then != 0
                    std::vector<int> bits;
                    for (const Val &x : {p, q}) {
                        emit_setp(x, SEL_P, *out);
                        int one = reg_of(Val::Const(1), *out);
                        int b = fresh(val_uni(x));
                        KInstr sel = mk(SM_SELP);
                        sel.rd = b; sel.ra = one; sel.useimm = true; sel.imm = 0; sel.guard = SEL_P;
                        out->push_back(sel);
                        bits.push_back(b);
                    }
                    int r = fresh(isuni(bits[0]) && isuni(bits[1]));
                    KInstr i = mk(op == "or.pred" ? SM_OR : SM_AND);
                    i.rd = r; i.ra = bits[0]; i.rb = bits[1];
                    out->push_back(i);
                    Val np;
                    np.k = Val::PRED; np.cmp = SM_CMP_NE; np.sgn = false; np.ra = r; np.rhs_const = true; np.rhs_v = 0;
                    vals[a[0].name] = np;
                    continue;
                }
                // (a >= n) or (b >= n) is max(a, b) >= n; and is min; the other way round for <
                bool upper = p.cmp == SM_CMP_GE || p.cmp == SM_CMP_GT;
                int kind = (upper == (op == "or.pred")) ? SM_MAX : SM_MIN;
                int r = fresh(isuni(p.ra) && isuni(q.ra));
                KInstr i = mk(kind);
                i.rd = r; i.ra = p.ra; i.rb = q.ra; i.sgn = p.sgn;
                out->push_back(i);
                Val np = p;
                np.ra = r;
                vals[a[0].name] = np;
            } else if (op == "add.s64" || (op == "add.s32" && [&] {
                           Val x = val(a[1]), y = val(a[2]);
                           return x.is(Val::PTR) || (x.is(Val::ADDR) && x.ptr) || (x.is(Val::REG) && x.ptr) || y.is(Val::PTR) || (y.is(Val::ADDR) && y.ptr)
                                  || (y.is(Val::REG) && y.ptr);
                       }())) {
                Val x = val(a[1]), y = val(a[2]);
                if ((x.is(Val::SCALED) || x.is(Val::CONST)) && !(y.is(Val::SCALED) || y.is(Val::CONST))) std::swap(x, y);
                if (x.is(Val::ADDR) && !x.ptr) x = Val::Reg(reg_of(x, *out));  // a 32-bit sum used as a byte count: materialize it
                if (y.is(Val::ADDR) && !y.ptr) y = Val::Reg(reg_of(y, *out));
                Val v;
                if (x.is(Val::PTR) && y.is(Val::SCALED)) v = Val::PAddr(words_reg(y, *out), x.v);
                else if (x.is(Val::ADDR) && y.is(Val::SCALED)) v = Val::PAddr(words_reg(y, *out, x.r), x.v);
                else if (x.is(Val::PTR) && y.is(Val::CONST)) v = Val::Ptr(x.v + whole_words(y));
                else if (x.is(Val::ADDR) && y.is(Val::CONST)) v = Val::PAddr(x.r, x.v + whole_words(y));
                else if (x.is(Val::REG) && x.ptr && y.is(Val::CONST)) {
                    int r = dest_for(a[0].name, isuni(x.r));
                    out->push_back(add_imm(r, x.r, (uint32_t)whole_words(y)));
                    v = Val::Reg(r, true);
                } else if (x.is(Val::REG) && x.ptr && y.is(Val::SCALED)) {
                    int w = words_reg(y, *out);
                    int r = dest_for(a[0].name, isuni(x.r) && isuni(w));
                    out->push_back(add_rr(r, x.r, w));
                    v = Val::Reg(r, true);
                } else if (x.is(Val::REG) && x.ptr && y.is(Val::WIDE)) {
                    int w = reg_of(*y.inner, *out);
                    // a 64-bit add of a pointer and a widened word index... only whole-word byte counts: y is bytes
                    throw Unsupported("pointer + a 64-bit register of bytes (scale it by 4 first)");
                    (void)w;
                } else {
                    throw Unsupported(op + " of kinds " + std::to_string(x.k) + ", " + std::to_string(y.k));
                }
                define(a[0].name, v, *out, true);
            } else if (op == "add.s32") {
                Val x = val(a[1]), y = val(a[2]);
                if (x.is(Val::CONST) && y.is(Val::CONST)) {
                    define(a[0].name, Val::Const(x.v + y.v), *out);
                    continue;
                }
                if (x.is(Val::CONST)) std::swap(x, y);
                // register + constant stays symbolic when it is defined once: a memory or block instruction takes
                // it as [register + immediate], anything else materializes the add where it is used
                if (y.is(Val::CONST) && !multi(a[0].name) && (x.is(Val::REG) || x.is(Val::ADDR))) {
                    vals[a[0].name] = x.is(Val::REG) ? Val::Addr(x.r, (int32_t)(uint32_t)y.v) : Val::Addr(x.r, (int32_t)(uint32_t)(x.v + y.v));
                    continue;
                }
                int ra = reg_of(x, *out);
                int rb = y.is(Val::CONST) ? -1 : reg_of(y, *out);
                bool uni = isuni(ra) && (rb < 0 || isuni(rb));
                int r = dest_for(a[0].name, uni);
                if (rb < 0) out->push_back(add_imm(r, ra, (uint32_t)y.v));
                else out->push_back(add_rr(r, ra, rb));
                vals[a[0].name] = Val::Reg(r);
            } else if (op == "cvt.s64.s32" || op == "cvt.u64.u32") {
                Val x = val(a[1]);
                if (x.is(Val::GATHER) || x.is(Val::SCALED)) x = Val::Reg(reg_of(x, *out));
                if (!(x.is(Val::REG) || x.is(Val::CONST))) throw Unsupported(op + " of kind " + std::to_string(x.k));
                bool is_home = false;
                if (x.is(Val::REG))
                    for (const auto &p : mreg)
                        if (p.second == x.r) is_home = true;
                if (x.is(Val::REG) && is_home && !multi(a[0].name)) {
                    int t = fresh(isuni(x.r));
                    out->push_back(add_imm(t, x.r, 0));
                    x = Val::Reg(t);
                }
                define(a[0].name, Val::Wide(x), *out);
            } else if (op == "cvt.u32.u64") {
                Val x = as_wide(val(a[1]));
                if (!x.is(Val::WIDE)) throw Unsupported(op + " of kind " + std::to_string(x.k));
                define(a[0].name, *x.inner, *out);
            } else if (op == "and.b64") {
                Val x = val(a[1]), y = val(a[2]);
                if (x.is(Val::CONST)) std::swap(x, y);
                if (!(x.is(Val::WIDE) && y.is(Val::CONST) && y.v % 4 == 0)) throw Unsupported("and.b64 of these operands");
                if (x.inner->is(Val::CONST)) {
                    vals[a[0].name] = Val::Const(x.inner->v & y.v);
                    continue;
                }
                int t = fresh(isuni(x.inner->r)), wd = fresh(isuni(x.inner->r));
                KInstr i = mk(SM_AND);
                i.rd = t; i.ra = x.inner->r; i.useimm = true; i.imm = (uint32_t)y.v;
                out->push_back(i);
                KInstr s = mk(SM_SHR);
                s.rd = wd; s.ra = t; s.sh = 2;
                out->push_back(s);
                vals[a[0].name] = Val::Scaled(wd, 1);
            } else if (op == "shl.b64") {
                Val x = as_wide(val(a[1])), y = val(a[2]);
                if (!(x.is(Val::WIDE) && y.is(Val::CONST) && y.v >= 2 && y.v < 32)) throw Unsupported("shl.b64 of these operands");
                if (x.inner->is(Val::CONST)) vals[a[0].name] = Val::RawConst(x.inner->v << y.v);
                else vals[a[0].name] = scaled(x.inner->r, (int64_t)1 << (y.v - 2), *out);
            } else if (op == "mul.wide.s32" || op == "mul.wide.u32") {
                Val x = val(a[1]), y = val(a[2]);
                if (!(y.is(Val::CONST) && y.v % 4 == 0 && y.v)) throw Unsupported("mul.wide by a value that is not a multiple of 4");
                if (x.is(Val::CONST)) {
                    define(a[0].name, Val::RawConst(y.v * x.v), *out);
                    continue;
                }
                if (!x.is(Val::REG)) x = Val::Reg(reg_of(x, *out));
                vals[a[0].name] = scaled(x.r, y.v / 4, *out);
            } else if (op.compare(0, 5, "setp.") == 0) {
                if (ins.parts.size() != 3) throw Unsupported(op);
                int cmp = cmp_index(ins.parts[1]);
                const std::string &ty = ins.parts[2];
                if (ty != "s32" && ty != "u32" && ty != "b32") throw Unsupported(op);
                Val x = val(a[1], true), y = val(a[2], true);
                bool sgn = ty == "s32";
                if (x.is(Val::CONST) && y.is(Val::CONST)) {
                    vals[a[0].name] = Val::Known(compare_const(cmp, (uint32_t)x.v, (uint32_t)y.v, sgn));
                    continue;
                }
                if (x.is(Val::CONST)) {
                    std::swap(x, y);
                    cmp = mirror(cmp);
                }
                Val p;
                p.k = Val::PRED; p.cmp = cmp; p.sgn = sgn; p.ra = reg_of(x, *out);
                // "tid.x < 32 m" and its kin compare a per-lane register, but every lane agrees: the analysis knows
                p.puni = uniform.count(a[0].name) != 0;
                if (y.is(Val::CONST)) {
                    p.rhs_const = true;
                    p.rhs_v = y.v;
                } else {
                    p.rhs_r = reg_of(y, *out);
                }
                vals[a[0].name] = p;
            } else if (base_op == "bra") {
                int t = bb_at.at(k.labels.at(a[0].name));
                if (ins.guard.empty()) {
                    terms[(size_t)gi].kind = SM_BR_JUMP;
                    terms[(size_t)gi].target = t;
                    branched = true;
                    continue;
                }
                Val p = vals.at(ins.guard);
                if (p.is(Val::KNOWN)) {
                    if (p.known != ins.neg) {
                        terms[(size_t)gi].kind = SM_BR_JUMP;
                        terms[(size_t)gi].target = t;
                        branched = true;
                    }
                    continue;
                }
                if (ret_blocks.count(t) && !branched && !uniform.count(ins.guard)) {
                    if (ins.neg || !exit_pred.empty()) throw Unsupported("more than one early exit, or @!p bra to ret");
                    exit_pred.push_back(p);
                    emit_setp(p, EXIT_P, *out);
                    exit_at[gi] = (int)out->size() - 1;
                    continue;
                }
                if (!uniform.count(ins.guard)) throw Unsupported("divergent branch on " + ins.guard);
                terms[(size_t)gi].kind = SM_BR_COND;
                terms[(size_t)gi].pred = p;
                terms[(size_t)gi].is_pred = true;
                terms[(size_t)gi].neg = ins.neg;
                terms[(size_t)gi].target = t;
                branched = true;
            } else if ((base_op == "bar" || base_op == "barrier") && ins.parts.size() > 1 && ins.parts[1] == "sync" && a[0].kind == ptx::Operand::IMM && a[0].value == 0) {
                barrier(gi, out);
            } else if (base_op == "ld" && ins.parts.size() > 1 && (ins.parts[1] == "global" || ins.parts[1] == "shared" || (ins.parts[1] == "volatile" && ins.parts.size() > 2 && (ins.parts[2] == "global" || ins.parts[2] == "shared")))
                       && ins.parts.back().substr(1) == "32") {
                auto [ra, w] = word_addr(a[1]);
                int r = dest_for(a[0].name, isuni(ra));
                KInstr l = mk(SM_LD);
                l.rd = r; l.ra = ra; l.imm = (uint32_t)w; l.bank = SM_BANK_M;
                out->push_back(l);
                vals[a[0].name] = Val::Reg(r);
            } else if (op == "shf.l.wrap.b32" || op == "shf.r.wrap.b32") {
                // the funnel shift: (b:a) shifted by c & 31, the high word (left) or the low word (right)
                Val xa = val(a[1]), xb = val(a[2]), xc = val(a[3]);
                int ra = reg_of(xa, *out), rb = reg_of(xb, *out);
                bool left = op[4] == 'l';
                bool uni = isuni(ra) && isuni(rb) && val_uni(xc);
                int c, c32;
                if (xc.is(Val::CONST)) {
                    c = reg_of(Val::Const(xc.v & 31), *out);
                    c32 = reg_of(Val::Const(32 - (xc.v & 31)), *out);
                } else {
                    int cr = reg_of(xc, *out);
                    if (!isuni(cr)) throw Unsupported("a funnel shift by a per-lane amount");
                    c = fresh(true);
                    KInstr an = mk(SM_AND);
                    an.rd = c; an.ra = cr; an.useimm = true; an.imm = 31;
                    out->push_back(an);
                    c32 = fresh(true);
                    KInstr sb = mk(SM_SUB);
                    sb.rd = c32; sb.ra = reg_of(Val::Const(32), *out); sb.rb = c;
                    out->push_back(sb);
                }
                int hi_part = fresh(uni), lo_part = fresh(uni);
                KInstr s1 = mk(left ? SM_SHL : SM_SHR), s2 = mk(left ? SM_SHR : SM_SHL);
                s1.rd = hi_part; s1.ra = left ? rb : ra; s1.rc = c; s1.sh = 63;
                s2.rd = lo_part; s2.ra = left ? ra : rb; s2.rc = c32; s2.sh = 63;
                out->push_back(s1);
                out->push_back(s2);
                int r = dest_for(a[0].name, uni);
                KInstr o = mk(SM_OR);
                o.rd = r; o.ra = hi_part; o.rb = lo_part;
                out->push_back(o);
                vals[a[0].name] = Val::Reg(r);
            } else if (base_op == "st" && ins.parts.size() > 1 && (ins.parts[1] == "global" || ins.parts[1] == "shared" || (ins.parts[1] == "volatile" && ins.parts.size() > 2 && (ins.parts[2] == "global" || ins.parts[2] == "shared")))
                       && ins.parts.back().substr(1) == "32") {
                auto [ra, w] = word_addr(a[0]);
                int src = reg_of(val(a[1]), *out);
                KInstr s = mk(SM_STV);
                s.ra = ra; s.rb = src; s.imm = (uint32_t)w; s.bank = SM_BANK_M;
                side_guard(s, gi, *out);
                out->push_back(s);
            } else if (base_op == "q8dot" || base_op == "q8dota" || base_op == "sdot") {
                // q8dot d, acc, wblk, ablk (q8dota: the lane block from A); sdot d, acc, addr, k8. The block or address may
                // be register + constant (the immediate); then the second operand must be a register, since the immediate
                // is taken
                Val av = val(a[2]), bv = val(a[3]);
                int rc = reg_of(val(a[1]), *out);
                auto [ra, off] = addr_of(av, *out);
                int rb;
                bool useimm = false;
                uint32_t imm = (uint32_t)off;
                if (base_op == "sdot") {
                    rb = reg_of(bv, *out);  // sdot's k8: a register, the immediate is the address offset alone
                } else if (bv.is(Val::CONST) && off == 0) {
                    rb = VZERO;
                    useimm = true;
                    imm = (uint32_t)bv.v;
                } else if (bv.is(Val::CONST)) {
                    // one immediate for both blocks: the uniform register holds the constant less the lane side's offset
                    rb = reg_of(Val::Const(bv.v - off), *out);
                } else {
                    auto [rb0, offb] = addr_of(bv, *out);
                    if (offb == off) {
                        rb = rb0;
                    } else {
                        rb = fresh(isuni(rb0));
                        out->push_back(add_imm(rb, rb0, (uint32_t)(offb - off)));
                    }
                }
                bool uni = isuni(ra) && (useimm || isuni(rb)) && isuni(rc);
                int r = dest_for(a[0].name, uni);
                KInstr i = mk(base_op == "sdot" ? SM_SDOT : SM_Q8DOT);
                i.rd = r; i.ra = ra; i.rb = rb; i.rc = rc; i.useimm = useimm; i.imm = imm; i.bank = base_op == "q8dota" ? 1 : 0;
                out->push_back(i);
                vals[a[0].name] = Val::Reg(r);
            } else if (base_op == "sup") {
                // sup acc, addr, delta, kreal: acc = max(acc, |the state element after the update|); the first operand is
                // read and written (a "+r" asm operand), so the machine's rd holds the running value
                int acc = reg_of(val(a[0]), *out);
                KInstr i = mk(SM_SUP);
                auto [ra, off] = addr_of(val(a[1]), *out);
                i.ra = ra;
                i.imm = (uint32_t)off;
                i.rb = reg_of(val(a[2]), *out);
                i.rc = reg_of(val(a[3]), *out);
                int r = dest_for(a[0].name, false);
                if (r != acc) out->push_back(add_imm(r, acc, 0));
                i.rd = r;
                side_guard(i, gi, *out);
                out->push_back(i);
                vals[a[0].name] = Val::Reg(r);
            } else if (base_op == "scfg") {
                KInstr i = mk(SM_SCFG);
                i.ra = reg_of(val(a[0]), *out);
                out->push_back(i);
            } else if (base_op == "wsum" || base_op == "wmax") {
                int ra = reg_of(val(a[1]), *out);
                int r = dest_for(a[0].name, true);
                KInstr i = mk(base_op == "wsum" ? SM_WSUM : SM_WMAX);
                i.rd = r; i.ra = ra;
                side_guard(i, gi, *out);  // lanes past the exit do not count
                out->push_back(i);
                vals[a[0].name] = Val::Reg(r);
            } else if (base_op == "udiv") {
                // udiv d, lo, div, hi
                int ra = reg_of(val(a[1]), *out), rb = reg_of(val(a[2]), *out), rc = reg_of(val(a[3]), *out);
                if (!(isuni(ra) && isuni(rb) && isuni(rc))) throw Unsupported("udiv of per-lane values");
                int r = dest_for(a[0].name, true);
                KInstr i = mk(SM_UDIV);
                i.rd = r; i.ra = ra; i.rb = rb; i.rc = rc;
                out->push_back(i);
                vals[a[0].name] = Val::Reg(r);
            } else if (base_op == "usqrt") {
                int ra = reg_of(val(a[1]), *out), rc = reg_of(val(a[2]), *out);
                if (!(isuni(ra) && isuni(rc))) throw Unsupported("usqrt of per-lane values");
                int r = dest_for(a[0].name, true);
                KInstr i = mk(SM_USQRT);
                i.rd = r; i.ra = ra; i.rc = rc;
                out->push_back(i);
                vals[a[0].name] = Val::Reg(r);
            } else if (base_op == "mulsh") {
                // mulsh d, a, b, sh: sh a constant 0..62, or a uniform register (signed; sh field 63)
                Val x = val(a[1]), y = val(a[2]), s = val(a[3]);
                int ra = reg_of(x, *out);
                int rb = y.is(Val::CONST) ? VZERO : reg_of(y, *out);
                int rc = VZERO, sh;
                if (s.is(Val::CONST) && (int32_t)(uint32_t)s.v >= 0 && (int32_t)(uint32_t)s.v <= 62) {
                    sh = (int)s.v;
                } else {  // a run-time shift, or a constant outside the field (negative: a left shift): through a uniform register
                    rc = reg_of(s, *out);
                    if (!isuni(rc)) throw Unsupported("mulsh by a per-lane shift");
                    sh = 63;
                }
                bool uni = isuni(ra) && (y.is(Val::CONST) || isuni(rb));
                int r = dest_for(a[0].name, uni);
                KInstr i = mk(SM_MULSH);
                i.rd = r; i.ra = ra; i.rb = rb; i.rc = rc; i.useimm = y.is(Val::CONST); i.imm = y.is(Val::CONST) ? (uint32_t)y.v : 0; i.sh = sh;
                out->push_back(i);
                vals[a[0].name] = Val::Reg(r);
            } else if (op == "clz.b32") {
                Val x = val(a[1]);
                if (x.is(Val::CONST)) {
                    define(a[0].name, Val::Const(x.v ? __builtin_clz((uint32_t)x.v) : 32), *out);
                    continue;
                }
                int ra = reg_of(x, *out);
                int r = dest_for(a[0].name, isuni(ra));
                KInstr i = mk(SM_CLZ);
                i.rd = r; i.ra = ra;
                out->push_back(i);
                vals[a[0].name] = Val::Reg(r);
            } else if (base_op == "lda8" || base_op == "lda32" || base_op == "lds" || base_op == "ldc" || base_op == "ldw8" || base_op == "ldw32" || base_op == "ldm") {
                int bank = base_op == "lda8" ? SM_BANK_A8 : base_op == "lda32" ? SM_BANK_A32 : base_op == "lds" ? SM_BANK_S : base_op == "ldc" ? SM_BANK_C
                           : base_op == "ldw8" ? SM_BANK_W8 : base_op == "ldw32" ? SM_BANK_W32 : SM_BANK_M;
                auto [ra, off] = addr_of(val(a[1]), *out);
                int r = dest_for(a[0].name, isuni(ra));
                KInstr l = mk(SM_LD);
                l.rd = r; l.ra = ra; l.imm = (uint32_t)off; l.bank = bank;
                out->push_back(l);
                vals[a[0].name] = Val::Reg(r);
            } else if (base_op == "stva8" || base_op == "stva32" || base_op == "stvs" || base_op == "stvc" || base_op == "stvm") {
                int bank = base_op == "stva8" ? SM_BANK_A8 : base_op == "stva32" ? SM_BANK_A32 : base_op == "stvs" ? SM_BANK_S : base_op == "stvc" ? SM_BANK_C : SM_BANK_M;
                auto [ra, off] = addr_of(val(a[0]), *out);
                int src = reg_of(val(a[1]), *out);
                KInstr s = mk(SM_STV);
                s.ra = ra; s.rb = src; s.imm = (uint32_t)off; s.bank = bank;
                side_guard(s, gi, *out);
                out->push_back(s);
            } else {
                throw Unsupported(op);
            }
        }
    }

    // the setp of a cond branch goes last in its group, so it is in the final machine block
    for (size_t gi = 0; gi < terms.size(); gi++)
        if (terms[gi].kind == SM_BR_COND) emit_setp(terms[gi].pred, BRANCH_P, code[gi]);
    // branch targets as groups
    for (Term &t : terms)
        if (t.kind == SM_BR_JUMP || t.kind == SM_BR_COND) t.target = group_of[(size_t)t.target];
    const int ng = (int)code.size();

    // liveness over the group CFG, on virtual registers
    auto succs = [&](int gi) {
        std::vector<int> s;
        const Term &t = terms[(size_t)gi];
        if (t.kind == SM_BR_NEXT || t.kind == SM_BR_BAR || t.kind == SM_BR_COND) {
            if (gi + 1 < ng) s.push_back(gi + 1);
        }
        if (t.kind == SM_BR_JUMP || t.kind == SM_BR_COND) s.push_back(t.target);
        return s;
    };
    std::vector<std::set<int>> use(ng), dfn(ng), live_in(ng), live_out(ng);
    for (int gi = 0; gi < ng; gi++) {
        for (const KInstr &ins : code[(size_t)gi]) {
            auto [rd, srcs] = ins.rw();
            for (int s : srcs)
                if (!dfn[(size_t)gi].count(s)) use[(size_t)gi].insert(s);
            if (rd != -1) dfn[(size_t)gi].insert(rd);
        }
    }
    bool changed = true;
    while (changed) {
        changed = false;
        for (int gi = ng - 1; gi >= 0; gi--) {
            std::set<int> lo;
            for (int s : succs(gi)) lo.insert(live_in[(size_t)s].begin(), live_in[(size_t)s].end());
            std::set<int> li = use[(size_t)gi];
            for (int x : lo)
                if (!dfn[(size_t)gi].count(x)) li.insert(x);
            if (lo != live_out[(size_t)gi] || li != live_in[(size_t)gi]) {
                live_out[(size_t)gi] = lo;
                live_in[(size_t)gi] = li;
                changed = true;
            }
        }
    }

    // register allocation: color the interference graph, separately for the lane and the uniform file
    std::map<int, std::set<int>> edges, reads_near;
    std::vector<int> order;
    for (int gi = 0; gi < ng; gi++) {
        std::set<int> read;
        for (const KInstr &ins : code[(size_t)gi])
            for (int x : ins.rw().second)
                if (x >= 0) read.insert(x);
        for (const KInstr &ins : code[(size_t)gi]) {
            int rd = ins.rw().first;
            if (rd >= 0)
                for (int x : read)
                    if (x != rd) reads_near[rd].insert(x);
        }
        std::set<int> live;
        for (int r : live_out[(size_t)gi])
            if (r >= 0) live.insert(r);
        for (int i = (int)code[(size_t)gi].size() - 1; i >= 0; i--) {
            auto [rd, srcs] = code[(size_t)gi][(size_t)i].rw();
            if (rd >= 0) {
                for (int x : live)
                    if (x != rd) {
                        edges[rd].insert(x);
                        edges[x].insert(rd);
                    }
                live.erase(rd);
            }
            for (int x : srcs)
                if (x >= 0) live.insert(x);
        }
        for (const KInstr &ins : code[(size_t)gi]) {
            auto note = [&](int r) {
                if (r < 0 || r == VZERO || r == VARGS) return;
                if (std::find(order.begin(), order.end(), r) == order.end()) order.push_back(r);
            };
            int rd = ins.rw().first;
            note(rd);
            for (int x : ins.rw().second) note(x);
        }
    }
    std::map<int, int> color;
    color[VZERO] = SM_ZERO;
    color[VARGS] = SM_UARGS;
    int lane_used = 0, uni_used = 0;
    for (int r : order) {
        if (color.count(r)) continue;
        bool uni = isuni(r);
        std::set<int> taken;
        for (int x : edges[r])
            if (color.count(x) && isuni(x) == uni) taken.insert(color[x]);
        std::vector<int> free;
        int limit = uni ? SM_UARGS : SM_ZERO;
        for (int c = 0; c < limit; c++)
            if (!taken.count(c)) free.push_back(c);
        if (free.empty()) throw Unsupported(std::string(uni ? "needs more than 14 uniform" : "needs more than 15 lane") + " registers");
        std::set<int> avoid;
        for (int x : reads_near[r])
            if (color.count(x) && isuni(x) == uni) avoid.insert(color[x]);
        int pick = free[0];
        for (int c : free)
            if (!avoid.count(c)) {
                pick = c;
                break;
            }
        color[r] = pick;
        if (uni) uni_used = std::max(uni_used, pick + 1);
        else lane_used = std::max(lane_used, pick + 1);
    }
    // the physical code, and per instruction the operand kinds
    struct PFlags {
        uint8_t ua, ub, uc, ud;
    };
    auto phys = [&](int r) { return r < 0 ? 0 : color.at(r); };
    std::vector<std::vector<KInstr>> pcode(ng);
    std::vector<std::vector<PFlags>> pflags(ng);
    for (int gi = 0; gi < ng; gi++) {
        for (const KInstr &ins : code[(size_t)gi]) {
            KInstr p = ins;
            PFlags f{0, 0, 0, 0};
            auto [rd, srcs] = ins.rw();
            // which fields the op uses
            bool use_ra = true, use_rb = !ins.useimm, use_rc = false, use_rd = rd >= 0;
            switch (ins.op) {
            case SM_TID: case SM_SREG: use_ra = use_rb = false; break;
            case SM_SHL: case SM_SHR: use_rb = false; use_rc = ins.sh == 63; break;  /* the amount register of a register shift */
            case SM_CLZ: case SM_LD: case SM_WSUM: case SM_WMAX: use_rb = false; break;
            case SM_MAD: case SM_Q8DOT: case SM_SDOT: case SM_UDIV: use_rc = true; break;
            case SM_MULSH: use_rc = ins.sh == 63; break;
            case SM_USQRT: use_rb = false; use_rc = true; break;
            case SM_SETP: use_rd = false; break;
            case SM_STV: use_rd = false; break;
            case SM_SUP: use_rc = true; break;
            case SM_SCFG: use_rd = false; use_rb = false; break;
            default: break;
            }
            p.rd = use_rd ? phys(ins.rd) : 0;
            p.ra = use_ra ? phys(ins.ra) : 0;
            p.rb = use_rb ? phys(ins.rb) : 0;
            p.rc = use_rc ? phys(ins.rc) : 0;
            f.ua = use_ra && isuni(ins.ra);
            f.ub = ins.useimm ? SM_SRC_IMM : (use_rb && isuni(ins.rb) ? SM_SRC_UNIFORM : SM_SRC_LANE);
            f.uc = use_rc && isuni(ins.rc);
            if (ins.op == SM_STV) f.ud = isuni(ins.ra) && isuni(ins.rb);
            else if (ins.op == SM_SETP || ins.op == SM_SCFG || ins.op == SM_NOP) f.ud = 0;
            else f.ud = use_rd && isuni(ins.rd);
            // a uniform result needs uniform operands (reductions and scalars take their own kinds)
            if (f.ud && ins.op != SM_WSUM && ins.op != SM_WMAX) {
                if ((use_ra && !f.ua) || (use_rb && f.ub == SM_SRC_LANE) || (use_rc && !f.uc)) throw Unsupported("a uniform result from a per-lane operand");
            }
            pcode[(size_t)gi].push_back(p);
            pflags[(size_t)gi].push_back(f);
        }
    }

    // machine blocks: each group split into stage-sized pieces; the flags follow the instructions
    Launch L;
    L.name = k.name;
    L.warps = (uint32_t)((int64_t)gx * gy * bx * by / SM_LANES);
    L.lwpb = bitlen(wpb) - 1;
    L.lane_regs = lane_used;
    L.uniform_regs = uni_used;
    std::vector<int> first;
    std::vector<std::vector<MBlock>> pieces(ng);
    std::vector<std::vector<std::vector<Launch::Flags>>> pieces_flags(ng);
    for (int gi = 0; gi < ng; gi++) {
        std::set<int> outs;
        for (int r : live_out[(size_t)gi])
            if (r >= 0 && color.count(r)) outs.insert(r);
        // split works on virtual registers (the live set is virtual); it keeps instruction order, so the flags map by position;
        // it needs to know which results are uniform (the late-result rules of a lane-serial record)
        for (size_t pi = 0; pi < code[(size_t)gi].size(); pi++) code[(size_t)gi][pi].ud = pflags[(size_t)gi][pi].ud != 0;
        std::vector<Block> bl;
        if (!code[(size_t)gi].empty()) bl = split(code[(size_t)gi], outs, opt.classes);
        size_t pos = 0;
        for (const Block &b : bl) {
            MBlock mb;
            std::vector<Launch::Flags> fl;
            for (const KInstr &ins : b.instrs) {
                if (ins.op == SM_NOP) {
                    KInstr nop;
                    mb.instrs.push_back(nop);
                    fl.push_back(Launch::Flags{});
                    continue;
                }
                mb.instrs.push_back(pcode[(size_t)gi][pos]);
                const PFlags &pf = pflags[(size_t)gi][pos];
                fl.push_back(Launch::Flags{pf.ua, pf.ub, pf.uc, pf.ud});
                pos++;
            }
            for (const auto &lv : b.live) mb.live.emplace_back(lv.first, lv.second >= 0 ? color.at(lv.second) : -1);
            mb.lwpb = L.lwpb;
            pieces[(size_t)gi].push_back(mb);
            pieces_flags[(size_t)gi].push_back(fl);
        }
        if (pos != pcode[(size_t)gi].size()) throw Unsupported("block formation lost instructions");
    }
    int idx = 0;
    for (int gi = 0; gi < ng; gi++) {
        first.push_back(idx);
        idx += std::max<int>((int)pieces[(size_t)gi].size(), 1);
    }
    for (int gi = 0; gi < ng; gi++) {
        if (pieces[(size_t)gi].empty()) {
            MBlock mb;
            mb.lwpb = L.lwpb;
            L.blocks.push_back(mb);
            L.flags.emplace_back();
        } else {
            for (size_t p = 0; p < pieces[(size_t)gi].size(); p++) {
                L.blocks.push_back(pieces[(size_t)gi][p]);
                L.flags.push_back(pieces_flags[(size_t)gi][p]);
            }
        }
        MBlock &last = L.blocks.back();
        const Term &t = terms[(size_t)gi];
        if (t.kind == SM_BR_BAR) last.branch = SM_BR_BAR;
        else if (t.kind == SM_BR_END) last.branch = SM_BR_END;
        else if (t.kind == SM_BR_JUMP) {
            last.branch = SM_BR_JUMP;
            last.target = first[(size_t)t.target];
        } else if (t.kind == SM_BR_COND) {
            int stage = -1;
            for (size_t s = 0; s < last.instrs.size(); s++)
                if (last.instrs[s].op == SM_SETP && last.instrs[s].pd == BRANCH_P) stage = (int)s;
            if (stage < 0) throw Unsupported("the branch's setp is not in the last block");
            last.branch = SM_BR_COND;
            last.bstage = stage;
            last.bneg = t.neg;
            last.target = first[(size_t)t.target];
        }
    }
    if (L.blocks.back().branch == SM_BR_NEXT) throw Unsupported("control falls off the end");
    // falling into an empty block that only ends the warp: end right here
    for (size_t b = 0; b + 1 < L.blocks.size(); b++) {
        MBlock &blk = L.blocks[b];
        const MBlock &nxt = L.blocks[b + 1];
        if (blk.branch == SM_BR_NEXT && nxt.instrs.empty() && nxt.branch == SM_BR_END) blk.branch = SM_BR_END;
    }
    for (const MBlock &b : L.blocks)
        for (const KInstr &i : b.instrs)
            if (i.op != SM_NOP) L.stats_instrs++;
    return L;
}

}  // namespace

Launch lower(const ptx::Kernel &k, int grid_x, int grid_y, int block_x, int block_y, const std::vector<ParamSpec> &params, const Options &opt) {
    Lowerer lw(k, opt);
    lw.params = params;
    lw.gx = grid_x;
    lw.gy = grid_y;
    lw.bx = block_x;
    lw.by = block_y;
    return lw.run();
}

void encode(const Launch &l, std::vector<SmBlock> &rom, uint32_t base) {
    if (base + l.blocks.size() > SM_NBLK) throw Unsupported("the ROM is full");
    if (rom.size() < base + l.blocks.size()) rom.resize(base + l.blocks.size());
    for (size_t b = 0; b < l.blocks.size(); b++) {
        const MBlock &mb = l.blocks[b];
        SmBlock sb;
        std::memset(&sb, 0, sizeof sb);
        for (size_t s = 0; s < mb.instrs.size() && s < SM_STAGES; s++) {
            const KInstr &ki = mb.instrs[s];
            SmInstr si;
            std::memset(&si, 0, sizeof si);
            si.op = (uint8_t)ki.op;
            si.valid = ki.op != SM_NOP;
            si.rd = (uint8_t)ki.rd;
            si.ra = (uint8_t)ki.ra;
            si.rb = (uint8_t)ki.rb;
            si.rc = (uint8_t)ki.rc;
            si.pd = (uint8_t)ki.pd;
            si.cmp = (uint8_t)ki.cmp;
            si.sgn = ki.sgn;
            si.imm = ki.imm;
            si.gvalid = ki.guard >= 0;
            si.g = (uint8_t)(ki.guard >= 0 ? ki.guard : 0);
            si.gneg = ki.gneg;
            si.bank = (uint8_t)ki.bank;
            si.sh = (uint8_t)ki.sh;
            const Launch::Flags &f = l.flags[b][s];
            si.ua = f.ua;
            si.ub = f.ub;
            si.uc = f.uc;
            si.ud = f.ud;
            sm_encode(&si, sb.rows[s]);
        }
        SmLive lv;
        std::memset(&lv, 0, sizeof lv);
        lv.nlive = (uint8_t)mb.live.size();
        for (size_t i = 0; i < mb.live.size() && i < SM_LMAX; i++) lv.src[i] = (uint8_t)mb.live[i].first;
        lv.branch = (uint8_t)mb.branch;
        lv.stage = (uint8_t)mb.bstage;
        lv.neg = mb.bneg;
        lv.any = mb.bany;
        lv.target = (uint16_t)(mb.branch == SM_BR_JUMP || mb.branch == SM_BR_COND ? base + (uint32_t)mb.target : 0);
        lv.lwpb = (uint8_t)mb.lwpb;
        sb.live = sm_live_encode(&lv);
        rom[base + b] = sb;
    }
}

std::string listing(const Launch &l) {
    std::ostringstream o;
    o << l.name << ": " << l.blocks.size() << " blocks, " << l.warps << " warps, wpb 2^" << l.lwpb << ", " << l.lane_regs << " lane + " << l.uniform_regs
      << " uniform registers, " << l.stats_instrs << " instructions\n";
    static const char *brn[] = {"next", "jump", "cond", "end", "bar"};
    for (size_t b = 0; b < l.blocks.size(); b++) {
        const MBlock &mb = l.blocks[b];
        o << "  block " << b << ":";
        for (size_t s = 0; s < mb.instrs.size(); s++) {
            const KInstr &i = mb.instrs[s];
            const Launch::Flags &f = l.flags[b][s];
            if (i.op == SM_NOP) {
                o << " nop;";
                continue;
            }
            o << " " << op_name(i.op);
            auto reg = [&](int r, bool uni) { return (uni ? "u" : "r") + std::to_string(r); };
            if (f.ud && i.op != SM_STV) o << " " << reg(i.rd, true);
            else if (i.op != SM_STV && i.op != SM_SETP && i.op != SM_SCFG) o << " " << reg(i.rd, false);
            if (i.op == SM_SETP) o << " p" << i.pd;
            if (i.op != SM_TID && i.op != SM_SREG) o << " " << reg(i.ra, f.ua);
            if (i.op == SM_LD || i.op == SM_STV || i.op == SM_Q8DOT || i.op == SM_SDOT || i.op == SM_SUP) o << "+" << i.imm;
            if (f.ub == SM_SRC_IMM) o << " #" << (int32_t)i.imm;
            else if (i.op != SM_SHL && i.op != SM_SHR && i.op != SM_LD && i.op != SM_WSUM && i.op != SM_WMAX && i.op != SM_TID && i.op != SM_SREG && i.op != SM_USQRT)
                o << " " << reg(i.rb, f.ub == SM_SRC_UNIFORM);
            if (i.op == SM_MAD || i.op == SM_Q8DOT || i.op == SM_SDOT || i.op == SM_UDIV || i.op == SM_USQRT || i.op == SM_SUP || (i.op == SM_MULSH && i.sh == 63))
                o << " " << reg(i.rc, f.uc);
            if (i.op == SM_SHL || i.op == SM_SHR || (i.op == SM_MULSH && i.sh != 63)) o << " <<" << i.sh;
            if (i.op == SM_SREG) o << " t>>" << (i.imm & 31) << "&" << (i.imm >> 8 & 31) << "b";
            if (i.op == SM_LD || i.op == SM_STV) o << " bank" << i.bank;
            if (i.guard >= 0) o << (i.gneg ? " @!p" : " @p") << i.guard;
            o << ";";
        }
        o << "  live:";
        for (const auto &lv : mb.live) o << " s" << lv.first << (lv.second < 0 ? "(stv)" : "");
        o << "  " << brn[mb.branch];
        if (mb.branch == SM_BR_COND) o << " s" << mb.bstage << (mb.bneg ? " !p" : " p") << " -> " << mb.target;
        if (mb.branch == SM_BR_JUMP) o << " -> " << mb.target;
        o << "\n";
    }
    return o.str();
}

}  // namespace lower
