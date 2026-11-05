// The SIMT machine in gates. See sm_circ.h, docs/machine.md and src/gpu/sm_ref.c (the semantics, pass for pass).
//
// One pass = one token of the stream. Within a pass, levels go up: the token's features, the sequencer's registers
// (written from the tokens), the ROM block at b, the decode, the scratch words and register files read low, the pending
// stores of the lane that just ended, the datapath (six stages with forwarding), then the scratch words, the state
// bank and the register files written high, and the exit that predicts the next token. State that changes by
// computation lives in scratch words (read at the bottom of the pass, written at its top); state the stream tells lives
// in token-written registers; a value a lane commits travels through its own VAL tokens.
#include "sm_circ.h"

#include <algorithm>
#include <array>

#include "../gpu/kinstr.h"

namespace {

constexpr double KAPPA = 1.0 / (1 << 20);  // the predicted code after the exit

std::vector<int> dims_of(const L &bits) {
    std::vector<int> r;
    for (const Expr &b : bits) r.push_back(b.only_dim());
    return r;
}

L const_bits(const Circ &c, uint64_t v, int n) {
    L r;
    for (int i = 0; i < n; i++) r.push_back((v >> i & 1) ? c.one : Expr());
    return r;
}

Expr eqc(Circ &c, const L &bits, uint64_t v, const std::string &name) { return c.and_(lits_of(bits, (long long)v), name); }

L sext(const L &bits, int n) {
    L r = bits;
    while ((int)r.size() < n) r.push_back(bits.back());
    return r;
}
L zext(const L &bits, int n) {
    L r = bits;
    while ((int)r.size() < n) r.push_back(Expr());
    return r;
}

// one-hot select among bit vectors: out[i] = sum_k sel[k] * vals[k][i]
L onehot_sel(Circ &c, const std::vector<Expr> &sel, const std::vector<L> &vals, int n, const std::string &name) {
    L out;
    for (int i = 0; i < n; i++) {
        Terms t;
        for (size_t k = 0; k < sel.size(); k++)
            if (i < (int)vals[k].size() && !vals[k][(size_t)i].empty() && !sel[k].empty()) t += c.t_and(L{sel[k], vals[k][(size_t)i]});
        out.push_back(c.node(t, name + ".b" + std::to_string(i)));
    }
    return out;
}

// a - b over n_nib nibbles: cout = 1 when a >= b (unsigned)
Adder subn(Circ &c, const L &a, const L &b, int n_nib, const std::string &name) { return add_full(c, zext(a, 4 * n_nib), zext(b, 4 * n_nib), c.one, true, n_nib, name); }

// (x + 2^(k-1)) >> k for k >= 1, x << -k for k <= 0: x a 64-bit two's complement value, k a 7-bit two's complement
// shift (-32..62); the low n bits of the result. The rounding bit is added through the columns, then a two-level
// barrel selects a window of x * 2^32 at offset k + 32.
L shr_round64(Circ &c, const L &x64, const L &k7, int n, const std::string &name) {
    L kpos = sl(k7, 0, 6);
    Expr knonneg = Dy(1) - k7[6];
    Cols cols(16);
    for (int i = 0; i < 64; i++) cols[(size_t)(i / 4)].push_back(CT{Dy(1LL << (i % 4)) * x64[(size_t)i], 0, 1LL << (i % 4)});
    for (int j = 0; j < 62; j++) {
        Expr r = c.and_(lits_of(kpos, j + 1) + L{knonneg}, name + ".rnd" + std::to_string(j));
        cols[(size_t)(j / 4)].push_back(CT{Dy(1LL << (j % 4)) * r, 0, 1LL << (j % 4)});
    }
    L xr = sum_columns(c, cols, 16, name + ".add");  // modulo 2^64
    // u = k + 32 in 0..94 (7 bits): the 7-bit two's complement k plus 32, modulo 128
    L u = add_nibbles(c, zext(k7, 8), const_bits(c, 32, 8), 2, {}, name + ".u");
    L u7 = sl(u, 0, 7);
    L q = sl(u7, 3, 7), r3 = sl(u7, 0, 3);
    std::vector<Expr> T((size_t)n + 8);
    for (int i = 0; i < n + 8; i++) {
        Terms t;
        for (int qq = 0; qq < 12; qq++) {
            const int j = i + 8 * qq;  // position in x * 2^32, sign-extended above bit 95
            if (j >= 32) t += c.t_and(lits_of(q, qq) + L{xr[(size_t)std::min(j - 32, 63)]});
        }
        T[(size_t)i] = c.node(t, name + ".w" + std::to_string(i));
    }
    L out;
    for (int i = 0; i < n; i++) {
        Terms t;
        for (int r = 0; r < 8; r++) t += c.t_and(lits_of(r3, r) + L{T[(size_t)(i + r)]});
        out.push_back(c.node(t, name + ".o" + std::to_string(i)));
    }
    return out;
}

// the unsigned 64-bit product of two 32-bit values
L mul32x32(Circ &c, const L &a, const L &b, const std::string &name) {
    Cols cols(16);
    L bn;
    std::vector<std::pair<long long, long long>> br;
    for (int k = 0; k < 8; k++) {
        bn.push_back(nib(b, k));
        br.push_back({0, 15});
    }
    product_cols(c, cols, a, 8, false, bn, br, name);
    return sum_columns(c, cols, 16, name);
}

Expr or2(Circ &c, const Expr &a, const Expr &b, const std::string &name) { return c.node(c.t_and(L{a}) + c.t_and(L{b}) + c.t_and(L{a, b}, -1), name); }

struct StageDec {
    L op, rd, ra, rb, rc, pd, cmp, ub, g, bank, sh, imm;
    Expr valid, sgn, ua, uc, ud, gvalid, gneg;
    Expr active, writes, uni, is_red, is_stv, is_sup, is_scalar, is_setp, is_scfg;
    std::vector<Expr> is;  // one-hot op
    Expr has(int op) const { return is[(size_t)op].empty() ? Expr() : is[(size_t)op]; }
};

}  // namespace

Circ *build_sm_circuit(const uint8_t classes[SM_STAGES], Target target, SmCircStats *stats) {
    Circ *cp = new Circ(target);
    Circ &c = *cp;
    const Expr one = c.one, zero;
    const Expr &isVLO = c.fam[F_VLO], &isVHI = c.fam[F_VHI], &isVEND = c.fam[F_VEND], &isEXEC = c.fam[F_EXEC], &isNEXT = c.fam[F_NEXT];
    const Expr &isWARP = c.fam[F_WARP], &isEMIT = c.fam[F_EMIT], &isHALT = c.fam[F_HALT], &isGO = c.fam[F_GO], &isLSTEP = c.fam[F_LSTEP], &isTEXT = c.fam[F_TEXT];
    const L &pVLO = c.pay[F_VLO], &pVHI = c.pay[F_VHI], &pVEND = c.pay[F_VEND], &pEXEC = c.pay[F_EXEC], &pNEXT = c.pay[F_NEXT], &pWARP = c.pay[F_WARP];
    L pHI;
    for (int i = 0; i < 16; i++) pHI.push_back(pVHI[(size_t)i] + pVEND[(size_t)i]);
    const Expr lane_end = isVEND + isLSTEP;
    const Expr hiTok = isVHI + isVEND;

    // ---------------------------------------------------------------- the sequencer's token-written registers
    L B = c.reg("seq.b", sl(pEXEC, 0, 12), isEXEC, zero, 12);
    Expr exec_phase = c.and_(L{isEXEC, pEXEC[12]}, "seq.exec_phase");
    L R = c.reg("seq.r", sl(pEXEC, 0, 12), exec_phase, zero, 12);
    L Q = c.reg("seq.q", sl(pNEXT, 0, 12), isNEXT, zero, 12);
    L W = c.reg("seq.w", pWARP, isWARP, isNEXT, 8);
    L EM = c.reg("seq.emit", L{one}, isEMIT, isEXEC + isNEXT + isWARP, 1);
    const Expr in_emit = EM[0], rec = one - EM[0];

    // the descriptor at NEXT: M[2k], M[2k + 1]
    L w0 = c.zeros(32, "q.w0"), w1 = c.zeros(32, "q.w1");
    {
        Bank bk;
        bk.mem = MEM_M;
        BankRead r0, r1;
        r0.addr = L{zero} + sl(pNEXT, 0, 12);
        r0.out = dims_of(w0);
        r1.addr = L{one} + sl(pNEXT, 0, 12);
        r1.out = dims_of(w1);
        bk.reads = {r0, r1};
        c.bank("bank.queue", bk);
    }
    L ENTRY = c.reg("seq.entry", sl(w0, 2, 14), isNEXT, zero, 12);
    L WARPS = c.reg("seq.warps", sl(w0, 14, 22), isNEXT, zero, 8);
    L LWPB = c.reg("seq.lwpb", sl(w0, 22, 24), isNEXT, zero, 2);
    L ADDR = c.reg("seq.addr", sl(w1, 0, 24), isNEXT, zero, 24);
    // the branch descriptor's word M[w1 & 0xFFFFFF] (at the NEXT pass) and the emit's word M[ADDR] (at EMIT and after)
    L mbr = c.zeros(32, "q.mbr"), memit = c.zeros(32, "q.memit");
    {
        Bank bk;
        bk.mem = MEM_M;
        BankRead r0, r1;
        r0.addr = sl(w1, 0, 24);
        r0.out = dims_of(mbr);
        r1.addr = ADDR;
        r1.out = dims_of(memit);
        bk.reads = {r0, r1};
        c.bank("bank.qword", bk);
    }
    // u14 = the args word, in every slot: written at NEXT
    {
        Bank bk;
        bk.mem = MEM_UREG;
        for (int slot = 0; slot < SM_WPB_MAX; slot++) {
            BankWrite w;
            w.addr = const_bits(c, (uint64_t)slot << 4 | SM_UARGS, 7);
            w.data = w1;
            w.data_taps.assign(32, LAG0);
            w.we = isNEXT;
            bk.writes.push_back(w);
        }
        c.bank("bank.uargs", bk);
    }

    // ---------------------------------------------------------------- the ROM at b
    L rom = c.zeros(PW_ROMBLK, "rom");
    {
        Bank bk;
        bk.mem = MEM_ROM;
        BankRead r;
        r.addr = B;
        r.out = dims_of(rom);
        r.width = PW_ROMBLK;
        bk.reads = {r};
        c.bank("bank.rom", bk);
    }
    L live = sl(rom, 480, 544);
    L nlive = sl(live, 0, 3), lbr = sl(live, 15, 18), lstage = sl(live, 18, 21), ltarget = sl(live, 23, 35), llwpb = sl(live, 35, 37);
    Expr lneg = live[21], lany = live[22];
    std::vector<L> lsrc;
    for (int i = 0; i < SM_LMAX; i++) lsrc.push_back(sl(live, 3 + 3 * i, 6 + 3 * i));
    std::vector<Expr> br_is(5);
    for (int k = 0; k < 5; k++) br_is[(size_t)k] = eqc(c, lbr, (uint64_t)k, "live.br" + std::to_string(k));

    // ---------------------------------------------------------------- the decode
    std::vector<StageDec> D(SM_STAGES);
    for (int s = 0; s < SM_STAGES; s++) {
        StageDec &d = D[(size_t)s];
        L f = sl(rom, 80 * s, 80 * s + 80);
        d.op = sl(f, 0, 5);
        d.rd = sl(f, 5, 9);
        d.ra = sl(f, 9, 13);
        d.rb = sl(f, 13, 17);
        d.rc = sl(f, 17, 21);
        d.valid = f[21];
        d.pd = sl(f, 22, 25);
        d.cmp = sl(f, 25, 28);
        d.sgn = f[28];
        d.ub = sl(f, 29, 31);
        d.ua = f[31];
        d.uc = f[32];
        d.ud = f[33];
        d.gvalid = f[34];
        d.g = sl(f, 35, 38);
        d.gneg = f[38];
        d.bank = sl(f, 39, 42);
        d.sh = sl(f, 42, 48);
        d.imm = sl(f, 48, 80);
        const std::string sn = "s" + std::to_string(s) + ".";
        d.is.resize(SM_NOPS);
        for (int op = 1; op < SM_NOPS; op++) {
            if (!(classes[s] & lower::class_of(op))) continue;  // a unit the stage does not have is never decoded
            d.is[(size_t)op] = c.and_(lits_of(d.op, op) + L{d.valid}, sn + "op" + std::to_string(op));
        }
        for (int op = 1; op < SM_NOPS; op++) d.active += d.has(op);
        d.is_red = d.has(SM_WSUM) + d.has(SM_WMAX);
        d.is_stv = d.has(SM_STV);
        d.is_sup = d.has(SM_SUP);
        d.is_setp = d.has(SM_SETP);
        d.is_scfg = d.has(SM_SCFG);
        d.is_scalar = d.has(SM_UDIV) + d.has(SM_USQRT);
        d.writes = d.active - d.is_setp - d.is_stv - d.is_scfg;
        d.uni = c.node(c.t_and(L{d.active, d.ud}) + c.t_and(L{d.is_red, one - d.ud}) + c.t_and(L{d.is_scalar, one - d.ud}), sn + "uni");
    }

    // live-outs: lane items, uniform items, their stages
    std::vector<Expr> live_valid(SM_LMAX);
    std::vector<std::vector<Expr>> src_is(SM_LMAX, std::vector<Expr>(SM_STAGES));
    std::vector<Expr> lane_live(SM_LMAX), uni_live(SM_LMAX);
    for (int i = 0; i < SM_LMAX; i++) {
        live_valid[(size_t)i] = c.node(c.t_ge(from_bits(nlive), i + 1, 0, 7), "live.v" + std::to_string(i));
        Terms tl, tu;
        for (int s = 0; s < SM_STAGES; s++) {
            src_is[(size_t)i][(size_t)s] = c.and_(lits_of(lsrc[(size_t)i], s) + L{live_valid[(size_t)i]}, "live.src" + std::to_string(i) + "." + std::to_string(s));
            tl += c.t_and(L{src_is[(size_t)i][(size_t)s], one - D[(size_t)s].uni});
            tu += c.t_and(L{src_is[(size_t)i][(size_t)s], D[(size_t)s].uni, one - D[(size_t)s].is_red});
        }
        lane_live[(size_t)i] = c.node(tl, "live.lane" + std::to_string(i));
        uni_live[(size_t)i] = c.node(tu, "live.uni" + std::to_string(i));
    }
    std::vector<Expr> red_live(SM_STAGES);  // the reduction's result is a live-out
    for (int s = 0; s < SM_STAGES; s++) {
        Terms t;
        for (int i = 0; i < SM_LMAX; i++) t += c.t_and(L{src_is[(size_t)i][(size_t)s], D[(size_t)s].is_red});
        red_live[(size_t)s] = c.node(t, "live.red" + std::to_string(s));
    }
    Expr nll = sum(lane_live), nred, nuni = sum(uni_live);
    for (int s = 0; s < SM_STAGES; s++) nred += D[(size_t)s].is_red;
    Expr nparts = nll + nred;
    Expr nparts_zero = c.node(c.t_eq(nparts, 0, 0, 6), "live.nparts0");
    Expr nuni_zero = c.node(c.t_eq(nuni, 0, 0, 4), "live.nuni0");
    std::vector<std::vector<Expr>> item_stage(6, std::vector<Expr>(SM_STAGES)), uitem_stage(4, std::vector<Expr>(SM_STAGES));
    {
        std::vector<Expr> before(SM_LMAX), ubefore(SM_LMAX), red_rank(SM_STAGES);
        for (int i = 0; i < SM_LMAX; i++)
            for (int j = 0; j < i; j++) {
                before[(size_t)i] += lane_live[(size_t)j];
                ubefore[(size_t)i] += uni_live[(size_t)j];
            }
        for (int s = 0; s < SM_STAGES; s++)
            for (int t = 0; t < s; t++) red_rank[(size_t)s] += D[(size_t)t].is_red;
        for (int k = 0; k < 6; k++)
            for (int s = 0; s < SM_STAGES; s++) {
                Terms t;
                if (k < SM_LMAX)
                    for (int i = 0; i < SM_LMAX; i++) t += c.t_eq(before[(size_t)i], k, 0, 3, 1, L{lane_live[(size_t)i], src_is[(size_t)i][(size_t)s]});
                t += c.t_eq(nll + red_rank[(size_t)s], k, 0, 5, 1, L{D[(size_t)s].is_red});
                item_stage[(size_t)k][(size_t)s] = c.node(t, "item" + std::to_string(k) + ".s" + std::to_string(s));
            }
        for (int k = 0; k < 4; k++)
            for (int s = 0; s < SM_STAGES; s++) {
                Terms t;
                for (int i = 0; i < SM_LMAX; i++) t += c.t_eq(ubefore[(size_t)i], k, 0, 3, 1, L{uni_live[(size_t)i], src_is[(size_t)i][(size_t)s]});
                uitem_stage[(size_t)k][(size_t)s] = c.node(t, "uitem" + std::to_string(k) + ".s" + std::to_string(s));
            }
    }

    // ---------------------------------------------------------------- the scratch words, read low
    std::vector<L> scr(SCR_WORDS);
    {
        Bank bk;
        bk.mem = MEM_SCR;
        for (int w = 0; w < SCR_WORDS; w++) {
            scr[(size_t)w] = c.zeros(32, "scr" + std::to_string(w));
            BankRead r;
            r.addr = const_bits(c, (uint64_t)w, 7);
            r.out = dims_of(scr[(size_t)w]);
            bk.reads.push_back(r);
        }
        c.bank("bank.scr.rd", bk);
    }
    const L LNp = sl(scr[SCR_LN], 0, 6), LIp = sl(scr[SCR_LI], 0, 3);
    const Expr in_uni_p = c.node(c.t_ge(from_bits(LNp), 32, 0, 63), "ln.uni_p");
    const Expr in_lanes_p = one - in_uni_p;
    const Expr ln31_p = eqc(c, LNp, 31, "ln.31p");
    Expr ln_step = c.and_(L{lane_end, in_lanes_p}, "ln.step");
    L LN1 = add_nibbles(c, zext(LNp, 8), zext(L{ln_step}, 8), 2, {}, "ln.inc");
    L LN = mux_bits(c, isEXEC, const_bits(c, 0, 6), sl(LN1, 0, 6), "ln");
    Expr in_uni = c.node(c.t_ge(from_bits(LN), 32, 0, 63), "ln.uni");
    Expr in_lanes = one - in_uni;
    Expr ln_zero = eqc(c, LN, 0, "ln.zero");
    L LN5 = sl(LN, 0, 5);
    L LI1 = add_nibbles(c, zext(LIp, 4), zext(L{isVHI}, 4), 1, {}, "li.inc");
    L LI = mux_bits(c, isEXEC + lane_end, const_bits(c, 0, 3), sl(LI1, 0, 3), "li");
    std::vector<Expr> li_is(6), lip_is(4);
    for (int k = 0; k < 6; k++) li_is[(size_t)k] = eqc(c, LI, (uint64_t)k, "li.is" + std::to_string(k));
    for (int k = 0; k < 4; k++) lip_is[(size_t)k] = eqc(c, LIp, (uint64_t)k, "li.isp" + std::to_string(k));
    const Expr new_lane = isEXEC + lane_end;  // this pass computes a new lane (or the final pass)
    std::vector<Expr> cur_stage(SM_STAGES), ucur_stage(SM_STAGES), curp_stage(SM_STAGES), ucurp_stage(SM_STAGES);
    for (int s = 0; s < SM_STAGES; s++) {
        Terms t, tu, tp, tup;
        for (int k = 0; k < 6; k++) t += c.t_and(L{li_is[(size_t)k], item_stage[(size_t)k][(size_t)s]});
        for (int k = 0; k < 4; k++) tu += c.t_and(L{li_is[(size_t)k], uitem_stage[(size_t)k][(size_t)s]});
        for (int k = 0; k < 4; k++) tp += c.t_and(L{lip_is[(size_t)k], item_stage[(size_t)k][(size_t)s]});
        for (int k = 0; k < 4; k++) tup += c.t_and(L{lip_is[(size_t)k], uitem_stage[(size_t)k][(size_t)s]});
        cur_stage[(size_t)s] = c.node(t, "cur.s" + std::to_string(s));
        ucur_stage[(size_t)s] = c.node(tu, "ucur.s" + std::to_string(s));
        curp_stage[(size_t)s] = c.node(tp, "curp.s" + std::to_string(s));    // the item being committed (by LI before the step)
        ucurp_stage[(size_t)s] = c.node(tup, "ucurp.s" + std::to_string(s));
    }
    Expr last_item, ulast_item;
    for (int k = 1; k <= 6; k++) last_item += c.node(c.t_eq(nparts, k, 0, 6, 1, L{li_is[(size_t)(k - 1)]}), "last.k" + std::to_string(k));
    for (int k = 1; k <= 4; k++) ulast_item += c.node(c.t_eq(nuni, k, 0, 4, 1, L{li_is[(size_t)(k - 1)]}), "ulast.k" + std::to_string(k));

    const L slot = sl(W, 0, 3);
    const L SCFG = scr[SCR_SCFG];

    const Expr uni_fire = c.node(c.t_and(L{isVEND, in_uni_p, rec}) + c.t_and(L{isEXEC}) + c.t_and(L{isNEXT}) + c.t_and(L{isWARP}), "stv.ufire");
    const Expr lane_fire = c.and_(L{lane_end, in_lanes_p}, "stv.lfire");
    // ---------------------------------------------------------------- the pending stores: lane slots fire at a lane's end, uniform
    // slots at the uniform part's VEND or at the next record's first token
    {
        std::vector<Bank> bks(MEM_COUNT);
        for (int k = 0; k < MEM_COUNT; k++) bks[(size_t)k].mem = k;
        // the uniform slots fire once: at the uniform part's VEND, or at the next record's first token when the record had no uniform
        // items; the flags are cleared in that pass (below), so nothing fires twice; a lane slot fires only when a lane ended
        for (int i = 0; i < STV_SLOTS; i++) {
            const L &pa = scr[(size_t)(SCR_STV0 + 3 * i)], &pv = scr[(size_t)(SCR_STV0 + 3 * i + 1)], &pf = scr[(size_t)(SCR_STV0 + 3 * i + 2)];
            const Expr fire = c.and_(L{i < 4 ? lane_fire : uni_fire, pf[3]}, "stv.fire" + std::to_string(i));
            const L bank3 = sl(pf, 0, 3);
            auto port = [&](int mem, uint64_t bankcode, int width) {
                BankWrite w;
                w.addr = pa;
                w.data = width == 8 ? sl(pv, 0, 8) : (width == 16 ? sl(pv, 0, 16) : pv);
                w.data_taps.assign(w.data.size(), LAG0);
                w.we = c.and_(L{fire} + lits_of(bank3, (long long)bankcode), "stv.we" + std::to_string(i) + "." + std::to_string(bankcode));
                w.width = width;
                bks[(size_t)mem].writes.push_back(w);
            };
            port(MEM_M, SM_BANK_M, PW_WORD);
            port(MEM_A, SM_BANK_A8, PW_BYTE);
            port(MEM_A, SM_BANK_A32, PW_WORD);
            port(MEM_S, SM_BANK_S, PW_HALF);
            port(MEM_C, SM_BANK_C, PW_BYTE);
        }
        c.bank("bank.stv.m", bks[MEM_M]);
        c.bank("bank.stv.a", bks[MEM_A]);
        c.bank("bank.stv.s", bks[MEM_S]);
        c.bank("bank.stv.c", bks[MEM_C]);
    }

    // ---------------------------------------------------------------- register files, read low: every operand of every stage
    std::vector<std::array<L, 4>> rl(SM_STAGES), ru(SM_STAGES);
    {
        Bank rb, ub;
        rb.mem = MEM_REG;
        ub.mem = MEM_UREG;
        for (int s = 0; s < SM_STAGES; s++) {
            const StageDec &d = D[(size_t)s];
            const L idx[4] = {d.ra, d.rb, d.rc, d.rd};
            for (int o = 0; o < 4; o++) {
                const std::string nm = std::string("reg.s") + std::to_string(s) + "." + "abcd"[o];
                rl[(size_t)s][(size_t)o] = c.zeros(32, nm + ".l");
                BankRead r;
                r.addr = idx[o] + LN5 + slot;  // bits 0..3 register, 4..8 lane, 9..11 slot
                r.out = dims_of(rl[(size_t)s][(size_t)o]);
                rb.reads.push_back(r);
                if (o < 3) {
                    ru[(size_t)s][(size_t)o] = c.zeros(32, nm + ".u");
                    BankRead q;
                    q.addr = idx[o] + slot;  // bits 0..3 register, 4..6 slot
                    q.out = dims_of(ru[(size_t)s][(size_t)o]);
                    ub.reads.push_back(q);
                }
            }
        }
        c.bank("bank.reg.rd", rb);
        c.bank("bank.ureg.rd", ub);
    }

    // ---------------------------------------------------------------- the datapath
    std::vector<L> res(SM_STAGES), Aval(SM_STAGES);
    std::vector<Expr> cmp_res(SM_STAGES), on_s(SM_STAGES);
    std::vector<L> stv_addr(SM_STAGES), stv_val(SM_STAGES), sup_addr(SM_STAGES), sup_val(SM_STAGES), red_val(SM_STAGES);
    std::vector<Expr> sup_we(SM_STAGES), red_rank1(SM_STAGES);
    L div_r = scr[SCR_DIV_R0] + L{scr[SCR_DIV_R1][0]}, div_n = scr[SCR_DIV_N0] + scr[SCR_DIV_N1];
    L sq_v = scr[SCR_SQ_V0] + scr[SCR_SQ_V1], sq_r = scr[SCR_SQ_R0] + L{scr[SCR_SQ_R1][0]};
    L div_r_new, div_n_new, sq_v_new, sq_r_new;
    Expr scalar_write;
    L tid_bits = LN5 + W;  // 32 w + lane, 13 bits
    // a reduction's running value: B after the lane last computed, A before it; a lane's own passes read A (idempotent)
    const L redA[2] = {scr[SCR_RED0A], scr[SCR_RED1A]}, redB[2] = {scr[SCR_RED0B], scr[SCR_RED1B]};
    L scfg_new = SCFG;

    for (int s = 0; s < SM_STAGES; s++) {
        const StageDec &d = D[(size_t)s];
        const std::string sn = "dp" + std::to_string(s) + ".";
        const uint8_t cls = classes[s];
        // ---- operands with forwarding
        auto operand = [&](int o, const L &idx, const Expr &is_uni, const Expr &is_imm, const L &imm_bits) {
            const char on = "abcd"[o];
            Expr idx_zero = eqc(c, idx, SM_ZERO, sn + on + ".z");
            std::vector<Expr> match(s), latest(s);
            for (int t = 0; t < s; t++) {
                const StageDec &dt = D[(size_t)t];
                L eqb;
                for (int b = 0; b < 4; b++)
                    eqb.push_back(c.node(c.t_and(L{idx[(size_t)b], dt.rd[(size_t)b]}) + c.t_and(L{one - idx[(size_t)b], one - dt.rd[(size_t)b]}), sn + on + ".eq" + std::to_string(t) + "." + std::to_string(b)));
                Expr kind = c.node(c.t_and(L{is_uni, dt.uni}) + c.t_and(L{one - is_uni, one - dt.uni}), sn + on + ".kind" + std::to_string(t));
                match[(size_t)t] = c.and_(eqb + L{dt.writes, kind, one - is_imm, one - idx_zero}, sn + on + ".m" + std::to_string(t));
            }
            for (int t = 0; t < s; t++) {
                L lits{match[(size_t)t]};
                for (int t2 = t + 1; t2 < s; t2++) lits.push_back(one - match[(size_t)t2]);
                latest[(size_t)t] = c.and_(lits, sn + on + ".fw" + std::to_string(t));
            }
            L nl;
            for (int t = 0; t < s; t++) nl.push_back(one - match[(size_t)t]);
            Expr none = s ? c.and_(nl, sn + on + ".none") : one;
            L out;
            for (int i = 0; i < 32; i++) {
                Terms tm;
                for (int t = 0; t < s; t++) tm += c.t_and(L{latest[(size_t)t], res[(size_t)t][(size_t)i]});
                if (o < 3) tm += c.t_and(L{none, one - is_imm, is_uni, ru[(size_t)s][(size_t)o][(size_t)i]});
                tm += c.t_and(L{none, one - is_imm, one - is_uni, rl[(size_t)s][(size_t)o][(size_t)i]});
                if (!is_imm.empty() && i < (int)imm_bits.size()) tm += c.t_and(L{is_imm, imm_bits[(size_t)i]});
                out.push_back(c.node(tm, sn + on + ".v" + std::to_string(i)));
            }
            return out;
        };
        const Expr ub_imm = eqc(c, d.ub, SM_SRC_IMM, sn + "ub.imm"), ub_uni = eqc(c, d.ub, SM_SRC_UNIFORM, sn + "ub.uni");
        L A = operand(0, d.ra, d.ua, Expr(), {});
        L Bv = operand(1, d.rb, ub_uni, ub_imm, d.imm);
        L Cv = operand(2, d.rc, d.uc, Expr(), {});
        L Dv = (cls & SM_CL_MUL) ? operand(3, d.rd, Expr(), Expr(), {}) : L();
        Aval[(size_t)s] = A;
        // ---- the guard: the predicate g as the latest earlier setp with pd == g left it
        Expr pv;
        if (s) {
            Terms t;
            for (int tt = 0; tt < s; tt++) {
                auto pd_eq = [&](int t2, const std::string &tag) {
                    L l{D[(size_t)t2].is_setp};
                    for (int b = 0; b < 3; b++)
                        l.push_back(c.node(c.t_and(L{d.g[(size_t)b], D[(size_t)t2].pd[(size_t)b]}) + c.t_and(L{one - d.g[(size_t)b], one - D[(size_t)t2].pd[(size_t)b]}), sn + "g.eq" + tag + "." + std::to_string(b)));
                    return c.and_(l, sn + "g.setp" + tag);
                };
                L lits{pd_eq(tt, std::to_string(tt))};
                for (int t2 = tt + 1; t2 < s; t2++) lits.push_back(one - pd_eq(t2, std::to_string(tt) + "_" + std::to_string(t2)));
                t += c.t_and(lits + L{cmp_res[(size_t)tt]});
            }
            pv = c.node(t, sn + "g.pv");
        }
        Expr on = pv.empty() ? c.node(c.t_and(L{one - d.gvalid}) + c.t_and(L{d.gvalid, d.gneg}), sn + "on")
                             : c.node(c.t_and(L{one - d.gvalid}) + c.t_and(L{d.gvalid, pv, one - d.gneg}) + c.t_and(L{d.gvalid, one - pv, d.gneg}), sn + "on");
        on_s[(size_t)s] = on;
        // ---- the address: A + imm
        L addr = add_nibbles(c, A, d.imm, 8, {}, sn + "addr");
        // ---- any: add / sub / min / max / setp (A + B and A - B), selp, tid, sreg, ld, stv
        Adder ads = add_full(c, A, Bv, one, true, 8, sn + "sub");      // A - B
        Adder ada = add_full(c, A, Bv, Expr(), false, 8, sn + "sum");  // A + B
        const L &diff = ads.r;
        Expr ult = one - ads.cout;
        Expr a31 = A[31], b31 = Bv[31], d31 = diff[31];
        Expr slt = c.node(c.t_and(L{a31, one - b31}) + c.t_and(L{a31, b31, d31}) + c.t_and(L{one - a31, one - b31, d31}), sn + "slt");
        Expr eq = c.and_(nots(diff), sn + "eq");
        Expr lt = c.node(c.t_and(L{d.sgn, slt}) + c.t_and(L{one - d.sgn, ult}), sn + "lt");
        std::vector<Expr> cmp_is(6);
        for (int k = 0; k < 6; k++) cmp_is[(size_t)k] = eqc(c, d.cmp, (uint64_t)k, sn + "cmp" + std::to_string(k));
        cmp_res[(size_t)s] = c.node(c.t_and(L{cmp_is[SM_CMP_EQ], eq}) + c.t_and(L{cmp_is[SM_CMP_NE], one - eq}) + c.t_and(L{cmp_is[SM_CMP_LT], lt}) + c.t_and(L{cmp_is[SM_CMP_LE], lt}) + c.t_and(L{cmp_is[SM_CMP_LE], eq}) + c.t_and(L{cmp_is[SM_CMP_GT], one - lt}) + c.t_and(L{cmp_is[SM_CMP_GT], eq}, -1) + c.t_and(L{cmp_is[SM_CMP_GE], one - lt}), sn + "cmpres");
        L maxv = mux_bits(c, lt, Bv, A, sn + "max"), minv = mux_bits(c, lt, A, Bv, sn + "min");
        L selv = mux_bits(c, on, A, Bv, sn + "selp");
        L tidv = zext(tid_bits, 32);
        L sregv;
        {
            L shv = sl(d.imm, 0, 5), wd = sl(d.imm, 8, 13);
            L t13 = zext(tid_bits, 32);
            for (int i = 0; i < 32; i++) {
                Terms t;
                for (int k = 0; k < 32 && i + k < 13; k++) t += c.t_and(lits_of(shv, k) + L{t13[(size_t)(i + k)]});
                Expr shifted = c.node(t, sn + "sreg.sh" + std::to_string(i));
                Expr keep = c.node(c.t_ge(from_bits(wd), i + 1, 0, 31), sn + "sreg.k" + std::to_string(i));
                sregv.push_back(c.and_(L{shifted, keep}, sn + "sreg.b" + std::to_string(i)));
            }
        }
        L ldm = c.zeros(32, sn + "ld.m"), lda8 = c.zeros(8, sn + "ld.a8"), lda32 = c.zeros(32, sn + "ld.a32"), lds = c.zeros(16, sn + "ld.s"), ldc = c.zeros(8, sn + "ld.c"), ldw8 = c.zeros(8, sn + "ld.w8"), ldw16 = c.zeros(16, sn + "ld.w16");
        {
            Bank bm, ba, bs, bc, bw;
            bm.mem = MEM_M; ba.mem = MEM_A; bs.mem = MEM_S; bc.mem = MEM_C; bw.mem = MEM_W;
            BankRead r;
            r.addr = addr;
            r.out = dims_of(ldm); r.width = PW_WORD; bm.reads.push_back(r);
            r.out = dims_of(lda8); r.width = PW_BYTE; ba.reads.push_back(r);
            r.out = dims_of(lda32); r.width = PW_WORD; ba.reads.push_back(r);
            r.out = dims_of(lds); r.width = PW_HALF; bs.reads.push_back(r);
            r.out = dims_of(ldc); r.width = PW_BYTE; bc.reads.push_back(r);
            r.out = dims_of(ldw8); r.width = PW_BYTE; bw.reads.push_back(r);
            r.out = dims_of(ldw16); r.width = PW_HALF; bw.reads.push_back(r);
            c.bank(sn + "bank.ld.m", bm);
            c.bank(sn + "bank.ld.a", ba);
            c.bank(sn + "bank.ld.s", bs);
            c.bank(sn + "bank.ld.c", bc);
            c.bank(sn + "bank.ld.w", bw);
        }
        L ldw32 = c.zeros(32, sn + "ld.w32");
        {
            Bank bf;
            bf.mem = MEM_F16;
            BankRead r;
            r.addr = ldw16;
            r.out = dims_of(ldw32);
            bf.reads.push_back(r);
            c.bank(sn + "bank.ld.f16", bf);
        }
        std::vector<Expr> bank_is(SM_NBANKS_LD);
        for (int k = 0; k < SM_NBANKS_LD; k++) bank_is[(size_t)k] = eqc(c, d.bank, (uint64_t)k, sn + "bank" + std::to_string(k));
        Expr jlow = one - addr[5];  // a byte j >= 32 of a block reads 0
        L lda8s = and_bits(c, sext(lda8, 32), jlow, sn + "ld.a8s"), ldw8s = and_bits(c, sext(ldw8, 32), jlow, sn + "ld.w8s");
        L ldv = onehot_sel(c, {bank_is[SM_BANK_M], bank_is[SM_BANK_A8], bank_is[SM_BANK_A32], bank_is[SM_BANK_S], bank_is[SM_BANK_C], bank_is[SM_BANK_W8], bank_is[SM_BANK_W32]},
                           {ldm, lda8s, lda32, sext(lds, 32), sext(ldc, 32), ldw8s, ldw32}, 32, sn + "ld");
        stv_addr[(size_t)s] = addr;
        stv_val[(size_t)s] = Bv;
        // ---- logic: and or xor, shifts, clz
        L logicv, shv, clzv;
        if (cls & SM_CL_LOGIC) {
            for (int i = 0; i < 32; i++) {
                Terms t;
                t += c.t_and(L{d.has(SM_AND), A[(size_t)i], Bv[(size_t)i]});
                t += c.t_and(L{d.has(SM_OR), A[(size_t)i]}) + c.t_and(L{d.has(SM_OR), Bv[(size_t)i]}) + c.t_and(L{d.has(SM_OR), A[(size_t)i], Bv[(size_t)i]}, -1);
                t += c.t_and(L{d.has(SM_XOR), A[(size_t)i], one - Bv[(size_t)i]}) + c.t_and(L{d.has(SM_XOR), one - A[(size_t)i], Bv[(size_t)i]});
                logicv.push_back(c.node(t, sn + "logic.b" + std::to_string(i)));
            }
            Expr sh63 = eqc(c, d.sh, 63, sn + "sh63");
            L k6 = mux_bits(c, sh63, sl(Cv, 0, 6), d.sh, sn + "k6");
            Expr kbig = k6[5];  // k >= 32
            L kq = sl(k6, 3, 5), kr = sl(k6, 0, 3);
            std::vector<Expr> l8(32), r8(32);
            for (int i = 0; i < 32; i++) {
                Terms tl, tr;
                for (int q = 0; q < 4; q++) {
                    if (i - 8 * q >= 0) tl += c.t_and(lits_of(kq, q) + L{A[(size_t)(i - 8 * q)]});
                    if (i + 8 * q < 32) tr += c.t_and(lits_of(kq, q) + L{A[(size_t)(i + 8 * q)]});
                    else tr += c.t_and(lits_of(kq, q) + L{d.sgn, A[31]});
                }
                l8[(size_t)i] = c.node(tl, sn + "shl8." + std::to_string(i));
                r8[(size_t)i] = c.node(tr, sn + "shr8." + std::to_string(i));
            }
            for (int i = 0; i < 32; i++) {
                Terms t;
                for (int r = 0; r < 8; r++) {
                    if (i - r >= 0) t += c.t_and(L{d.has(SM_SHL), one - kbig} + lits_of(kr, r) + L{l8[(size_t)(i - r)]});
                    if (i + r < 32) t += c.t_and(L{d.has(SM_SHR), one - kbig} + lits_of(kr, r) + L{r8[(size_t)(i + r)]});
                    else t += c.t_and(L{d.has(SM_SHR), one - kbig} + lits_of(kr, r) + L{d.sgn, A[31]});
                }
                t += c.t_and(L{d.has(SM_SHR), kbig, d.sgn, A[31]});
                shv.push_back(c.node(t, sn + "shift.b" + std::to_string(i)));
            }
            L th;
            for (int n = 1; n <= 32; n++) th.push_back(c.and_(nots(sl(A, 32 - n, 32)), sn + "clz.z" + std::to_string(n)));
            clzv = bits_of(th, 6, 32);
        }
        // ---- the scfg constants in force at this stage: the latest earlier scfg's operand, else the committed word
        L cfg = SCFG;
        for (int t = 0; t < s; t++)
            if (!D[(size_t)t].is_scfg.empty()) cfg = mux_bits(c, c.and_(L{D[(size_t)t].is_scfg, one - D[(size_t)t].uni}, sn + "cfgfw" + std::to_string(t)), Aval[(size_t)t], cfg, sn + "cfg" + std::to_string(t));  // a uniform scfg runs in the final pass: no lane sees it in its block
        // ---- mul: mad, mulhi, mulsh, sup
        L madv, mulhiv, mulshv, supv;
        if (cls & SM_CL_MUL) {
            L X = mux_bits(c, d.is_sup, Cv, A, sn + "mul.x");  // sup multiplies kreal (C) by delta (B)
            L P = mul32x32(c, X, Bv, sn + "mul.p");
            L corr1 = and_bits(c, Bv, X[31], sn + "mul.c1"), corr2 = and_bits(c, X, Bv[31], sn + "mul.c2");
            L hi1 = subn(c, sl(P, 32, 64), corr1, 8, sn + "mul.h1").r;
            L hi_s = subn(c, hi1, corr2, 8, sn + "mul.h2").r;
            L Ps = sl(P, 0, 32) + hi_s;  // the signed 64-bit product
            madv = add_nibbles(c, sl(P, 0, 32), Cv, 8, {}, sn + "mad");
            mulhiv = sl(P, 32, 64);
            Expr sh63 = eqc(c, d.sh, 63, sn + "mulsh.sh63");
            L k7 = mux_bits(c, sh63, sl(Cv, 0, 7), zext(d.sh, 7), sn + "mulsh.k");
            mulshv = shr_round64(c, Ps, k7, 32, sn + "mulsh");
            // sup
            L sold = c.zeros(16, sn + "sup.old");
            {
                Bank bs;
                bs.mem = MEM_S;
                BankRead r;
                r.addr = addr;
                r.out = dims_of(sold);
                r.width = PW_HALF;
                bs.reads.push_back(r);
                c.bank(sn + "bank.sup.rd", bs);
            }
            L e16 = sl(cfg, 0, 16), sh1 = sl(cfg, 16, 23), sh2 = sl(cfg, 23, 30);
            L oe;
            {
                Cols cols(8);
                L en;
                std::vector<std::pair<long long, long long>> er;
                for (int k = 0; k < 4; k++) {
                    en.push_back(nib(e16, k));
                    er.push_back({0, 15});
                }
                product_cols(c, cols, sold, 4, true, en, er, sn + "sup.oe");
                oe = sum_columns(c, cols, 8, sn + "sup.oe");
            }
            L k1 = subn(c, zext(sh1, 8), const_bits(c, 32, 8), 2, sn + "sup.k1").r;
            L k2 = subn(c, zext(sh2, 8), const_bits(c, 32, 8), 2, sn + "sup.k2").r;
            L y1 = shr_round64(c, sext(oe, 64), sl(k1, 0, 7), 56, sn + "sup.y1");
            L y2 = shr_round64(c, Ps, const_bits(c, 16, 7), 56, sn + "sup.y2");
            L val = add_nibbles(c, y1, y2, 14, {}, sn + "sup.val");
            L y3 = shr_round64(c, sext(val, 64), sl(k2, 0, 7), 56, sn + "sup.y3");
            Expr inr = c.node(c.t_and(nots(sl(y3, 15, 56))) + c.t_and(sl(y3, 15, 56)), sn + "sup.inr");
            L sv;
            for (int b = 0; b < 16; b++) {
                Terms tt = c.t_and(L{inr, y3[(size_t)b]});
                if (b < 15) tt += c.t_and(L{one - inr, one - y3[55]});
                if (b == 0 || b == 15) tt += c.t_and(L{one - inr, y3[55]});
                sv.push_back(c.node(tt, sn + "sup.sv" + std::to_string(b)));
            }
            L inv = nots(sl(sv, 0, 15)) + L{zero};
            L neg = add_nibbles(c, inv, const_bits(c, 1, 16), 4, {}, sn + "sup.neg");
            L mag = mux_bits(c, sv[15], neg, sv, sn + "sup.mag");
            L mag32 = zext(sl(mag, 0, 16), 32);
            Adder dm = subn(c, Dv, mag32, 8, sn + "sup.cmp");  // D - mag: cout = D >= mag
            L acc_new = mux_bits(c, dm.cout, Dv, mag32, sn + "sup.acc");
            supv = mux_bits(c, on, acc_new, Dv, sn + "sup.res");
            sup_addr[(size_t)s] = addr;
            sup_val[(size_t)s] = sext(sv, 32);
            sup_we[(size_t)s] = c.and_(L{d.is_sup, on}, sn + "sup.we");  // the write waits in a pending word until the lane ends
        }
        // ---- dot: q8dot
        L dotv;
        if (cls & SM_CL_DOT) {
            Expr bank1 = d.bank[0];
            L xblk = mux_bits(c, ub_imm, A, addr, sn + "dot.xblk");
            L ablk = mux_bits(c, ub_imm, d.imm, add_nibbles(c, Bv, d.imm, 8, {}, sn + "dot.badd"), sn + "dot.ablk");
            L wblk = c.zeros(PW_WBLK, sn + "dot.w"), ablkb = c.zeros(PW_ABLK, sn + "dot.a"), xab = c.zeros(PW_ABLK, sn + "dot.xa");
            {
                Bank bw, ba;
                bw.mem = MEM_W;
                ba.mem = MEM_A;
                BankRead r;
                r.addr = xblk; r.out = dims_of(wblk); r.width = PW_WBLK; bw.reads.push_back(r);
                r.addr = ablk; r.out = dims_of(ablkb); r.width = PW_ABLK; ba.reads.push_back(r);
                r.addr = xblk; r.out = dims_of(xab); r.width = PW_ABLK; ba.reads.push_back(r);
                c.bank(sn + "bank.dot.w", bw);
                c.bank(sn + "bank.dot.a", ba);
            }
            L mwf = c.zeros(32, sn + "dot.mwf");
            {
                Bank bf;
                bf.mem = MEM_F16;
                BankRead r;
                r.addr = sl(wblk, 0, 16);
                r.out = dims_of(mwf);
                bf.reads.push_back(r);
                c.bank(sn + "bank.dot.f16", bf);
            }
            std::vector<L> xq(32), aq(32);
            for (int i = 0; i < 32; i++) {
                xq[(size_t)i] = mux_bits(c, bank1, sl(xab, 8 * i, 8 * i + 8), sl(wblk, 16 + 8 * i, 24 + 8 * i), sn + "dot.x" + std::to_string(i));
                aq[(size_t)i] = sl(ablkb, 8 * i, 8 * i + 8);
            }
            L mw = mux_bits(c, bank1, sl(xab, 256, 288), mwf, sn + "dot.mw");
            L xs = sl(ablkb, 256, 288);
            L w0, w1;
            for (int i = 0; i < 32; i++) {
                w0.push_back(nib(xq[(size_t)i], 0));
                w1.push_back(nib(xq[(size_t)i], 1) - Dy(16) * xq[(size_t)i][7]);  // the high nibble as signed
            }
            Cols cols(6);
            long long cst = 0;
            plane_cols(c, cols, cst, aq, w0, 0, 15, 0, sn + "dot.lo");
            plane_cols(c, cols, cst, aq, w1, -8, 7, 1, sn + "dot.hi");
            cst = fmod_py(cst, 1LL << 24);
            for (int k = 0; k < 6; k++) {
                const long long v = cst >> (4 * k) & 15;
                if (v) cols[(size_t)k].push_back(CT{Dy(v) * one, v, v});
            }
            L inner = sum_columns(c, cols, 6, sn + "dot.in");
            L mwn, xsn;
            for (int j = 0; j < 8; j++) {
                mwn.push_back(nib(mw, j));
                xsn.push_back(nib(xs, j));
            }
            L t = mul_round(c, inner, 6, mwn, sn + "dot.t");
            L u = mul_round(c, t, 8, xsn, sn + "dot.u");
            dotv = add_nibbles(c, u, Cv, 8, {}, sn + "dot.acc");
        }
        // ---- state: sdot
        L sdotv;
        if (cls & SM_CL_STATE) {
            L sel = c.zeros(16, sn + "sdot.s");
            {
                Bank bs;
                bs.mem = MEM_S;
                BankRead r;
                r.addr = addr;
                r.out = dims_of(sel);
                r.width = PW_HALF;
                bs.reads.push_back(r);
                c.bank(sn + "bank.sdot", bs);
            }
            Cols cols(8);
            L kn{nib(Bv, 0), from_bits(sl(Bv, 4, 7)) - Dy(8) * Bv[7]};
            const std::vector<std::pair<long long, long long>> kr{{0, 15}, {-8, 7}};
            product_cols(c, cols, sel, 4, true, kn, kr, sn + "sdot.p");
            for (int k = 0; k < 8; k++) cols[(size_t)k].push_back(CT{nib(Cv, k), 0, 15});
            sdotv = sum_columns(c, cols, 8, sn + "sdot");
        }
        // ---- reductions: the running value after this lane; in the final pass, the value as it stands
        if (cls & SM_CL_RED) {
            Expr rank;
            for (int t = 0; t < s; t++) rank += D[(size_t)t].is_red;
            Expr r1 = c.node(c.t_ge(rank, 1, 0, 5), sn + "red.rank1");
            red_rank1[(size_t)s] = r1;
            // the value before this lane: B (the last lane computed) when this pass computes a new lane, else A
            // the value before this lane: B (the last lane computed) when this pass computes a new lane, A on a lane's own
            // recomputation passes; in the final pass there is no lane before: B throughout (in_uni_p is read low: no depth)
            const Expr useB = or2(c, new_lane, in_uni_p, sn + "red.useB");  // LN_prev >= 32: a final-pass recomputation token
            L prev = mux_bits(c, r1, mux_bits(c, useB, redB[1], redA[1], sn + "red.p1"), mux_bits(c, useB, redB[0], redA[0], sn + "red.p0"), sn + "red.prev");
            L ident = mux_bits(c, d.has(SM_WMAX), const_bits(c, 0x80000000u, 32), const_bits(c, 0, 32), sn + "red.id");
            L base = mux_bits(c, ln_zero, ident, prev, sn + "red.base");
            L sumv = add_nibbles(c, base, A, 8, {}, sn + "red.sum");
            Adder cmpv = subn(c, A, base, 8, sn + "red.cmp");  // A - base
            Expr sa = A[31], sb = base[31], sd = cmpv.r[31];
            Expr a_gt = c.node(c.t_and(L{one - sa, sb}) + c.t_and(L{sa, sb, one - sd}) + c.t_and(L{one - sa, one - sb, one - sd}), sn + "red.gt");
            Expr neq = one - c.and_(nots(cmpv.r), sn + "red.eq");
            Expr take = c.and_(L{a_gt, neq}, sn + "red.take");
            L maxv2 = mux_bits(c, take, A, base, sn + "red.max");
            L comb = mux_bits(c, d.has(SM_WMAX), maxv2, sumv, sn + "red.comb");
            L after = mux_bits(c, on, comb, base, sn + "red.after");
            red_val[(size_t)s] = mux_bits(c, in_lanes, after, prev, sn + "red.val");
        }
        // ---- scalars: udiv (two restoring steps a pass), usqrt (one step a pass), the state in scratch words
        L udivv, usqrtv;
        if (cls & SM_CL_SCALAR) {
            L R0 = mux_bits(c, ln_zero, const_bits(c, 0, 33), div_r, sn + "div.r0");
            L N0 = mux_bits(c, ln_zero, A + Cv, div_n, sn + "div.n0");
            L Rk = R0, Nk = N0;
            for (int step = 0; step < 2; step++) {
                const std::string st = sn + "div" + std::to_string(step);
                L Rsh = L{Nk[63]} + sl(Rk, 0, 32);  // (R << 1) | N[63]
                Adder sb = subn(c, Rsh, Bv, 9, st + ".sub");
                Expr ge = sb.cout;
                Rk = mux_bits(c, ge, sl(sb.r, 0, 33), Rsh, st + ".r");
                Nk = L{ge} + sl(Nk, 0, 63);
            }
            Expr bz = c.and_(nots(Bv), sn + "div.bz");
            div_r_new = Rk;
            div_n_new = Nk;
            udivv = and_bits(c, sl(div_n, 0, 32), one - bz, sn + "div.q");
            L V0 = mux_bits(c, ln_zero, A + Cv, sq_v, sn + "sq.v0");
            L Rr0 = mux_bits(c, ln_zero, const_bits(c, 0, 33), sq_r, sn + "sq.r0");
            L bitv;
            for (int i = 0; i < 64; i++) bitv.push_back(i % 2 ? Expr() : eqc(c, LN5, (uint64_t)((62 - i) / 2), sn + "sq.bit" + std::to_string(i)));
            L rb = add_nibbles(c, zext(Rr0, 64), bitv, 16, {}, sn + "sq.rb");  // r + bit
            Adder vs = subn(c, V0, rb, 16, sn + "sq.sub");
            Expr ge = vs.cout;
            L Vn = mux_bits(c, ge, vs.r, V0, sn + "sq.v");
            L rhalf = sl(Rr0, 1, 33) + L{zero};
            L rplus = add_nibbles(c, zext(rhalf, 64), bitv, 16, {}, sn + "sq.rp");
            L Rn = mux_bits(c, ge, sl(rplus, 0, 33), rhalf, sn + "sq.r");
            sq_v_new = Vn;
            sq_r_new = Rn;
            usqrtv = sl(sq_r, 0, 32);
            scalar_write = c.and_(L{d.is_scalar, new_lane, in_lanes}, sn + "scalar.we");
        }
        // ---- the stage's result
        std::vector<Expr> sels;
        std::vector<L> vals;
        auto add_res = [&](int op, const L &v) {
            if (d.has(op).empty() || v.empty()) return;
            sels.push_back(d.has(op));
            vals.push_back(v);
        };
        add_res(SM_ADD, ada.r);
        add_res(SM_SUB, ads.r);
        add_res(SM_MAX, maxv);
        add_res(SM_MIN, minv);
        add_res(SM_SELP, selv);
        add_res(SM_TID, tidv);
        add_res(SM_SREG, sregv);
        add_res(SM_LD, ldv);
        add_res(SM_AND, logicv);
        add_res(SM_OR, logicv);
        add_res(SM_XOR, logicv);
        add_res(SM_SHL, shv);
        add_res(SM_SHR, shv);
        add_res(SM_CLZ, clzv);
        add_res(SM_MAD, madv);
        add_res(SM_MULHI, mulhiv);
        add_res(SM_MULSH, mulshv);
        add_res(SM_SUP, supv);
        add_res(SM_Q8DOT, dotv);
        add_res(SM_SDOT, sdotv);
        add_res(SM_WSUM, red_val[(size_t)s]);
        add_res(SM_WMAX, red_val[(size_t)s]);
        add_res(SM_UDIV, udivv);
        add_res(SM_USQRT, usqrtv);
        res[(size_t)s] = onehot_sel(c, sels, vals, 32, sn + "res");
        if (!d.is_scfg.empty()) scfg_new = mux_bits(c, d.is_scfg, A, scfg_new, sn + "scfg.new");
    }

    // ---------------------------------------------------------------- the branch predicate of this lane
    Expr pred_br;
    {
        Terms t;
        for (int s = 0; s < SM_STAGES; s++) t += c.t_and(lits_of(lstage, s) + L{cmp_res[(size_t)s]});
        pred_br = c.node(t, "br.pred");
    }
    const Expr any_p = scr[SCR_ANY][0], p0_p = scr[SCR_P0][0];

    // ---------------------------------------------------------------- the item values
    std::vector<L> item_val(SM_STAGES), uitem_val(SM_STAGES);
    for (int s = 0; s < SM_STAGES; s++) {
        const StageDec &d = D[(size_t)s];
        L v = mux_bits(c, d.is_stv, stv_val[(size_t)s], res[(size_t)s], "iv.s" + std::to_string(s));
        item_val[(size_t)s] = v;
        uitem_val[(size_t)s] = v;
    }
    L cur_val = onehot_sel(c, cur_stage, item_val, 32, "cur.val");
    L ucur_val = onehot_sel(c, ucur_stage, uitem_val, 32, "ucur.val");

    // the record ends after the last lane when there are no uniform items, or at the uniform part's VEND: the commit point
    // of everything uniform (the exit decides the next token at that pass too)
    const Expr cross = c.and_(L{lane_end, ln31_p}, "ex.cross");  // the lanes are done with this token
    const Expr rend = c.node(c.t_and(L{cross, nuni_zero}) + c.t_and(L{isVEND, in_uni_p, rec}), "ex.rend");

    // ---------------------------------------------------------------- the high writes: scratch, the state bank, the register files
    {
        Bank bk;
        bk.mem = MEM_SCR;
        auto word = [&](int w, const L &bits, const Expr &we) {
            BankWrite x;
            x.addr = const_bits(c, (uint64_t)w, 7);
            x.data = zext(bits, 32);
            x.data_taps.assign(32, LAG0);
            x.we = we;
            bk.writes.push_back(x);
        };
        word(SCR_LN, LN, one);
        word(SCR_LI, LI, one);
        // ANY over lanes 1..31: a new lane (not lane 0) with the predicate set; cleared at EXEC
        Expr any_step = c.and_(L{lane_end, in_lanes, pred_br}, "any.step");
        Expr any_new = c.node(c.t_and(L{one - isEXEC, any_p}) + c.t_and(L{one - isEXEC, any_step}) + c.t_and(L{one - isEXEC, any_p, any_step}, -1), "any.new");
        word(SCR_ANY, L{any_new}, one);
        word(SCR_P0, L{pred_br}, isEXEC);
        // the reductions' running values: at every new-lane pass in the lanes (rank 0, rank 1)
        for (int r = 0; r < 2; r++) {
            Terms t[32];
            Expr we;
            for (int s = 0; s < SM_STAGES; s++) {
                if (red_val[(size_t)s].empty()) continue;
                Expr sel = c.and_(L{D[(size_t)s].is_red, r ? red_rank1[(size_t)s] : one - red_rank1[(size_t)s]}, "red.w" + std::to_string(r) + ".s" + std::to_string(s));
                we += sel;
                for (int i = 0; i < 32; i++) t[i] += c.t_and(L{sel, red_val[(size_t)s][(size_t)i]});
            }
            L v;
            for (int i = 0; i < 32; i++) v.push_back(c.node(t[i], "red.w" + std::to_string(r) + ".b" + std::to_string(i)));
            const Expr wen = c.and_(L{we, new_lane, in_lanes}, "red.we" + std::to_string(r));
            word(r ? SCR_RED1B : SCR_RED0B, v, wen);          // after this lane
            word(r ? SCR_RED1A : SCR_RED0A, redB[r], wen);    // before it: what the last lane left
        }
        if (!div_r_new.empty()) {
            word(SCR_DIV_R0, sl(div_r_new, 0, 32), scalar_write);
            word(SCR_DIV_R1, L{div_r_new[32]}, scalar_write);
            word(SCR_DIV_N0, sl(div_n_new, 0, 32), scalar_write);
            word(SCR_DIV_N1, sl(div_n_new, 32, 64), scalar_write);
            word(SCR_SQ_V0, sl(sq_v_new, 0, 32), scalar_write);
            word(SCR_SQ_V1, sl(sq_v_new, 32, 64), scalar_write);
            word(SCR_SQ_R0, sl(sq_r_new, 0, 32), scalar_write);
            word(SCR_SQ_R1, L{sq_r_new[32]}, scalar_write);
        }
        // scfg commits at the final pass
        word(SCR_SCFG, scfg_new, cross);  // scfg is uniform: computed in the final pass (its computing pass), in force after the block
        // pending stores: lane slots refilled at a lane's computing pass, uniform slots at the final pass
        for (int i = 0; i < STV_SLOTS; i++) {
            const bool lane_slot = i < 4;
            const std::vector<Expr> &stg = lane_slot ? item_stage[(size_t)i] : uitem_stage[(size_t)(i - 4)];
            Terms ta[32], tv[32], tf[4];
            for (int s = 0; s < SM_STAGES; s++) {
                Expr sel = c.and_(L{stg[(size_t)s], D[(size_t)s].is_stv}, "pst" + std::to_string(i) + ".s" + std::to_string(s));
                for (int b = 0; b < 32; b++) {
                    ta[b] += c.t_and(L{sel, stv_addr[(size_t)s][(size_t)b]});
                    tv[b] += c.t_and(L{sel, stv_val[(size_t)s][(size_t)b]});
                }
                for (int b = 0; b < 3; b++) tf[b] += c.t_and(L{sel, D[(size_t)s].bank[(size_t)b]});
                tf[3] += c.t_and(L{sel, on_s[(size_t)s]});
            }
            L a, v, f;
            for (int b = 0; b < 32; b++) a.push_back(c.node(ta[b], "pst" + std::to_string(i) + ".a" + std::to_string(b)));
            for (int b = 0; b < 32; b++) v.push_back(c.node(tv[b], "pst" + std::to_string(i) + ".v" + std::to_string(b)));
            for (int b = 0; b < 4; b++) f.push_back(c.node(tf[b], "pst" + std::to_string(i) + ".f" + std::to_string(b)));
            // a lane slot is filled at the lane's computing pass, a uniform slot at the final part's computing pass only (its later
            // passes have seen their own stores land)
            const Expr we = lane_slot ? c.and_(L{new_lane, in_lanes}, "pst.we" + std::to_string(i)) : cross;
            word(SCR_STV0 + 3 * i, a, we);
            word(SCR_STV0 + 3 * i + 1, v, we);
            word(SCR_STV0 + 3 * i + 2, f, we);
        }
        // the state unit's pending writes: filled at the lane's computing pass, landed at the lane's end (below the datapath)
        for (int s = 0; s < SM_STAGES; s++) {
            if (sup_we[(size_t)s].empty()) continue;
            const Expr wen = c.and_(L{new_lane, in_lanes}, "psup.we" + std::to_string(s));
            word(SCR_SUP0 + 3 * s, sup_addr[(size_t)s], wen);
            word(SCR_SUP0 + 3 * s + 1, sl(sup_val[(size_t)s], 0, 16), wen);
            word(SCR_SUP0 + 3 * s + 2, L{sup_we[(size_t)s]}, wen);
        }
        // the uniform slots' flags are cleared when they fire
        for (int i = 4; i < STV_SLOTS; i++) word(SCR_STV0 + 3 * i + 2, L{}, uni_fire);
        c.bank("bank.scr.wr", bk);
    }
    {
        Bank bs;
        bs.mem = MEM_S;
        for (int s = 0; s < SM_STAGES; s++) {
            if (sup_we[(size_t)s].empty()) continue;
            BankWrite w;
            w.addr = scr[(size_t)(SCR_SUP0 + 3 * s)];
            w.data = sl(scr[(size_t)(SCR_SUP0 + 3 * s + 1)], 0, 16);
            w.data_taps.assign(16, LAG0);
            w.we = c.and_(L{lane_fire, scr[(size_t)(SCR_SUP0 + 3 * s + 2)][0]}, "psup.fire" + std::to_string(s));
            w.width = PW_HALF;
            bs.writes.push_back(w);
        }
        c.bank("bank.sup.pending", bs);
    }
    {
        // register commits from the payloads. A lane's items land in its registers only when the lane ends, the uniform
        // items when the uniform part ends: a part's later passes recompute it and must read the block's input state. An
        // item that ends with VHI waits in a pending word (its value; rd | valid << 4), the part's last item (VEND) lands
        // directly in the pass the pending ones fire; the pending words are cleared in that pass. A reduction's result
        // lands at the record's end from its after word.
        Bank rb, ub, pend;
        rb.mem = MEM_REG;
        ub.mem = MEM_UREG;
        pend.mem = MEM_SCR;
        auto rd_of = [&](const std::vector<Expr> &stg, const std::string &name) {
            L out;
            for (int b = 0; b < 4; b++) {
                Terms t;
                for (int s = 0; s < SM_STAGES; s++) t += c.t_and(L{stg[(size_t)s], D[(size_t)s].rd[(size_t)b]});
                out.push_back(c.node(t, name + ".rd" + std::to_string(b)));
            }
            return out;
        };
        Expr is_reg_item, is_ureg_item;
        {
            Terms t, tu;
            for (int s = 0; s < SM_STAGES; s++) {
                t += c.t_and(L{curp_stage[(size_t)s], D[(size_t)s].writes, one - D[(size_t)s].is_red, one - D[(size_t)s].uni});
                tu += c.t_and(L{ucurp_stage[(size_t)s], D[(size_t)s].writes, one - D[(size_t)s].is_red, D[(size_t)s].uni});
            }
            is_reg_item = c.node(t, "commit.isreg");
            is_ureg_item = c.node(tu, "commit.isureg");
        }
        const L rd_l = rd_of(curp_stage, "commit.l"), rd_u = rd_of(ucurp_stage, "commit.u");
        const Expr rd_l_zero = eqc(c, rd_l, SM_ZERO, "commit.lz"), rd_u_zero = eqc(c, rd_u, SM_ZERO, "commit.uz");
        const Expr lane_ok = c.and_(L{rec, in_lanes_p, is_reg_item, one - rd_l_zero}, "commit.lok");
        const Expr uni_ok = c.and_(L{rec, in_uni_p, is_ureg_item, one - rd_u_zero}, "commit.uok");
        const Expr uend = c.and_(L{isVEND, rec, in_uni_p}, "commit.uend");
        std::vector<Taps> val_taps(16, LAG1);
        val_taps.resize(32, LAG0);
        const std::vector<Taps> now32(32, LAG0);
        const L val = pVLO + pHI;  // the VLO payload one pass back, this pass's VHI/VEND payload
        auto pend_word = [&](int w, const L &bits, const std::vector<Taps> &taps, const Expr &we) {
            BankWrite x;
            x.addr = const_bits(c, (uint64_t)w, 7);
            x.data = zext(bits, 32);
            x.data_taps = taps;
            x.we = we;
            pend.writes.push_back(x);
        };
        for (int k = 0; k < 4; k++) {
            const std::string ks = std::to_string(k);
            const Expr lset = c.and_(L{isVHI, lane_ok, lip_is[(size_t)k]}, "pend.lset" + ks), uset = c.and_(L{isVHI, uni_ok, lip_is[(size_t)k]}, "pend.uset" + ks);
            pend_word(SCR_PREG0 + 2 * k, val, val_taps, lset);
            pend_word(SCR_PREG0 + 2 * k + 1, rd_l + L{one}, now32, lset);
            pend_word(SCR_PUREG0 + 2 * k, val, val_taps, uset);
            pend_word(SCR_PUREG0 + 2 * k + 1, rd_u + L{one}, now32, uset);
            const L &pv = scr[(size_t)(SCR_PREG0 + 2 * k)], &pm = scr[(size_t)(SCR_PREG0 + 2 * k + 1)];
            {
                BankWrite w;
                w.addr = sl(pm, 0, 4) + sl(LNp, 0, 5) + slot;
                w.data = pv;
                w.data_taps = now32;
                w.we = c.and_(L{lane_end, in_lanes_p, pm[4]}, "pend.lfire" + ks);
                rb.writes.push_back(w);
            }
            const L &uv = scr[(size_t)(SCR_PUREG0 + 2 * k)], &um = scr[(size_t)(SCR_PUREG0 + 2 * k + 1)];
            {
                BankWrite w;
                w.addr = sl(um, 0, 4) + slot;
                w.data = uv;
                w.data_taps = now32;
                w.we = c.and_(L{uend, um[4]}, "pend.ufire" + ks);
                ub.writes.push_back(w);
            }
            pend_word(SCR_PREG0 + 2 * k + 1, L{}, now32, lane_end);
            pend_word(SCR_PUREG0 + 2 * k + 1, L{}, now32, uend);
        }
        {
            BankWrite w;
            w.addr = rd_l + sl(LNp, 0, 5) + slot;
            w.data = val;
            w.data_taps = val_taps;
            w.we = c.and_(L{isVEND, lane_ok}, "commit.we.l");
            rb.writes.push_back(w);
        }
        {
            BankWrite w;
            w.addr = rd_u + slot;
            w.data = val;
            w.data_taps = val_taps;
            w.we = c.and_(L{isVEND, uni_ok}, "commit.we.u");
            ub.writes.push_back(w);
        }
        for (int s = 0; s < SM_STAGES; s++) {
            if (red_val[(size_t)s].empty()) continue;
            BankWrite w;
            w.addr = D[(size_t)s].rd + slot;
            w.data = mux_bits(c, red_rank1[(size_t)s], redB[1], redB[0], "commit.red" + std::to_string(s));
            w.data_taps = now32;
            w.we = c.and_(L{rend, red_live[(size_t)s], one - eqc(c, D[(size_t)s].rd, SM_ZERO, "commit.redz" + std::to_string(s))}, "commit.red.we" + std::to_string(s));
            ub.writes.push_back(w);
        }
        c.bank("bank.scr.pending", pend);
        c.bank("bank.reg.wr", rb);
        c.bank("bank.ureg.wr", ub);
    }

    // ---------------------------------------------------------------- the exit: the next token
    std::vector<SelCase> fam_cases, pay_cases;
    auto cs0 = [&](const L &lits, int fid) { fam_cases.push_back(SelCase{lits, true, fid, {}}); };
    auto csb = [&](const L &lits, int fid, const L &bits) {
        fam_cases.push_back(SelCase{lits, true, fid, {}});
        pay_cases.push_back(SelCase{lits, false, 0, bits});
    };
    const L cur_lo = sl(cur_val, 0, 16), cur_hi = sl(cur_val, 16, 32), ucur_lo = sl(ucur_val, 0, 16), ucur_hi = sl(ucur_val, 16, 32);
    // A: EXEC -> lane 0's first item, or LSTEP
    cs0(L{isEXEC, nparts_zero}, F_LSTEP);
    csb(L{isEXEC, one - nparts_zero}, F_VLO, cur_lo);
    // B, C: within a lane's part
    csb(L{isVLO, rec, in_lanes, one - last_item}, F_VHI, cur_hi);
    csb(L{isVLO, rec, in_lanes, last_item}, F_VEND, cur_hi);
    csb(L{isVHI, rec, in_lanes}, F_VLO, cur_lo);
    // D: a lane ended, more lanes follow
    cs0(L{lane_end, in_lanes, nparts_zero}, F_LSTEP);
    csb(L{lane_end, in_lanes, one - nparts_zero}, F_VLO, cur_lo);
    // E: the last lane ended: the uniform items, or the record's end
    csb(L{cross, one - nuni_zero}, F_VLO, ucur_lo);
    // F, G: the uniform part
    csb(L{isVLO, rec, in_uni, one - ulast_item}, F_VHI, ucur_hi);
    csb(L{isVLO, rec, in_uni, ulast_item}, F_VEND, ucur_hi);
    csb(L{isVHI, rec, in_uni}, F_VLO, ucur_lo);
    // the record's end: after the last lane with no uniform items, or after the uniform VEND
    L b1 = add_nibbles(c, zext(B, 12), const_bits(c, 1, 12), 3, {}, "ex.b1");
    Expr any_all = or2(c, any_p, p0_p, "ex.anyall");
    Expr pred_sel = c.node(c.t_and(L{lany, any_all}) + c.t_and(L{one - lany, p0_p}), "ex.predsel");
    Expr taken = c.node(c.t_and(L{pred_sel, one - lneg}) + c.t_and(L{one - pred_sel, lneg}), "ex.taken");
    L exec_next = L{} + b1 + L{zero};  // EXEC(b + 1, 0): 13 bits
    L exec_tgt = ltarget + L{zero};
    csb(L{rend, br_is[SM_BR_NEXT]}, F_EXEC, exec_next);
    csb(L{rend, br_is[SM_BR_JUMP]}, F_EXEC, exec_tgt);
    csb(L{rend, br_is[SM_BR_COND], taken}, F_EXEC, exec_tgt);
    csb(L{rend, br_is[SM_BR_COND], one - taken}, F_EXEC, exec_next);
    // END, BAR: the warp changes, or the launch ends
    std::vector<Expr> lw(4);
    for (int k = 0; k < 4; k++) lw[(size_t)k] = eqc(c, LWPB, (uint64_t)k, "ex.lwpb" + std::to_string(k));
    Expr last_warp = c.node(c.t_and(L{lw[0]}) + c.t_and(L{lw[1], W[0]}) + c.t_and(L{lw[2], W[0], W[1]}) + c.t_and(L{lw[3], W[0], W[1], W[2]}), "ex.lastwarp");
    L w1b = add_nibbles(c, zext(W, 12), const_bits(c, 1, 12), 3, {}, "ex.w1");  // w + 1, 9 bits matter
    // warps9 = WARPS, or 256 when WARPS == 0
    Expr warps0 = eqc(c, WARPS, 0, "ex.warps0");
    L warps9 = WARPS + L{warps0};
    Adder wcmp = subn(c, sl(w1b, 0, 9), warps9, 3, "ex.wcmp");  // w1 - warps: cout = w1 >= warps
    Expr over = wcmp.cout;
    L q1 = add_nibbles(c, zext(Q, 12), const_bits(c, 1, 12), 3, {}, "ex.q1");
    csb(L{rend, br_is[SM_BR_END], last_warp, over}, F_NEXT, q1);
    csb(L{rend, br_is[SM_BR_END], last_warp, one - over}, F_WARP, sl(w1b, 0, 8));
    csb(L{rend, br_is[SM_BR_END], one - last_warp}, F_WARP, sl(w1b, 0, 8));
    // BAR: the last warp of the thread block goes back wpb - 1 (its low lwpb bits cleared), the others go on by one
    L wbar;
    for (int b = 0; b < 8; b++) {
        Terms t;
        for (int k = 0; k < 4; k++)
            if (b >= k) t += c.t_and(L{lw[(size_t)k], W[(size_t)b]});
        wbar.push_back(c.node(t, "ex.wbar" + std::to_string(b)));
    }
    csb(L{rend, br_is[SM_BR_BAR], last_warp}, F_WARP, wbar);
    csb(L{rend, br_is[SM_BR_BAR], one - last_warp}, F_WARP, sl(w1b, 0, 8));
    // I: after WARP: the new warp starts a phase when its index is a multiple of wpb
    Expr wlow0 = c.node(c.t_and(L{lw[0]}) + c.t_and(L{lw[1], one - W[0]}) + c.t_and(L{lw[2], one - W[0], one - W[1]}) + c.t_and(L{lw[3], one - W[0], one - W[1], one - W[2]}), "ex.wlow0");
    csb(L{isWARP, br_is[SM_BR_END], wlow0}, F_EXEC, ENTRY + L{one});
    csb(L{isWARP, br_is[SM_BR_END], one - wlow0}, F_EXEC, R + L{zero});
    csb(L{isWARP, br_is[SM_BR_BAR], wlow0}, F_EXEC, b1 + L{one});
    csb(L{isWARP, br_is[SM_BR_BAR], one - wlow0}, F_EXEC, R + L{zero});
    // J: the queue
    std::vector<Expr> kind_is(4);
    for (int k = 0; k < 4; k++) kind_is[(size_t)k] = eqc(c, sl(w0, 0, 2), (uint64_t)k, "ex.kind" + std::to_string(k));
    L k1 = add_nibbles(c, zext(sl(pNEXT, 0, 12), 12), const_bits(c, 1, 12), 3, {}, "ex.k1");
    Expr mbr_nz = one - c.and_(nots(mbr), "ex.mbrz");
    csb(L{isNEXT, kind_is[SM_Q_LAUNCH]}, F_EXEC, sl(w0, 2, 14) + L{one});
    csb(L{isNEXT, kind_is[SM_Q_BRANCH], mbr_nz}, F_NEXT, sl(w0, 24, 32) + sl(w1, 24, 28));
    csb(L{isNEXT, kind_is[SM_Q_BRANCH], one - mbr_nz}, F_NEXT, k1);
    cs0(L{isNEXT, kind_is[SM_Q_EMIT]}, F_EMIT);
    cs0(L{isNEXT, kind_is[SM_Q_HALT]}, F_HALT);
    // K, L, M: the emit
    csb(L{isEMIT}, F_VLO, sl(memit, 0, 16));
    csb(L{isVLO, in_emit}, F_VEND, sl(memit, 16, 32));
    csb(L{isVEND, in_emit}, F_NEXT, q1);
    // N: idle
    cs0(L{isHALT}, F_HALT);
    csb(L{isGO}, F_NEXT, q1);
    cs0(L{isTEXT}, F_TEXT);

    c.out_fam = select_bits(c, fam_cases, NFAM, "exit.fam");
    c.out_pay = select_bits(c, pay_cases, NPAY, "exit.pay");
    int exit_min = 0;
    for (const Expr &b : c.out_fam) exit_min = std::max(exit_min, c.level_of(b));
    for (const Expr &b : c.out_pay) exit_min = std::max(exit_min, c.level_of(b));
    exit_min -= 1;
    for (const Expr &e : c.code) {
        Terms t{Term{Dy(-1), one, e}, Term{Dy(1), one, Dy(-1) * e}};
        c.ffn(t, "code.clear", &e);
    }
    for (int i = 0; i < NFAM + NPAY; i++) {
        const Expr &e = c.code[(size_t)i];
        const std::vector<SelCase> &cs = i < NFAM ? fam_cases : pay_cases;
        const int bit = i < NFAM ? i : i - NFAM;
        Terms terms = c.t_const(Dy(-KAPPA));
        const Dy twoK = Dy(2) * Dy(KAPPA);
        for (const SelCase &k : cs) {
            if (k.is_int) {
                if (k.ival >> bit & 1) terms += c.t_and(k.lits, twoK);
            } else if (bit < (int)k.bits.size() && !k.bits[(size_t)bit].empty()) {
                terms += c.t_and(k.lits + L{k.bits[(size_t)bit]}, twoK);
            }
        }
        c.ffn(terms, "exit.code" + std::to_string(i), &e, exit_min);
    }
    for (const Expr &e : c.code) c.keep.push_back(e.only_dim());
    (void)llwpb;
    (void)lip_is;
    reschedule(c, c.target.intermediate, c.target.key_heads);
    sched_check(c);
    if (stats) {
        stats->nodes = c.nodes.size();
        for (const Node &n : c.nodes) {
            if (n.is_bank) stats->banks++;
            else if (n.is_ffn) stats->units += n.terms.size();
            else stats->heads++;
        }
        stats->dims = c.dims.size();
        stats->layers = c.n_layers();
    }
    return cp;
}
