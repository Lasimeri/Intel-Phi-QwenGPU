// q2b_build MODEL.gguf KERNELS.ptx OUTDIR: the GPU image of the 2B. Lowers every kernel the program needs (one ROM
// copy per distinct set of baked constants), lays out M (the queue and its args blocks, the job words, the scalars,
// the weight table, the function tables, the constants of every layer, the RoPE table), writes W as the GGUF's own
// Q8_0 blocks concatenated in Q2B_MAT order, and writes out/gpu/{rom.bin, m.bin, w.q8, meta.txt}. Host. The 2B's
// weights are not changed: bytes move, values do not.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "../gpu/lower.h"
#include "../guest/q2b_map.h"
#include "../text/gguf.h"
#include "../text/q2b_fix.h"
#include "../text/q35.h"

namespace {

[[noreturn]] void die(const std::string &what) {
    std::fprintf(stderr, "q2b_build: %s\n", what.c_str());
    std::exit(2);
}

using lower::ParamSpec;
ParamSpec C(int64_t v) {
    ParamSpec p;
    p.kind = ParamSpec::CONST;
    p.value = v;
    return p;
}
ParamSpec P(int64_t word) {
    ParamSpec p;
    p.kind = ParamSpec::PTR;
    p.value = word;
    return p;
}
ParamSpec A(int k) {
    ParamSpec p;
    p.kind = ParamSpec::ARG;
    p.value = k;
    return p;
}

struct Builder {
    std::map<std::string, ptx::Kernel> kernels;
    std::vector<SmBlock> rom;
    std::map<std::string, uint32_t> inst_base;  // key -> ROM block
    std::map<std::string, std::string> inst_list;
    std::vector<int32_t> m;
    std::vector<std::pair<uint32_t, uint32_t>> queue;  // (word0, word1)
    std::vector<std::string> qnames;                   // what each descriptor is, for queue.txt
    struct Check {
        uint32_t d;
        std::string tap;
        int layer;
        uint32_t word, count;
    };
    std::vector<Check> checks;  // after descriptor d, M[word..word+count) = q2b_fix's tensor `tap` of layer `layer`
    struct LaunchRec {  // a launch as program.txt lists it: what an OpenCL host replays through the API
        std::string name;
        int gx, bx;
        std::vector<ParamSpec> ps;
        std::vector<int32_t> args;
    };
    std::map<uint32_t, LaunchRec> recs;  // by descriptor
    uint64_t instrs = 0;
    void check(const std::string &tap, int layer, uint32_t word, uint32_t count) { checks.push_back({(uint32_t)queue.size() - 1, tap, layer, word, count}); }

    std::string key_of(const std::string &name, const std::vector<ParamSpec> &ps) {
        std::string k = name;
        for (const ParamSpec &p : ps) k += p.kind == ParamSpec::CONST ? ":c" : p.kind == ParamSpec::PTR ? ":p" : ":a", k += std::to_string(p.value);
        return k;
    }
    // an instance: lowered once per distinct baked parameters
    uint32_t instance(const std::string &name, int gx, int bx, const std::vector<ParamSpec> &ps, lower::Launch *out = nullptr) {
        std::string key = key_of(name, ps) + ":g" + std::to_string(gx) + ":b" + std::to_string(bx);
        auto it = inst_base.find(key);
        if (!kernels.count(name)) die("no kernel " + name);
        lower::Options opt;
        sm_classes_env(opt.classes);
        lower::Launch L = lower::lower(kernels[name], gx, 1, bx, 1, ps, opt);
        if (out) *out = L;
        if (it != inst_base.end()) return it->second;
        uint32_t base = (uint32_t)rom.size();
        lower::encode(L, rom, base);
        inst_base[key] = base;
        inst_list[key] = lower::listing(L);
        instrs += L.stats_instrs;
        return base;
    }
    // a launch descriptor with its args block
    void launch(const std::string &name, int gx, int bx, const std::vector<ParamSpec> &ps, const std::vector<int32_t> &args = {}) {
        lower::Launch L;
        uint32_t base = instance(name, gx, bx, ps, &L);
        // the ROM copy's warps / wpb are the launch's: the grid is part of the instance key
        uint32_t d = (uint32_t)queue.size();
        if (args.size() > M_ARGS_WORDS) die(name + ": too many args");
        uint32_t ab = (uint32_t)M_ARGS + d * (uint32_t)M_ARGS_WORDS;
        for (size_t i = 0; i < args.size(); i++) m[ab + i] = args[i];
        queue.emplace_back(sm_qword(SM_Q_LAUNCH, base, L.warps, (uint32_t)L.lwpb, 0), ab);
        qnames.push_back(name);
        recs[d] = LaunchRec{name, gx, bx, ps, args};
    }
    uint32_t branch(uint32_t word, uint32_t target) {
        uint32_t d = (uint32_t)queue.size();
        queue.emplace_back(sm_qword(SM_Q_BRANCH, 0, 0, 0, target & 255), (word & 0xFFFFFF) | (target >> 8) << 24);
        qnames.push_back("branch");
        return d;
    }
    void patch_branch(uint32_t d, uint32_t target) {
        queue[d].first = sm_qword(SM_Q_BRANCH, 0, 0, 0, target & 255);
        queue[d].second = (queue[d].second & 0xFFFFFF) | (target >> 8) << 24;
    }
    void emit(uint32_t word) {
        queue.emplace_back(sm_qword(SM_Q_EMIT, 0, 0, 0, 0), word);
        qnames.push_back("emit");
    }
    void halt() {
        queue.emplace_back(sm_qword(SM_Q_HALT, 0, 0, 0, 0), 0);
        qnames.push_back("halt");
    }
};

}  // namespace

int main(int argc, char **argv) {
    if (argc < 4) {
        std::fputs("usage: q2b_build MODEL.gguf KERNELS.ptx OUTDIR\n", stderr);
        return 2;
    }
    const std::string model = argv[1], ptxpath = argv[2], outdir = argv[3];
    gguf::File f;
    gguf::open(f, model.c_str());
    Builder B;
    B.kernels = ptx::load(ptxpath);
    B.m.assign((size_t)M_END, 0);
    std::printf("host: M is %u words (%.1f MB), A %u blocks (%.1f MB), S %u elements\n", (unsigned)M_END, M_END * 4.0 / 1e6, (unsigned)A_END, A_END * 36.0 / 1e6,
                (unsigned)Q2B_S_ELEMS);

    // ------------------------------------------------------------------ W: the Q8_0 blocks, and the table in M
    std::vector<uint32_t> wbase((size_t)Q2B_NMATS, 0), wblocks((size_t)Q2B_NMATS, 0);
    std::vector<uint64_t> wrows((size_t)Q2B_NMATS, 0);
    {
        const std::string wpath = outdir + "/w.q8";
        FILE *wf = std::fopen(wpath.c_str(), "wb");
        if (!wf) die("cannot create " + wpath);
        uint64_t at = 0;
        std::vector<uint8_t> buf(16u << 20);
        auto put_tensor = [&](int mi, const std::string &name) {
            if (!f.has(name)) return;
            const gguf::Tensor &t = f.tensor(name);
            if (t.type != gguf::T_Q8_0) die(name + " is not Q8_0");
            const uint64_t cols = t.dims[0], rows = t.dims[1];
            if (cols % 32) die(name + ": columns not a multiple of 32");
            if (at % 34) die("W is not block aligned");
            wbase[(size_t)mi] = (uint32_t)(at / 34);
            wblocks[(size_t)mi] = (uint32_t)(cols / 32);
            wrows[(size_t)mi] = rows;
            for (uint64_t off = 0; off < t.nbytes;) {
                const uint64_t n = std::min<uint64_t>(buf.size(), t.nbytes - off);
                gguf::read_at(f, t.offset + off, buf.data(), n);
                if (std::fwrite(buf.data(), 1, n, wf) != n) die("write w.q8");
                off += n;
                at += n;
            }
        };
        put_tensor(MAT_EMB, "token_embd.weight");
        for (int il = 0; il < Q2B_LAYERS; il++) {
            const std::string p = "blk." + std::to_string(il) + ".";
            put_tensor(q2b_mat_index(il, MAT_QKV), p + "attn_qkv.weight");
            put_tensor(q2b_mat_index(il, MAT_GATE), p + "attn_gate.weight");
            put_tensor(q2b_mat_index(il, MAT_ALPHA), p + "ssm_alpha.weight");
            put_tensor(q2b_mat_index(il, MAT_BETA), p + "ssm_beta.weight");
            put_tensor(q2b_mat_index(il, MAT_OUT), p + "ssm_out.weight");
            put_tensor(q2b_mat_index(il, MAT_Q), p + "attn_q.weight");
            put_tensor(q2b_mat_index(il, MAT_K), p + "attn_k.weight");
            put_tensor(q2b_mat_index(il, MAT_V), p + "attn_v.weight");
            put_tensor(q2b_mat_index(il, MAT_O), p + "attn_output.weight");
            put_tensor(q2b_mat_index(il, MAT_FG), p + "ffn_gate.weight");
            put_tensor(q2b_mat_index(il, MAT_FU), p + "ffn_up.weight");
            put_tensor(q2b_mat_index(il, MAT_FD), p + "ffn_down.weight");
        }
        std::fclose(wf);
        std::printf("host: W %.3f GB, %llu blocks\n", at / 1e9, (unsigned long long)(at / 34));
        for (int mi = 0; mi < Q2B_NMATS; mi++) {
            B.m[(size_t)M_WBASE + 2 * (size_t)mi] = (int32_t)wbase[(size_t)mi];
            B.m[(size_t)M_WBASE + 2 * (size_t)mi + 1] = (int32_t)wblocks[(size_t)mi];
        }
    }

    // ------------------------------------------------------------------ M: the tables and constants
    {
        const std::vector<int32_t> *tabs[4] = {&q2b::fix_table(0), &q2b::fix_table(1), &q2b::fix_table(2), &q2b::fix_table(3)};
        const uint32_t at[4] = {M_SILU, M_SIG, M_SP, M_EXP};
        const size_t n[4] = {Q2B_SILU_N, Q2B_SIG_N, Q2B_SP_N, Q2B_EXP_N};
        for (int t = 0; t < 4; t++) {
            if (tabs[t]->size() != n[t]) die("table size");
            std::memcpy(&B.m[at[t]], tabs[t]->data(), n[t] * 4);
        }
        std::vector<int32_t> on = q2b::fix_read_q16(f, "output_norm.weight");
        if (on.size() != Q2B_D) die("output_norm");
        std::memcpy(&B.m[M_OUTNORM], on.data(), Q2B_D * 4);
        for (int il = 0; il < Q2B_LAYERS; il++) {
            const std::string p = "blk." + std::to_string(il) + ".";
            const uint32_t L = q2b_layer_m(il);
            std::vector<int32_t> an = q2b::fix_read_q16(f, p + "attn_norm.weight"), pn = q2b::fix_read_q16(f, p + "post_attention_norm.weight");
            if (q2b_is_full(il)) {
                std::memcpy(&B.m[L + LA_ATTN], an.data(), Q2B_D * 4);
                std::memcpy(&B.m[L + LA_POST], pn.data(), Q2B_D * 4);
                std::vector<int32_t> qn = q2b::fix_read_q16(f, p + "attn_q_norm.weight"), kn = q2b::fix_read_q16(f, p + "attn_k_norm.weight");
                if (qn.size() != Q2B_HD || kn.size() != Q2B_HD) die("q/k norm size");
                std::memcpy(&B.m[L + LA_QN], qn.data(), Q2B_HD * 4);
                std::memcpy(&B.m[L + LA_KN], kn.data(), Q2B_HD * 4);
            } else {
                std::memcpy(&B.m[L + LG_ATTN], an.data(), Q2B_D * 4);
                std::memcpy(&B.m[L + LG_POST], pn.data(), Q2B_D * 4);
                std::vector<int32_t> conv = q2b::fix_read_q16(f, p + "ssm_conv1d.weight");  // channel-major: w[c * 4 + t]
                if (conv.size() != (size_t)Q2B_QKV * 4) die("conv size");
                for (int c = 0; c < Q2B_QKV; c++)
                    for (int t = 0; t < 4; t++) B.m[L + LG_CONV + (uint32_t)t * Q2B_QKV + (uint32_t)c] = conv[(size_t)c * 4 + (size_t)t];
                std::vector<int32_t> a = q2b::fix_read_q16(f, p + "ssm_a"), dt = q2b::fix_read_q16(f, p + "ssm_dt.bias"), sn = q2b::fix_read_q16(f, p + "ssm_norm.weight");
                if (a.size() != Q2B_VHEADS || dt.size() != Q2B_VHEADS || sn.size() != Q2B_ST) die("gdn constants");
                std::memcpy(&B.m[L + LG_AV], a.data(), Q2B_VHEADS * 4);
                std::memcpy(&B.m[L + LG_DT], dt.data(), Q2B_VHEADS * 4);
                std::memcpy(&B.m[L + LG_NORM], sn.data(), Q2B_ST * 4);
                for (int hd = 0; hd < Q2B_VHEADS; hd++) B.m[L + LG_SE + (uint32_t)hd] = 15;
            }
        }
        // RoPE: position x (32 cos, 32 sin), Q.15
        for (uint32_t pos = 0; pos < Q2B_CTX; pos++) {
            float cf[Q2B_ROT / 2], sf[Q2B_ROT / 2];
            q35::rope_row(pos, cf, sf);
            for (int i = 0; i < Q2B_ROT / 2; i++) {
                B.m[M_ROPE + pos * Q2B_ROT + (uint32_t)i] = (int32_t)std::lrint((double)cf[i] * 32768.0);
                B.m[M_ROPE + pos * Q2B_ROT + 32 + (uint32_t)i] = (int32_t)std::lrint((double)sf[i] * 32768.0);
            }
        }
        B.m[JOB_ONE] = 1;
    }

    // ------------------------------------------------------------------ the program: one position per pass of the queue
    auto norm_chain = [&](uint32_t w) {
        B.launch("norm_a", 64, 32, {P(M_X)});
        B.launch("norm_b", 1, 32, {});
        B.launch("norm_c", 64, 32, {P(M_X)});
        B.launch("norm_d", 1, 32, {C(Q2B_D)});
        B.launch("norm_e", 64, 32, {P(M_X), A(0), P(M_H)}, {(int32_t)w});
    };
    auto gemv_h = [&](int mi, uint32_t y, int rows, int ablk) {
        B.launch("gemv_h64", rows / 32, 32, {A(0), C(ablk), A(1)}, {(int32_t)wbase[(size_t)mi], (int32_t)y});
    };
    B.launch("embed", 64, 32, {P(M_X), C(wbase[MAT_EMB]), P(M_TOKS)});
    B.check("inp_embd", -1, M_X, Q2B_D);
    for (int il = 0; il < Q2B_LAYERS; il++) {
        const uint32_t L = q2b_layer_m(il);
        const bool full = q2b_is_full(il);
        norm_chain(L + (full ? LA_ATTN : LG_ATTN));
        B.check("attn_norm", il, M_H, Q2B_D);
        B.launch("quant_blocks", 64, 32, {A(0), A(1)}, {(int32_t)M_H, (int32_t)A_H});
        if (!full) {
            B.launch("gemv_ring64", Q2B_QKV / 32, 32, {A(0), C(A_H), A(1)}, {(int32_t)wbase[(size_t)q2b_mat_index(il, MAT_QKV)], (int32_t)(L + LG_WIN)});
            B.check("linear_attn_qkv_mixed", il, L + LG_WIN, Q2B_QKV);  // slot 0; the probe adds the position's slot
            gemv_h(q2b_mat_index(il, MAT_GATE), M_Z, Q2B_VAL_DIM, A_H);
            B.check("z", il, M_Z, Q2B_VAL_DIM);
            gemv_h(q2b_mat_index(il, MAT_ALPHA), M_ALPHA, 32, A_H);
            gemv_h(q2b_mat_index(il, MAT_BETA), M_BETA, 32, A_H);
            B.launch("gdn_gates", 1, 32, {P(M_ALPHA), P(M_BETA), A(0), A(1), P(M_DECAY), P(M_BETAS)}, {(int32_t)(L + LG_DT), (int32_t)(L + LG_AV)});
            B.check("gdn_decay", il, M_DECAY, Q2B_VHEADS);
            B.check("gdn_beta", il, M_BETAS, Q2B_VHEADS);
            B.launch("conv_silu", Q2B_QKV / 32, 32, {A(0), A(1), P(M_CONV)}, {(int32_t)(L + LG_WIN), (int32_t)(L + LG_CONV)});
            B.check("conv_output_silu", il, M_CONV, Q2B_QKV);
            B.launch("quant_vec128_unit", 16, 128, {A(0), A(1), A(2), A(3)}, {(int32_t)M_CONV, (int32_t)M_QI, (int32_t)M_QS, 1});
            B.launch("quant_vec128_unit", 16, 128, {A(0), A(1), A(2), A(3)}, {(int32_t)(M_CONV + Q2B_KEY_DIM), (int32_t)M_KI, (int32_t)M_KS, 0});
            B.launch("quant_vec128_scale", 16, 128, {P(M_CONV + 2 * Q2B_KEY_DIM), P(M_VI), P(M_VS)});
            B.check("gdn_qi", il, M_QI, Q2B_KEY_DIM);
            B.check("gdn_ki", il, M_KI, Q2B_KEY_DIM);
            B.check("gdn_vi", il, M_VI, Q2B_VAL_DIM);
            B.check("gdn_qs", il, M_QS, Q2B_KHEADS);
            B.check("gdn_ks", il, M_KS, Q2B_KHEADS);
            B.check("gdn_vs", il, M_VS, Q2B_VHEADS);
            const int gl = q2b_gdn_index(il);
            B.launch("gdn_head", 16, 128,
                     {A(0), A(1), A(2), P(M_KI), P(M_QI), P(M_VI), P(M_DECAY), P(M_BETAS), P(M_KS), P(M_QS), P(M_VS), P(M_Z), A(3), P(M_MIX), P(M_KREAL), P(M_O), P(M_GDBG), P(M_OQ), P(M_SZ)},
                     {(int32_t)q2b_s_row(gl, 0, 0), (int32_t)(L + LG_SE), (int32_t)(L + LG_SMAX), (int32_t)(L + LG_NORM)});
            B.check("gdn_kreal", il, M_KREAL, Q2B_VHEADS * Q2B_ST);
            B.check("gdn_o", il, M_O, Q2B_VHEADS * Q2B_ST);
            B.check("gdn_dbg", il, M_GDBG, Q2B_VHEADS * 8);
            B.check("gdn_on", il, M_OQ, Q2B_VHEADS * Q2B_ST);
            B.check("gdn_sz", il, M_SZ, Q2B_VHEADS * Q2B_ST);
            B.check("gdn_se", il, L + LG_SE, Q2B_VHEADS);
            B.check("gdn_smax", il, L + LG_SMAX, Q2B_VHEADS);
            B.check("gdn_state", il, (uint32_t)q2b_s_row(gl, 0, 0), Q2B_VHEADS * Q2B_ST * Q2B_ST);  // the S bank, int16
            B.check("final_output", il, M_MIX, Q2B_VAL_DIM);
            B.launch("quant_blocks", 64, 32, {A(0), A(1)}, {(int32_t)M_MIX, (int32_t)A_MIX});
            B.check("mix_q", il, A_MIX, Q2B_VAL_DIM);       // A8: the int8 lanes of the A blocks from A_MIX
            B.check("mix_xs", il, A_MIX, Q2B_VAL_DIM / 32); // A32: their Q.24 scales
            gemv_h(q2b_mat_index(il, MAT_OUT), M_Y, Q2B_D, A_MIX);
            B.check("linear_attn_out", il, M_Y, Q2B_D);
        } else {
            const int al = q2b_att_index(il);
            gemv_h(q2b_mat_index(il, MAT_Q), M_QG, 2 * Q2B_QHEADS * Q2B_HD, A_H);
            gemv_h(q2b_mat_index(il, MAT_K), M_KK, Q2B_KVHEADS * Q2B_HD, A_H);
            gemv_h(q2b_mat_index(il, MAT_V), M_VV, Q2B_KVHEADS * Q2B_HD, A_H);
            B.launch("norm256", Q2B_QHEADS, 256, {A(0), A(1), A(2), A(3)}, {(int32_t)M_QG, 2 * Q2B_HD, (int32_t)(L + LA_QN), (int32_t)M_QV});
            B.launch("norm256", Q2B_KVHEADS, 256, {A(0), A(1), A(2), A(3)}, {(int32_t)M_KK, Q2B_HD, (int32_t)(L + LA_KN), (int32_t)M_KK});
            B.launch("rotate", 1, 256, {A(0), P(M_ROPE)}, {(int32_t)M_QV});
            B.check("attn_query", il, M_QV, Q2B_QHEADS * Q2B_HD);  // not *_q: that suffix means an A8 compare to the probe
            B.launch("rotate", 1, 64, {A(0), P(M_ROPE)}, {(int32_t)M_KK});
            B.check("attn_k", il, M_KK, Q2B_KVHEADS * Q2B_HD);
            B.launch("quant_vec256", Q2B_QHEADS, 256, {P(M_QV), C(A_Q), P(M_QSCALE)});
            B.check("aq_q", il, A_Q, Q2B_QHEADS * Q2B_HD);       // A8 from A_Q
            B.check("aq_xs", il, A_Q, Q2B_QHEADS * Q2B_HD / 32); // A32
            B.launch("quant_kc", Q2B_KVHEADS, 256, {P(M_KK), A(0)}, {(int32_t)q2b_kc_block(al, 0, 0)});
            B.check("kc0_q", il, q2b_kc_block(al, 0, 0), Q2B_HD);  // the probe adds this position's 8 blocks
            B.check("kc0_xs", il, q2b_kc_block(al, 0, 0), Q2B_HD / 32);
            B.check("kc1_q", il, q2b_kc_block(al, 1, 0), Q2B_HD);
            B.check("kc1_xs", il, q2b_kc_block(al, 1, 0), Q2B_HD / 32);
            B.launch("quant_vt", Q2B_KVHEADS, 256, {P(M_VV), A(0), A(1), A(2)}, {(int32_t)q2b_vt_block(al, 0, 0, 0), (int32_t)(L + LA_VSC), (int32_t)(L + LA_VSMAX)});
            B.check("vt0_q", il, q2b_vt_block(al, 0, 0, 0), Q2B_HD);  // V^T: dim i at block + i * CTX/32, lane pos % 32
            B.check("vt1_q", il, q2b_vt_block(al, 1, 0, 0), Q2B_HD);
            B.check("attn_vs", il, L + LA_VSC, Q2B_KVHEADS);  // the probe adds pos * 2
            B.check("attn_vsmax", il, L + LA_VSMAX, Q2B_KVHEADS);
            for (int hq = 0; hq < Q2B_QHEADS; hq++) {
                const int kvh = hq * Q2B_KVHEADS / Q2B_QHEADS;
                const bool last = hq == Q2B_QHEADS - 1;  // the reference taps the last head's scores, softmax and F blocks
                B.launch("attn_scores", 1, 256, {A(0), A(1), A(2), P(M_P)}, {(int32_t)q2b_kc_block(al, 0, 0), (int32_t)(A_Q + hq * 8), kvh});
                if (last) {
                    B.check("attn_p", il, M_P, Q2B_CTX);
                    B.check("attn_top", il, SC_TOP, 1);
                }
                B.launch("attn_softmax", 1, 256, {P(M_P), P(M_EXPV)});
                if (last) {
                    B.check("attn_e", il, M_EXPV, Q2B_CTX);
                    B.check("attn_sum", il, SC_SUM, 1);
                }
                B.launch("attn_gq", 1, 256, {P(M_EXPV), A(0), A(1), C(A_F), C(A_F2), A(2)}, {(int32_t)(L + LA_VSC), kvh, (int32_t)(L + LA_VSMAX)});
                if (last) {
                    B.check("f_q", il, A_F, Q2B_CTX);
                    B.check("f_xs", il, A_F, Q2B_CTX / 32);
                    B.check("f2_q", il, A_F2, Q2B_CTX);
                    B.check("f2_xs", il, A_F2, Q2B_CTX / 32);
                }
                B.launch("attn_pv", 1, 256, {A(0), A(1), C(A_F), C(A_F2), A(2), A(3), A(4)},
                         {(int32_t)q2b_vt_block(al, 0, 0, 0), kvh, (int32_t)(M_QG + hq * 2 * Q2B_HD + Q2B_HD), (int32_t)(M_OA + hq * Q2B_HD), (int32_t)(L + LA_VSMAX)});
            }
            B.check("attn_gated", il, M_OA, Q2B_QHEADS * Q2B_HD);
            B.launch("quant_blocks", 64, 32, {A(0), A(1)}, {(int32_t)M_OA, (int32_t)A_OA});
            gemv_h(q2b_mat_index(il, MAT_O), M_Y, Q2B_D, A_OA);
            B.check("attn_output", il, M_Y, Q2B_D);
        }
        B.launch("add_into", 64, 32, {P(M_X), P(M_Y)});
        B.check("attn_residual", il, M_X, Q2B_D);
        norm_chain(L + (full ? LA_POST : LG_POST));
        B.check("attn_post_norm", il, M_H, Q2B_D);
        B.launch("quant_blocks", 64, 32, {A(0), A(1)}, {(int32_t)M_H, (int32_t)A_H});
        gemv_h(q2b_mat_index(il, MAT_FG), M_G, Q2B_FF, A_H);
        gemv_h(q2b_mat_index(il, MAT_FU), M_U, Q2B_FF, A_H);
        B.launch("silu_mul", Q2B_FF / 32, 32, {P(M_G), P(M_U), P(M_FFX)});
        B.launch("quant_blocks", Q2B_FF / 32, 32, {A(0), A(1)}, {(int32_t)M_FFX, (int32_t)A_FFX});
        B.launch("gemv_ffx192", Q2B_D / 32, 32, {A(0), C(A_FFX), P(M_Y)}, {(int32_t)wbase[(size_t)q2b_mat_index(il, MAT_FD)]});
        B.check("ffn_out", il, M_Y, Q2B_D);
        B.launch("add_into", 64, 32, {P(M_X), P(M_Y)});
        B.check("l_out", il, M_X, Q2B_D);
    }
    norm_chain(M_OUTNORM);
    B.check("result_norm", -1, M_H, Q2B_D);
    B.launch("quant_blocks", 64, 32, {A(0), A(1)}, {(int32_t)M_H, (int32_t)A_H});
    B.launch("step_pre", 1, 32, {});
    const uint32_t skip_branch = B.branch(SC_SKIP, 0);  // patched to step_end
    for (int row0 = 0; row0 < Q2B_VOCAB; row0 += 8192) {
        const int rows = std::min(8192, Q2B_VOCAB - row0);
        B.launch("gemv_logits", rows / 32, 32, {C(wbase[MAT_EMB]), C(A_H), A(0)}, {row0});
    }
    B.check("result_output", -1, SC_BEST, 2);  // the probe compares SC_BEST / SC_BEST_IDX with the logits' max / argmax
    const uint32_t desc_step_end = (uint32_t)B.queue.size();
    B.patch_branch(skip_branch, desc_step_end);
    B.launch("step_end", 1, 32, {P(M_TOKS)});
    const uint32_t noemit_branch = B.branch(SC_NOEMIT, 0);
    B.emit(JOB_NEXT);
    const uint32_t desc_cont = (uint32_t)B.queue.size();
    B.patch_branch(noemit_branch, desc_cont);
    B.branch(JOB_CONT, 0);
    const uint32_t desc_halt = (uint32_t)B.queue.size();
    B.halt();
    B.branch(JOB_ONE, 0);  // after GO: the next turn starts at the top
    if (B.queue.size() > (1u << SM_QBITS)) die("too many descriptors");
    for (size_t d = 0; d < B.queue.size(); d++) {
        B.m[SM_M_QUEUE + 2 * d] = (int32_t)B.queue[d].first;
        B.m[SM_M_QUEUE + 2 * d + 1] = (int32_t)B.queue[d].second;
    }

    // ------------------------------------------------------------------ the files
    {
        std::ofstream r(outdir + "/rom.bin", std::ios::binary);
        r.write((const char *)B.rom.data(), (std::streamsize)(B.rom.size() * sizeof(SmBlock)));
        std::ofstream mm(outdir + "/m.bin", std::ios::binary);
        mm.write((const char *)B.m.data(), (std::streamsize)(B.m.size() * 4));
        std::ofstream meta(outdir + "/meta.txt");
        meta << "nblk " << B.rom.size() << "\ndescriptors " << B.queue.size() << "\nm_words " << B.m.size() << "\na_blocks " << A_END << "\ns_elems " << Q2B_S_ELEMS
             << "\nw_blocks " << (wbase[(size_t)Q2B_NMATS - 1] + wblocks[(size_t)Q2B_NMATS - 1] * wrows[(size_t)Q2B_NMATS - 1]) << "\ndesc_step_end " << desc_step_end
             << "\ndesc_cont " << desc_cont << "\ndesc_halt " << desc_halt << "\n";
        std::ofstream lst(outdir + "/listing.txt");
        for (const auto &p : B.inst_list) lst << p.first << "\n" << p.second << "\n";
        std::ofstream q(outdir + "/queue.txt");
        for (size_t d = 0; d < B.queue.size(); d++) q << d << " " << B.qnames[d] << "\n";
        for (const Builder::Check &c : B.checks) q << "check " << c.d << " " << c.tap << " " << c.layer << " " << c.word << " " << c.count << "\n";
        // the whole program for an OpenCL host (tests/cl/q2b_cl.c): one descriptor a line
        //   launch NAME GX BX  c:V | p:WORD | a:K ...  |  ARGS...      branch WORD TARGET      emit WORD      halt
        std::ofstream pr(outdir + "/program.txt");
        for (size_t d = 0; d < B.queue.size(); d++) {
            const uint32_t w0 = B.queue[d].first, w1 = B.queue[d].second, kind = w0 & 3;
            if (kind == SM_Q_LAUNCH) {
                const Builder::LaunchRec &r = B.recs.at((uint32_t)d);
                pr << "launch " << r.name << " " << r.gx << " " << r.bx;
                const ptx::Kernel &kk = B.kernels.at(r.name);  // a runtime arg holding a pointer is q:K, a scalar a:K
                for (size_t i = 0; i < r.ps.size(); i++) {
                    const ParamSpec &p = r.ps[i];
                    pr << " " << (p.kind == ParamSpec::CONST ? "c:" : p.kind == ParamSpec::PTR ? "p:" : kk.params[i].size == 8 ? "q:" : "a:") << p.value;
                }
                pr << " |";
                for (int32_t a : r.args) pr << " " << a;
                pr << "\n";
            } else if (kind == SM_Q_BRANCH) {
                pr << "branch " << (w1 & 0xFFFFFF) << " " << ((w0 >> 24) | (w1 >> 24) << 8) << "\n";
            } else if (kind == SM_Q_EMIT) {
                pr << "emit " << (w1 & 0xFFFFFF) << "\n";
            } else {
                pr << "halt\n";
            }
        }
    }
    std::printf("host: wrote %s: ROM %zu blocks (%llu instructions, %zu instances), queue %zu descriptors, M %zu words\n", outdir.c_str(), B.rom.size(),
                (unsigned long long)B.instrs, B.inst_base.size(), B.queue.size(), B.m.size());
    return 0;
}
