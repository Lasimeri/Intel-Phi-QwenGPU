// q2b_gpu_probe IMAGE_DIR MODEL.gguf "PROMPT" [--positions P] [--show N]
//
// The first P prompt tokens through the 2B's program on the reference machine, the machine stopped after every
// descriptor that out/gpu/queue.txt marks with a check, and the M region the check names compared word for word with
// q2b_fix's tensor of the same name at the same position (its tap). The arithmetic is meant to be identical, so the
// first stage whose words differ is the bug. Host.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "../gpu/image.h"
#include "../guest/q2b_map.h"
#include "../text/gguf.h"
#include "../text/q2b_fix.h"
#include "../text/tok.h"

namespace {

struct Check {
    uint32_t d = 0;
    std::string tap;
    int layer = 0;
    uint32_t word = 0, count = 0;
};

[[noreturn]] void die(const std::string &what) {
    std::fprintf(stderr, "q2b_gpu_probe: %s\n", what.c_str());
    std::exit(2);
}

inline int64_t shr_round(int64_t v, int s) { return (v + (1LL << (s - 1))) >> s; }

// One gemv row from the image's own bytes with the reference's formula (q8dot: t = (inner * mw + 2^15) >> 16,
// u = (t * xs + 2^15) >> 16, the sum wrapping in 32 bits), block by block: W row `r` of matrix `mi` against the A
// blocks from `ablk0`. Printed beside the machine's and the reference's values so the deviating side is named.
void recompute_row(const Image &im, int mi, uint32_t ablk0, int r, int32_t machine, int32_t reference) {
    const int32_t wbase = im.m[(size_t)M_WBASE + 2 * (size_t)mi], nb = im.m[(size_t)M_WBASE + 2 * (size_t)mi + 1];
    uint32_t acc = 0;
    std::printf("    row %d of matrix %d (W base %d, %d blocks) against A blocks from %u: machine %d, reference %d\n", r, mi, wbase, nb, ablk0, machine, reference);
    for (int b = 0; b < nb; b++) {
        const uint8_t *wp = im.w + ((uint64_t)wbase + (uint64_t)r * (uint64_t)nb + (uint64_t)b) * 34;
        const uint8_t *ap = &im.a[((size_t)ablk0 + (size_t)b) * 36];
        uint16_t h;
        std::memcpy(&h, wp, 2);
        int32_t xs;
        std::memcpy(&xs, ap + 32, 4);
        int32_t inner = 0;
        for (int i = 0; i < 32; i++) inner += (int32_t)(int8_t)wp[2 + i] * (int32_t)(int8_t)ap[i];
        const int64_t mw = im.f16_q24[h];
        const int64_t t = shr_round((int64_t)inner * mw, 16), u = shr_round(t * xs, 16);
        acc += (uint32_t)u;
        if (b < 6 || u != 0) std::printf("      b %2d: h %04x mw %lld inner %d t %lld xs %d u %lld acc %d\n", b, h, (long long)mw, inner, (long long)t, xs, (long long)u, (int32_t)acc);
    }
    std::printf("    formula on the image's bytes: %d\n", (int32_t)acc);
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 4) {
        std::fputs("usage: q2b_gpu_probe IMAGE_DIR MODEL.gguf \"PROMPT\" [--positions P] [--show N]\n", stderr);
        return 2;
    }
    const std::string dir = argv[1], model = argv[2], prompt = argv[3];
    int positions = 1, show = 4;
    uint32_t trace_lo = 0, trace_hi = 0, trace_desc = 0;
    bool trace_desc_on = false;
    std::vector<std::pair<uint32_t, uint32_t>> watch;  // --watch WORD:COUNT, printed after every stop
    for (int i = 4; i < argc; i++) {
        if (!std::strcmp(argv[i], "--positions") && i + 1 < argc) positions = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--show") && i + 1 < argc) show = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--watch") && i + 1 < argc) {
            const char *s = argv[++i];
            const char *colon = std::strchr(s, ':');
            watch.emplace_back((uint32_t)std::strtoul(s, nullptr, 0), colon ? (uint32_t)std::strtoul(colon + 1, nullptr, 0) : 1u);
        } else if (!std::strcmp(argv[i], "--trace-desc") && i + 1 < argc) {  // the uniform registers after every block of warp 0 in this descriptor
            trace_desc = (uint32_t)std::strtoul(argv[++i], nullptr, 0);
            trace_desc_on = true;
        } else if (!std::strcmp(argv[i], "--trace-store") && i + 1 < argc) {  // WORD:COUNT: every M store into the range, to stderr
            const char *s = argv[++i];
            const char *colon = std::strchr(s, ':');
            trace_lo = (uint32_t)std::strtoul(s, nullptr, 0);
            trace_hi = trace_lo + (colon ? (uint32_t)std::strtoul(colon + 1, nullptr, 0) : 1u);
        }
    }
    // the checks
    std::vector<Check> checks;
    std::vector<std::string> qnames;
    {
        std::ifstream q(dir + "/queue.txt");
        if (!q) die("cannot open " + dir + "/queue.txt");
        std::string line;
        while (std::getline(q, line)) {
            std::istringstream ss(line);
            std::string a;
            ss >> a;
            if (a == "check") {
                Check c;
                ss >> c.d >> c.tap >> c.layer >> c.word >> c.count;
                checks.push_back(c);
            } else {
                std::string name;
                ss >> name;
                qnames.push_back(name);
            }
        }
    }
    gguf::File f;
    gguf::open(f, model.c_str());
    Tokenizer tok(f);
    std::vector<int32_t> toks = tok.encode(Tokenizer::chat_prompt(prompt), true);
    if ((int)toks.size() < positions) positions = (int)toks.size();
    std::printf("host: prompt %zu tokens, probing %d positions, %zu checks per position\n", toks.size(), positions, checks.size());
    Image im;
    image_load(im, dir);
    int32_t *m = im.m.data();
    for (int i = 0; i < positions; i++) m[M_TOKS + i] = toks[(size_t)i];
    m[JOB_SEQ] = 1;
    m[JOB_N] = positions;
    m[JOB_GEN] = 1;
    m[JOB_POS] = 0;
    m[JOB_PROMPT_N] = positions;
    m[JOB_MADE] = 0;
    m[JOB_NSTOPS] = 0;
    SmMachine *mc = (SmMachine *)std::calloc(1, sizeof *mc);
    sm_init(mc, &im.img);
    mc->watch_lo = trace_lo;
    mc->watch_hi = trace_hi;
    mc->trace_desc = trace_desc;
    mc->trace_desc_on = trace_desc_on;

    q2b::Fix fix(f);
    std::vector<int32_t> logits(q2b::VOCAB);
    std::map<std::pair<std::string, int>, std::vector<int32_t>> taps;
    q2b::FixTap tap = [&](const char *name, int il, const int32_t *v, size_t n) { taps[{name, il}] = std::vector<int32_t>(v, v + n); };

    int bad_stages = 0;
    for (int pos = 0; pos < positions; pos++) {
        taps.clear();
        fix.step(toks[(size_t)pos], logits.data(), &tap);
        std::printf("host: position %d token %d '%s': %zu reference tensors\n", pos, toks[(size_t)pos], tok.decode(toks[(size_t)pos]).c_str(), taps.size());
        for (const Check &c : checks) {
            mc->stop_qi = c.d + 1;
            int rc = sm_run(mc, 1ull << 62);
            if (rc != 3) {
                std::printf("host: machine returned %d (%s) before descriptor %u (%s): stopping\n", rc, mc->fault_what, c.d, qnames[c.d].c_str());
                return 1;
            }
            const std::string kname = c.d < qnames.size() ? qnames[c.d] : "?";
            if (!watch.empty() && (&c == &checks[0] || (&c)[-1].d != c.d)) {
                for (const auto &w : watch) {
                    std::printf("  after d %u M[%u..+%u]:", c.d, w.first, w.second);
                    for (uint32_t i = 0; i < w.second; i++) std::printf(" %d", m[w.first + i]);
                    std::printf("\n");
                }
            }
            if (c.tap == "gdn_state") {  // the S bank (int16) against the reference layer's state after this position
                const int16_t *ref = fix.gdn_state(c.layer).state;
                size_t ndiff = 0, first = c.count;
                int maxd = 0;
                for (size_t i = 0; i < c.count; i++) {
                    int d = (int)im.s[c.word + i] - ref[i];
                    if (d) {
                        if (ndiff == 0) first = i;
                        ndiff++;
                        maxd = std::max(maxd, d < 0 ? -d : d);
                    }
                }
                if (ndiff == 0) {
                    std::printf("  d %4u %-18s %-22s L%-3d ok (%u state elements)\n", c.d, kname.c_str(), c.tap.c_str(), c.layer, c.count);
                } else {
                    std::printf("  d %4u %-18s %-22s L%-3d DIFFER: %zu of %u elements, max |diff| %d, first at %zu (head %zu row %zu col %zu):", c.d, kname.c_str(), c.tap.c_str(),
                                c.layer, ndiff, c.count, maxd, first, first / (Q2B_ST * Q2B_ST), first / Q2B_ST % Q2B_ST, first % Q2B_ST);
                    int shown = 0;
                    for (size_t i = first; i < c.count && shown < show; i++)
                        if (im.s[c.word + i] != ref[i]) {
                            std::printf(" [%zu] %d vs %d", i, im.s[c.word + i], ref[i]);
                            shown++;
                        }
                    std::printf("\n");
                    bad_stages++;
                }
                continue;
            }
            auto it = taps.find({c.tap, c.layer});
            if (it == taps.end()) {
                std::printf("  d %4u %-18s %-22s L%-3d no reference tensor\n", c.d, kname.c_str(), c.tap.c_str(), c.layer);
                continue;
            }
            const std::vector<int32_t> &ref = it->second;
            if (c.tap == "result_output") {
                if (pos + 1 < positions) {  // a prompt position: the machine skips the logits by design (SC_SKIP)
                    std::printf("  d %4u %-18s %-22s      skipped: a prompt position, the machine computes no logits here\n", c.d, kname.c_str(), c.tap.c_str());
                    continue;
                }
                int32_t best = 0;
                for (int32_t k = 1; k < (int32_t)ref.size(); k++)
                    if (ref[(size_t)k] > ref[(size_t)best]) best = k;
                const int32_t mbest = m[SC_BEST], midx = m[SC_BEST_IDX];
                const bool ok = mbest == ref[(size_t)best] && midx == best;
                std::printf("  d %4u %-18s %-22s      %s: machine best %d at %d, reference %d at %d\n", c.d, kname.c_str(), c.tap.c_str(), ok ? "ok" : "DIFFER", mbest, midx,
                            ref[(size_t)best], best);
                if (!ok) {  // both rows from the image's bytes: the embedding matrix against the A blocks of the output norm
                    recompute_row(im, MAT_EMB, (uint32_t)A_H, best, 0, ref[(size_t)best]);
                    if (midx >= 0 && midx < (int32_t)ref.size()) recompute_row(im, MAT_EMB, (uint32_t)A_H, midx, mbest, ref[(size_t)midx]);
                }
                bad_stages += !ok;
                continue;
            }
            uint32_t word = c.word;
            if (c.tap == "linear_attn_qkv_mixed") word += (uint32_t)(pos & 3) * Q2B_QKV;  // the window slot of this position
            if (c.tap.compare(0, 2, "kc") == 0) word += (uint32_t)pos * 8;                 // this position's key blocks
            if (c.tap == "attn_vs") word += (uint32_t)pos * 2;                              // this position's value scales
            const size_t n = std::min<size_t>(ref.size(), c.count);
            // a tap named *_q compares the A bank's int8 lanes from block `word`; *_xs its Q.24 scales; vt*_q the
            // transposed value cache (dim i at block word + i * CTX/32 + pos/32, lane pos % 32); else M words
            const bool vt = c.tap.compare(0, 2, "vt") == 0;
            const bool a8 = !vt && c.tap.size() > 2 && c.tap.compare(c.tap.size() - 2, 2, "_q") == 0;
            const bool a32 = c.tap.size() > 3 && c.tap.compare(c.tap.size() - 3, 3, "_xs") == 0;
            auto machine = [&](size_t i) -> int64_t {
                if (vt) return (int8_t)im.a[((size_t)word + i * (Q2B_CTX / 32) + (size_t)pos / 32) * 36 + (size_t)pos % 32];
                if (a8) return (int8_t)im.a[((size_t)word + i / 32) * 36 + i % 32];
                if (a32) {
                    int32_t s;
                    std::memcpy(&s, &im.a[((size_t)word + i) * 36 + 32], 4);
                    return s;
                }
                return m[word + i];
            };
            size_t ndiff = 0, first = n;
            int64_t maxd = 0;
            for (size_t i = 0; i < n; i++) {
                int64_t d = machine(i) - ref[i];
                if (d) {
                    if (ndiff == 0) first = i;
                    ndiff++;
                    maxd = std::max<int64_t>(maxd, d < 0 ? -d : d);
                }
            }
            if (ndiff == 0) {
                std::printf("  d %4u %-18s %-22s L%-3d ok (%zu words)\n", c.d, kname.c_str(), c.tap.c_str(), c.layer, n);
            } else {
                std::printf("  d %4u %-18s %-22s L%-3d DIFFER: %zu of %zu words, max |diff| %lld, first at %zu:", c.d, kname.c_str(), c.tap.c_str(), c.layer, ndiff, n,
                            (long long)maxd, first);
                int shown = 0;
                for (size_t i = first; i < n && shown < show; i++)
                    if (machine(i) != ref[i]) {
                        std::printf(" [%zu] %lld vs %d", i, (long long)machine(i), ref[i]);
                        shown++;
                    }
                std::printf("\n");
                bad_stages++;
                if (c.tap == "linear_attn_out" && bad_stages == 1)
                    recompute_row(im, q2b_mat_index(c.layer, MAT_OUT), (uint32_t)A_MIX, (int)first, (int32_t)machine(first), ref[first]);
            }
            if (ref.size() != c.count) std::printf("  (reference has %zu values, the check %u words)\n", ref.size(), c.count);
        }
    }
    mc->stop_qi = 0;
    int rc = sm_run(mc, 1ull << 62);
    std::printf("host: run to the end: rc %d, %llu records, %llu trace tokens, %d differing stages over %d positions\n", rc, (unsigned long long)mc->records,
                (unsigned long long)mc->sink.n, bad_stages, positions);
    return bad_stages ? 1 : 0;
}
