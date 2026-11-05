// The gate of the circuit against the machine: the reference runs the 2B's program (the real job: the prompt in M, as
// q2b_gpu_check sets it) and feeds every token to the exact evaluator of the circuit, which must predict each next token
// before it arrives. The first disagreement names the token, the pass, and what the circuit said. Host only.
//
//   circ_gate IMAGE_DIR MODEL.gguf GEN "PROMPT" [--max N] [--check] [--classes c0,...,c5] [--stats]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <string>
#include <vector>

#include "../circuit/eval.h"
#include "../circuit/sm_circ.h"
#include "../gpu/image.h"
#include "../guest/q2b_map.h"
#include "../text/tok.h"

namespace {

struct Feed {
    Evaluator *ev;
    uint64_t n = 0, agree = 0, max = 0;
    int64_t predicted = -1;  // what the circuit predicted for the token about to arrive
    bool stop = false, first_report = true;
    double t0 = 0;
    uint32_t ring[512] = {0};  // the last tokens, for the mismatch report (back to the block's EXEC)
    SmMachine *mc = nullptr;
};

double now() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

const char *fam_name(int f) {
    static const char *const names[] = {"VLO", "VHI", "VEND", "EXEC", "NEXT", "WARP", "EMIT", "HALT", "GO", "LSTEP", "TEXT"};
    return f >= 0 && f < F_COUNT ? names[f] : "?";
}

std::string tok_str(int64_t t) {
    if (t < 0) return "none";
    Fam f = famof((uint32_t)t);
    char buf[64];
    std::snprintf(buf, sizeof buf, "%s(%u)", fam_name(f.id), f.p);
    return buf;
}

void on_token(void *ctx, uint32_t tok) {
    Feed *f = static_cast<Feed *>(ctx);
    if (f->stop) return;
    if (f->n > 0 && f->predicted != (int64_t)tok) {
        std::printf("host: MISMATCH at token %llu: the machine wrote %s, the circuit predicted %s\n", (unsigned long long)f->n, tok_str((int64_t)tok).c_str(), tok_str(f->predicted).c_str());
        std::printf("host: the tokens before it:");
        uint64_t from = f->n > 512 ? f->n - 512 : 0;
        for (uint64_t i = f->n; i > from; i--)
            if (famof(f->ring[(i - 1) % 512]).id == F_EXEC) {
                from = i - 1;
                break;
            }
        for (uint64_t i = from; i < f->n; i++) std::printf(" %s", tok_str((int64_t)f->ring[i % 512]).c_str());
        std::printf("\n");
        f->stop = true;
        f->mc->stop_req = 1;
        return;
    }
    f->ring[f->n % 512] = tok;
    if (f->n > 0) f->agree++;
    f->ev->step(tok);
    f->predicted = f->ev->predict();
    f->n++;
    if (f->n % 100000 == 0) {
        const double dt = now() - f->t0;
        std::printf("host: %llu tokens agree, %.1f passes/s\n", (unsigned long long)f->agree, (double)f->n / dt);
        std::fflush(stdout);
    }
    if (f->max && f->n >= f->max) {
        f->stop = true;
        f->mc->stop_req = 1;
    }
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 5) {
        std::fputs("usage: circ_gate IMAGE_DIR MODEL.gguf GEN \"PROMPT\" [--max N] [--check] [--stats]\n", stderr);
        return 2;
    }
    const std::string dir = argv[1], model = argv[2], prompt = argv[4];
    const int gen = std::atoi(argv[3]);
    uint64_t max = 0;
    bool check = false, stats_only = false;
    long long watch = -1;
    for (int i = 5; i < argc; i++) {
        if (!std::strcmp(argv[i], "--max") && i + 1 < argc) max = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--check")) check = true;
        else if (!std::strcmp(argv[i], "--stats")) stats_only = true;
        else if (!std::strcmp(argv[i], "--watch") && i + 1 < argc) watch = std::atoll(argv[++i]);
    }
    uint8_t classes[SM_STAGES] = SM_STAGE_CLASSES_DEFAULT;
    sm_classes_env(classes);
    Target target;
    target.key_heads = 16;
    target.value_heads = 32;
    target.intermediate = 8192;
    const double tb = now();
    SmCircStats st;
    Circ *circ = build_sm_circuit(classes, target, &st);
    std::printf("host: circuit: %llu units, %llu nodes (%llu heads, %llu banks), %llu dims, %d layers, built in %.1f s\n", (unsigned long long)st.units, (unsigned long long)st.nodes,
                (unsigned long long)st.heads, (unsigned long long)st.banks, (unsigned long long)st.dims, st.layers, now() - tb);
    std::fflush(stdout);
    if (stats_only) {
        // the shape of the layers: units and heads per sublayer, and the residual slots the dims need with lifetime sharing
        std::map<int, int> units, heads, banks, nodes;
        for (const Node &n : circ->nodes) {
            nodes[n.level]++;
            if (n.is_bank) banks[n.level]++;
            else if (n.is_ffn) units[n.level] += (int)n.terms.size();
            else heads[n.level]++;
        }
        int top = 0;
        for (const Node &n : circ->nodes) top = std::max(top, n.level);
        std::printf("host: sublayers 1..%d (odd: heads/banks, even: units):\n", top);
        for (int lv = 1; lv <= top; lv++)
            std::printf("  %3d: %5d nodes, %6d units, %2d heads, %2d banks\n", lv, nodes[lv], units[lv], heads[lv], banks[lv]);
        const int slots = allocate_slots(*circ, circ->keep, target.intermediate);
        std::printf("host: %zu dims need %d residual slots with lifetime sharing (hidden must hold 2048 + 8 ballast + these)\n", circ->dims.size(), slots);
        return 0;
    }

    Image im;
    image_load(im, dir);
    gguf::File gf;
    gguf::open(gf, model.c_str());
    Tokenizer tok(gf);
    std::vector<int32_t> toks = tok.encode(Tokenizer::chat_prompt(prompt), true);
    int32_t *m = im.img.m;
    for (size_t i = 0; i < toks.size(); i++) m[M_TOKS + i] = toks[i];
    m[JOB_SEQ] = 1;
    m[JOB_N] = (int32_t)toks.size();
    m[JOB_GEN] = gen;
    m[JOB_POS] = 0;
    m[JOB_PROMPT_N] = (int32_t)toks.size();
    m[JOB_MADE] = 0;
    m[JOB_NSTOPS] = 1;
    m[JOB_STOPS] = tok.id_of("<|im_end|>");
    std::printf("host: prompt %zu tokens, %d to generate; the circuit predicts every token of the reference's stream\n", toks.size(), gen);

    Evaluator ev(*circ, check);
    ev.load(im.img);
    if (watch >= 0) ev.watch(watch);
    Feed feed;
    feed.ev = &ev;
    feed.max = max;
    feed.t0 = now();
    SmMachine *mc = new SmMachine;
    sm_init(mc, &im.img);
    mc->sink.emit = on_token;
    mc->sink.ctx = &feed;
    feed.mc = mc;
    if (watch >= 0) {
        mc->watch_lo = (uint32_t)watch;
        mc->watch_hi = (uint32_t)watch + 1;
    }
    // the reference runs; the sink stops it at the first mismatch or at the token limit (within the record)
    int rc = sm_run(mc, 1ull << 62);
    const double dt = now() - feed.t0;
    std::printf("host: rc %d; %llu tokens fed, %llu agreed, %llu passes, %.1f passes/s, %.1f s\n", rc, (unsigned long long)feed.n, (unsigned long long)feed.agree, (unsigned long long)ev.passes(), (double)feed.n / dt, dt);
    if (rc == 2) {
        std::printf("host: the reference faulted: %s\n", mc->fault_what);
        return 1;
    }
    if (feed.stop && !(max && feed.n >= max)) {
        // where the two memories differ: M words, the uniform registers, the lane registers (the machine ran on past the mismatch by
        // at most the tokens of the record, so a few more differences than the cause may show)
        const std::vector<uint8_t> &cm = ev.memory(MEM_M);
        int shown = 0;
        for (uint32_t w = 0; w < im.img.m_words && shown < 12; w++) {
            int32_t v;
            std::memcpy(&v, &cm[(size_t)w * 4], 4);
            if (v != im.img.m[w]) {
                std::printf("host: M[%u]: machine %d, circuit %d\n", w, im.img.m[w], v);
                shown++;
            }
        }
        const std::vector<uint8_t> &cu = ev.memory(MEM_UREG);
        for (int s = 0; s < SM_WPB_MAX; s++)
            for (int r = 0; r < SM_NUREG; r++) {
                uint32_t v;
                std::memcpy(&v, &cu[((size_t)s * 16 + (size_t)r) * 4], 4);
                if (v != mc->ureg[s][r] && r != SM_ZERO) std::printf("host: u%d of slot %d: machine %u, circuit %u\n", r, s, mc->ureg[s][r], v);
            }
        const std::vector<uint8_t> &cr = ev.memory(MEM_REG);
        shown = 0;
        for (int s = 0; s < SM_WPB_MAX && shown < 12; s++)
            for (int L = 0; L < SM_LANES && shown < 12; L++)
                for (int r = 0; r < SM_NREG - 1 && shown < 12; r++) {
                    uint32_t v;
                    std::memcpy(&v, &cr[(((size_t)s * 32 + (size_t)L) * 16 + (size_t)r) * 4], 4);
                    if (v != mc->regs[s][r][L]) {
                        std::printf("host: r%d of lane %d slot %d: machine %u, circuit %u\n", r, L, s, mc->regs[s][r][L], v);
                        shown++;
                    }
                }
        const std::vector<uint8_t> &cs = ev.memory(MEM_SCR);
        std::printf("host: scratch:");
        for (int w = 0; w < SCR_WORDS; w++) {
            uint32_t v;
            std::memcpy(&v, &cs[(size_t)w * 4], 4);
            std::printf(" %u", v);
        }
        std::printf("\n");
        return 1;
    }
    std::printf("host: circ_gate: every token predicted (%s)\n", rc == 0 ? "to HALT" : "to the token limit");
    return 0;
}
