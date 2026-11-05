// q2b_gpu_check IMAGE_DIR MODEL.gguf GEN "PROMPT" [--hashlog FILE] [--max-records N]
//
// The 2B's program on the reference SIMT machine against the integer pass, token by token: the prompt goes into M as a
// job, the machine runs the queue until it halts, and every token it emits is compared with q2b_fix's argmax at the
// same position (q2b_fix teacher-forced on the machine's own tokens). Prints the text, the trace tokens per position,
// the records, the hash of the whole stream and the machine's rate. Host: the reference, not the result.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../gpu/image.h"
#include "../guest/q2b_map.h"
#include "../text/gguf.h"
#include "../text/q2b_fix.h"
#include "../text/tok.h"

namespace {

struct HashLog {
    FILE *f = nullptr;
    uint64_t every = 4096, n = 0, h = SM_FNV0;
};

void on_tok(void *ctx, uint32_t tok) {
    HashLog *hl = (HashLog *)ctx;
    hl->n++;
    hl->h = sm_fnv(hl->h, tok);
    if (hl->f && hl->n % hl->every == 0) std::fprintf(hl->f, "%llu %016llx\n", (unsigned long long)hl->n, (unsigned long long)hl->h);
}

int32_t argmax(const std::vector<int32_t> &v) {
    int32_t best = 0;
    for (int32_t k = 1; k < (int32_t)v.size(); k++)
        if (v[(size_t)k] > v[(size_t)best]) best = k;
    return best;
}

std::string shown(const std::string &s) {
    std::string out;
    for (char c : s) out += c == '\n' ? std::string("\\n") : std::string(1, c);
    return out;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 5) {
        std::fputs("usage: q2b_gpu_check IMAGE_DIR MODEL.gguf GEN \"PROMPT\" [--hashlog FILE] [--max-records N]\n", stderr);
        return 2;
    }
    const std::string dir = argv[1], model = argv[2];
    const int gen = std::atoi(argv[3]);
    const std::string prompt = argv[4];
    HashLog hl;
    uint64_t max_records = ~0ull;
    for (int i = 5; i < argc; i++) {
        if (!std::strcmp(argv[i], "--hashlog") && i + 1 < argc) hl.f = std::fopen(argv[++i], "w");
        else if (!std::strcmp(argv[i], "--max-records") && i + 1 < argc) max_records = std::strtoull(argv[++i], nullptr, 0);
    }
    gguf::File f;
    gguf::open(f, model.c_str());
    Tokenizer tok(f);
    std::vector<int32_t> toks = tok.encode(Tokenizer::chat_prompt(prompt), true);
    std::printf("host: prompt %zu tokens, ids:", toks.size());
    for (int32_t t : toks) std::printf(" %d", t);
    std::printf("\n");
    Image im;
    image_load(im, dir);
    std::printf("host: image: ROM %zu blocks, %u descriptors, M %zu words, A %u blocks, S %u, W %llu blocks\n", im.rom.size(), im.descriptors, im.m.size(), im.a_blocks,
                im.s_elems, (unsigned long long)im.img.w_blocks);
    // the job
    int32_t *m = im.m.data();
    for (size_t i = 0; i < toks.size(); i++) m[M_TOKS + i] = toks[i];
    m[JOB_SEQ] = 1;
    m[JOB_N] = (int32_t)toks.size();
    m[JOB_GEN] = gen;
    m[JOB_POS] = 0;
    m[JOB_PROMPT_N] = (int32_t)toks.size();
    m[JOB_MADE] = 0;
    m[JOB_NSTOPS] = 3;
    m[JOB_STOPS + 0] = tok.id_of("<|im_end|>");
    m[JOB_STOPS + 1] = tok.id_of("<|endoftext|>");
    m[JOB_STOPS + 2] = m[JOB_STOPS + 0];
    SmMachine *mc = (SmMachine *)std::calloc(1, sizeof *mc);
    sm_init(mc, &im.img);
    mc->sink.emit = on_tok;
    mc->sink.ctx = &hl;

    // the reference, teacher-forced on the machine's tokens
    q2b::Fix fix(f);
    std::vector<int32_t> logits(q2b::VOCAB);
    std::vector<int32_t> fix_next;  // the reference's argmax after each position
    size_t fed = 0;

    uint64_t last_n = 0;
    auto t0 = std::chrono::steady_clock::now();
    int emitted = 0, same = 0;
    std::string text;
    // run position by position: a pass of the queue ends at the emit (or at the branch back for prompt positions); we
    // stop at records and watch the job words instead: simpler to run to HALT and read the emitted tokens from M as
    // they arrive via the EMIT token. The sink sees EMIT then two VALs: catch them here.
    struct Emits {
        std::vector<uint32_t> toks;
        int state = 0;
        uint32_t lo = 0;
    } em;
    struct Both {
        HashLog *hl;
        Emits *em;
    } both{&hl, &em};
    mc->sink.ctx = &both;
    mc->sink.emit = [](void *ctx, uint32_t t) {
        Both *b = (Both *)ctx;
        on_tok(b->hl, t);
        Emits &e = *b->em;
        if (e.state == 0) {
            if (t == SM_EMIT) e.state = 1;
        } else if (e.state == 1) {
            e.lo = t;
            e.state = 2;
        } else {
            e.toks.push_back(e.lo | t << 16);
            e.state = 0;
        }
    };
    int rc;
    size_t seen = 0;
    for (;;) {
        rc = sm_run(mc, 1u << 20);
        // report every chunk
        auto now = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(now - t0).count();
        while (seen < em.toks.size()) {
            uint32_t t = em.toks[seen++];
            // the reference: feed the prompt up to this position, then the machine's tokens
            while (fed < toks.size()) {
                fix.step(toks[fed++], logits.data());
            }
            int32_t ref = argmax(logits);
            fix_next.push_back(ref);
            bool ok = (int32_t)t == ref;
            same += ok;
            emitted++;
            std::printf("position %zu: machine %u '%s'  reference %d '%s'  %s  (%llu trace tokens so far, %.1f tok/s)\n", toks.size() - 1, t, shown(tok.decode((int32_t)t)).c_str(), ref,
                        shown(tok.decode(ref)).c_str(), ok ? "same" : "DIFFER", (unsigned long long)mc->sink.n, mc->sink.n / secs);
            std::fflush(stdout);
            text += tok.decode((int32_t)t);
            toks.push_back((int32_t)t);  // the machine's token is the next position's, for both
        }
        if (rc == 0 || rc == 2) break;
        if (mc->records >= max_records) break;
        if (mc->sink.n - last_n > 50000000ull) {
            last_n = mc->sink.n;
            std::printf("host: %llu records, %llu tokens, %.1f tok/s, position %d\n", (unsigned long long)mc->records, (unsigned long long)mc->sink.n, mc->sink.n / secs,
                        m[JOB_POS]);
            std::fflush(stdout);
        }
    }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("host: rc %d %s; %d tokens emitted, %d same as the reference; %llu records, %llu launches, %llu trace tokens, hash %016llx, %.1f s, %.1f trace tok/s, %.2f trace tokens per position over %d positions\n",
                rc, mc->fault_what, emitted, same, (unsigned long long)mc->records, (unsigned long long)mc->launches, (unsigned long long)mc->sink.n, (unsigned long long)mc->sink.hash,
                secs, mc->sink.n / secs, m[JOB_POS] ? (double)mc->sink.n / m[JOB_POS] : 0.0, m[JOB_POS]);
    std::printf("text: %s\n", shown(text).c_str());
    std::printf("ops:");
    for (int op = 0; op < SM_NOPS; op++)
        if (mc->by_op[op]) std::printf(" %d:%llu", op, (unsigned long long)mc->by_op[op]);
    std::printf("\n");
    if (hl.f) std::fclose(hl.f);
    return emitted && same == emitted ? 0 : 1;
}
