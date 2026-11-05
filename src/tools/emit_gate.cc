// The gate of the model file against the machine: the reference runs the 2B's real job and feeds every token of its
// stream to the forward over the file (src/exec/tforward.cc), which must have predicted it; with --dims the exact
// evaluator of the circuit runs beside them and every chip slot of the file's residual is compared with the evaluator's
// dim after each pass. The first disagreement names the token, the pass, and the dims. Host only.
//
//   emit_gate MODEL.gguf IMAGE_DIR MODEL_2B.gguf GEN "PROMPT" [--max N] [--dims] [--check]
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include "../circuit/eval.h"
#include "../circuit/sm_circ.h"
#include "../exec/tforward.h"
#include "../gpu/image.h"
#include "../guest/q2b_map.h"
#include "../text/tok.h"

namespace {

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

struct Feed {
    TForward *tf;
    Evaluator *ev = nullptr;
    const Circ *circ = nullptr;
    std::vector<int> final_dim; // slot -> dim
    int off = 0, m = 0;
    uint64_t n = 0, agree = 0, max = 0, dims_checked = 0;
    uint64_t hash = 0xcbf29ce484222325ull;  // FNV-1a over the fed stream, as gpu_run hashes its own
    FILE *hlog = nullptr;
    int64_t predicted = -1;
    bool stop = false;
    double t0 = 0;
    uint32_t ring[48] = {0};
    SmMachine *mc = nullptr;
    // --probe: the file's normed input of every sublayer, kept for the report; the slot map and the allocation
    std::vector<std::vector<float>> hn_by_level;
    std::string probe_name;
    const Alloc *al = nullptr;
    std::vector<int> slot_dim;
};

struct HnProbe : TProbe {
    Feed *f;
    void on_hn(int layer, int side, const float *hn, const float *) override {
        const size_t lv = (size_t)(2 * layer + 1 + side);
        if (f->hn_by_level.size() <= lv) f->hn_by_level.resize(lv + 1);
        f->hn_by_level[lv].assign(hn, hn + f->tf->hidden());
    }
};

// the node that writes a dim, its terms, and every input's value in the evaluator and in the file (hn / c at the node's level)
void probe_dim(Feed &f, const Alloc &al, const std::vector<int> &slot_dim, double c_scale) {
    const Circ &c = *f.circ;
    int d = -1;
    for (size_t i = 0; i < c.dims.size(); i++)
        if (c.dims[i].name == f.probe_name) d = (int)i;
    if (d < 0) {
        std::printf("host: probe: no dim named %s\n", f.probe_name.c_str());
        return;
    }
    const std::vector<double> &r = f.ev->residual();
    std::printf("host: probe dim %d (%s): level %d, slot dim %d, evaluator %.9g, file (before the exit) %.9g\n", d, f.probe_name.c_str(), c.dims[(size_t)d].level, slot_dim[(size_t)d], r[(size_t)d],
                slot_dim[(size_t)d] >= 0 ? (double)f.tf->before()[(size_t)slot_dim[(size_t)d]] : 0.0);
    if (slot_dim[(size_t)d] >= 0) {
        std::printf("host: probe: the slot's value (hn / c) at every sublayer's input:");
        for (size_t lv = 1; lv < f.hn_by_level.size(); lv++)
            if (!f.hn_by_level[lv].empty()) std::printf(" L%zu:%.9g", lv, (double)f.hn_by_level[lv][(size_t)slot_dim[(size_t)d]] / c_scale);
        std::printf("\n");
    }
    const auto hs = al.subtract.find(d);
    if (hs != al.subtract.end()) std::printf("host: probe: born by handover from dim %d (%s, level %d, evaluator %.9g)\n", hs->second, c.dims[(size_t)hs->second].name.c_str(), c.dims[(size_t)hs->second].level, r[(size_t)hs->second]);
    // every dim that ever owns this slot, by level: the handover chain
    if (al.slot[(size_t)d] >= 0) {
        std::vector<int> owners;
        for (size_t i = 1; i < c.dims.size(); i++)
            if (al.slot[i] == al.slot[(size_t)d]) owners.push_back((int)i);
        std::sort(owners.begin(), owners.end(), [&](int p, int q) { return c.dims[(size_t)p].level < c.dims[(size_t)q].level; });
        std::printf("host: probe: slot %d owners by level:", al.slot[(size_t)d]);
        for (int o : owners) {
            const auto cl = al.subtract.find(o);
            std::printf(" [%d %s L%d ev %.9g%s]", o, c.dims[(size_t)o].name.c_str(), c.dims[(size_t)o].level, r[(size_t)o], cl != al.subtract.end() ? " sub" : "");
        }
        std::printf("\n");
    }
    for (const Node &n : c.nodes) {
        if (!n.is_ffn || n.out != d) continue;
        std::printf("host: probe: node %s at level %d, %zu terms%s\n", n.name.c_str(), n.level, n.terms.size(), n.into ? " (into)" : "");
        const std::vector<float> *hn = (size_t)n.level < f.hn_by_level.size() ? &f.hn_by_level[(size_t)n.level] : nullptr;
        auto show = [&](const char *what, const Expr &e) {
            double ev_v = 0, file_v = 0;
            std::printf("host:     %s =", what);
            for (const auto &p : e.t) {
                const double coef = p.second.to_double();
                const double ev_x = p.first == ONE ? 1.0 : r[(size_t)p.first];
                double file_x = 0;
                if (hn) file_x = p.first == ONE ? (*hn)[(size_t)f.off] / 32.0 : (slot_dim[(size_t)p.first] >= 0 ? (*hn)[(size_t)slot_dim[(size_t)p.first]] / c_scale : 0.0);
                ev_v += coef * ev_x;
                file_v += coef * file_x;
                std::printf(" %+g*%s[ev %.9g, file %.9g]", coef, p.first == ONE ? "ONE" : c.dims[(size_t)p.first].name.c_str(), ev_x, file_x);
            }
            std::printf("  -> ev %.9g, file %.9g\n", ev_v, file_v);
        };
        for (const Term &t : n.terms) {
            std::printf("host:   term c = %g\n", t.c.to_double());
            show("a", t.a);
            show("b", t.b);
        }
    }
}

// every dim of the circuit against the file: a dim's value is its slot's value at the input of the sublayer after its producer's
// (the normed input hn there, divided by c), where the slot still holds it whatever handover follows; the dims the exit leaves
// (above the last sublayer) are read from the residual before the exit layers
bool compare_dims(Feed &f, uint32_t tok, int *lowest) {
    const std::vector<float> &x = f.tf->before();
    const std::vector<double> &r = f.ev->residual();
    const double c = std::ldexp(1.0, -35);
    struct Diff {
        int level, dim;
        size_t slot;
        double a, b;
    };
    std::vector<Diff> diffs;
    for (size_t d = 1; d < f.circ->dims.size(); d++) {
        const int sd = f.slot_dim[d];
        if (sd < 0) continue;
        const size_t lv = (size_t)f.circ->dims[d].level + 1;
        const double a = lv < f.hn_by_level.size() && !f.hn_by_level[lv].empty() ? (double)f.hn_by_level[lv][(size_t)sd] / c : (double)x[(size_t)sd];
        const double b = r[d];
        f.dims_checked++;
        if (a != b) diffs.push_back(Diff{f.circ->dims[d].level, (int)d, (size_t)(sd - f.off - f.m), a, b});
    }
    if (diffs.empty()) return true;
    if (lowest) *lowest = -1;
    std::sort(diffs.begin(), diffs.end(), [](const Diff &p, const Diff &q) { return p.level < q.level || (p.level == q.level && p.dim < q.dim); });
    if (lowest) *lowest = diffs[0].dim;
    std::printf("host: DIMS DIFFER after token %llu (%s): %zu dims; the lowest levels:\n", (unsigned long long)f.n, tok_str((int64_t)tok).c_str(), diffs.size());
    for (size_t i = 0; i < diffs.size() && i < 24; i++)
        std::printf("host:   level %d slot %zu = dim %d (%s): file %.9g, evaluator %.9g\n", diffs[i].level, diffs[i].slot, diffs[i].dim, f.circ->dims[(size_t)diffs[i].dim].name.c_str(), diffs[i].a, diffs[i].b);
    return false;
}

void on_token(void *ctx, uint32_t tok) {
    Feed *f = static_cast<Feed *>(ctx);
    if (f->stop) return;
    if (f->n > 0 && f->predicted != (int64_t)tok) {
        std::printf("host: MISMATCH at token %llu: the machine wrote %s, the file predicted %s\n", (unsigned long long)f->n, tok_str((int64_t)tok).c_str(), tok_str(f->predicted).c_str());
        std::printf("host: the tokens before it:");
        const uint64_t from = f->n > 48 ? f->n - 48 : 0;
        for (uint64_t i = from; i < f->n; i++) std::printf(" %s", tok_str((int64_t)f->ring[i % 48]).c_str());
        std::printf("\n");
        f->stop = true;
        f->mc->stop_req = 1;
        return;
    }
    f->ring[f->n % 48] = tok;
    f->hash = (f->hash ^ tok) * 0x100000001b3ull;
    if (f->hlog && (f->n + 1) % 4096 == 0) {
        std::fprintf(f->hlog, "tokens %llu hash %016llx\n", (unsigned long long)(f->n + 1), (unsigned long long)f->hash);
        std::fflush(f->hlog);
    }
    if (f->n > 0) f->agree++;
    f->tf->step(tok);
    f->predicted = (int64_t)f->tf->next_token();
    if (f->ev) {
        f->ev->step(tok);
        int lowest = -1;
        if (!compare_dims(*f, tok, &lowest)) {
            if (f->probe_name.empty() && lowest >= 0) f->probe_name = f->circ->dims[(size_t)lowest].name;  // the lowest differing dim
            if (!f->probe_name.empty() && f->al) probe_dim(*f, *f->al, f->slot_dim, std::ldexp(1.0, -35));
            f->stop = true;
            f->mc->stop_req = 1;
            return;
        }
        const int64_t ep = f->ev->predict();
        if (ep != f->predicted) {
            std::printf("host: the evaluator predicts %s, the file %s, after token %llu\n", tok_str(ep).c_str(), tok_str(f->predicted).c_str(), (unsigned long long)f->n);
            f->stop = true;
            f->mc->stop_req = 1;
            return;
        }
    }
    f->n++;
    if (f->n % 10000 == 0) {
        const double dt = now() - f->t0;
        std::printf("host: %llu tokens agree, %.1f passes/s%s\n", (unsigned long long)f->agree, (double)f->n / dt, f->ev ? ", dims equal" : "");
        std::fflush(stdout);
    }
    if (f->max && f->n >= f->max) {
        f->stop = true;
        f->mc->stop_req = 1;
    }
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 6) {
        std::fputs("usage: emit_gate MODEL.gguf IMAGE_DIR MODEL_2B.gguf GEN \"PROMPT\" [--max N] [--dims] [--check]\n", stderr);
        return 2;
    }
    const std::string model = argv[1], dir = argv[2], model2b = argv[3], prompt = argv[5];
    const int gen = std::atoi(argv[4]);
    uint64_t max = 0;
    bool dims = false, check = false;
    std::string probe;
    FILE *hlog = nullptr;
    for (int i = 6; i < argc; i++) {
        if (!std::strcmp(argv[i], "--max") && i + 1 < argc) max = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--dims")) dims = true;
        else if (!std::strcmp(argv[i], "--check")) check = true;
        else if (!std::strcmp(argv[i], "--probe") && i + 1 < argc) probe = argv[++i];
        else if (!std::strcmp(argv[i], "--hashlog") && i + 1 < argc) hlog = std::fopen(argv[++i], "w");
    }
    const double tl = now();
    TForward tf(model);
    const int exit_layers = (int)tf.meta("qwen35moe.chip.exit_layers");
    if (dims) tf.keep_before(tf.layers() - exit_layers);  // the residual before the exit's clears: every slot still holds its dim
    std::printf("host: the file: %d layers, hidden %d, %llu nonzero weights, %.2f GB of data, loaded in %.1f s\n", tf.layers(), tf.hidden(), (unsigned long long)tf.nnz(), (double)tf.file_data_bytes() / 1e9, now() - tl);
    std::fflush(stdout);

    Image im;
    image_load(im, dir);
    gguf::File gf;
    gguf::open(gf, model2b.c_str());
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
    m[JOB_NSTOPS] = 3;
    m[JOB_STOPS + 0] = tok.id_of("<|im_end|>");
    m[JOB_STOPS + 1] = tok.id_of("<|endoftext|>");
    m[JOB_STOPS + 2] = m[JOB_STOPS + 0];
    // the file's words memory gets the same job
    {
        std::vector<uint8_t> &mw = tf.memory(MB_WORDS);
        if (mw.size() != (size_t)im.img.m_words * 4) {
            std::printf("host: the file's words memory has %zu bytes, the image %u words\n", mw.size(), im.img.m_words);
            return 1;
        }
        std::memcpy(mw.data(), m, mw.size());
    }
    Feed feed;
    feed.tf = &tf;
    feed.probe_name = probe;
    feed.hlog = hlog;
    HnProbe hp;
    hp.f = &feed;
    if (dims) tf.probe(&hp);
    feed.max = max;
    feed.t0 = now();
    Circ *circ = nullptr;
    Evaluator *ev = nullptr;
    if (dims) {
        uint8_t classes[SM_STAGES] = SM_STAGE_CLASSES_DEFAULT;
        sm_classes_env(classes);
        Target target;
        target.key_heads = 16;
        target.value_heads = 32;
        target.intermediate = 8192;
        circ = build_sm_circuit(classes, target, nullptr);
        ev = new Evaluator(*circ, check);
        ev->load(im.img);
        feed.ev = ev;
        feed.circ = circ;
        feed.off = (int)tf.meta("qwen35moe.chip.first_dim");
        feed.m = (int)tf.meta("qwen35moe.chip.ballast_dims");
        const gguf::Tensor &fo = tf.file().tensor("chip.final");
        std::vector<float> f(fo.dims[0]);
        gguf::read_at(tf.file(), fo.offset, f.data(), fo.nbytes);
        for (float v : f) feed.final_dim.push_back((int)v);
        std::printf("host: dims compared every pass: %zu slots\n", feed.final_dim.size());
    }
    SmMachine *mc = new SmMachine;
    sm_init(mc, &im.img);
    mc->sink.emit = on_token;
    mc->sink.ctx = &feed;
    feed.mc = mc;
    std::printf("host: prompt %zu tokens, %d to generate; the file predicts every token of the reference's stream\n", toks.size(), gen);
    std::fflush(stdout);
    Alloc al;
    if (circ) {
        al = allocate(*circ, circ->keep, 8192);
        const gguf::Tensor &sl = tf.file().tensor("chip.slot");
        std::vector<float> sv(sl.dims[0]);
        gguf::read_at(tf.file(), sl.offset, sv.data(), sl.nbytes);
        for (float v : sv) feed.slot_dim.push_back((int)v);
        feed.al = &al;
    }
    const int rc = sm_run(mc, 1ull << 62);
    const double dt = now() - feed.t0;
    std::printf("host: rc %d; %llu tokens fed, %llu agreed, %.1f passes/s, %.1f s, %llu full scans of the output head%s\n", rc, (unsigned long long)feed.n, (unsigned long long)feed.agree, (double)feed.n / dt, dt,
                (unsigned long long)tf.full_scans(), dims ? (", " + std::to_string(feed.dims_checked) + " slot values compared").c_str() : "");
    if (rc == 2) {
        std::printf("host: the reference faulted: %s\n", mc->fault_what);
        return 1;
    }
    if (feed.stop && !(max && feed.n >= max)) return 1;
    std::printf("host: emit_gate: every token predicted by the file (%s)\n", rc == 0 ? "to HALT" : "to the token limit");
    return 0;
}
