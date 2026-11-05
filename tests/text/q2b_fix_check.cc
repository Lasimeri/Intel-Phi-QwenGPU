/* Stage 2's gate: the 2B in integers against the 2B exact. Host, test only.
 *
 *   q2b_fix_check MODEL GEN PROMPTS_FILE [FREE_RUN_PROMPT ...]
 *
 * The exact reference writes GEN greedy tokens for each prompt; the integer pass runs teacher-forced on them and at every
 * position after the prompt we ask whether it picks the same next token, and where the exact model's token ranks. Then it
 * writes on its own, for a person to read, and reports the operations it did as trace tokens at the unit's costs.
 */
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "../../src/text/q2b.h"
#include "../../src/text/q2b_fix.h"
#include "../../src/text/tok.h"

namespace {

std::string shown(const std::string &s) {
    std::string out;
    for (char c : s) {
        if (c == '\n') out += "\\n";
        else out += c;
    }
    return out;
}

template <typename T>
int32_t argmax(const std::vector<T> &v) {
    int32_t best = 0;
    for (int32_t k = 1; k < q2b::VOCAB; k++)
        if (v[static_cast<size_t>(k)] > v[static_cast<size_t>(best)]) best = k;
    return best;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 4) {
        fputs("usage: q2b_fix_check MODEL GEN PROMPTS_FILE [FREE_RUN_PROMPT ...]\n", stderr);
        return 2;
    }
    gguf::File f;
    gguf::open(f, argv[1]);
    Tokenizer tok(f);
    const int gen = atoi(argv[2]);
    std::vector<std::string> prompts;
    {
        std::ifstream in(argv[3]);
        std::string line;
        while (std::getline(in, line))
            if (!line.empty()) prompts.push_back(line);
    }
    std::vector<float> flog(q2b::VOCAB);
    std::vector<int32_t> ilog(q2b::VOCAB);
    q2b::Fix fix(f);
    uint64_t n = 0, same = 0, top5 = 0;
    double rank_sum = 0;
    {
        q2b::Ref exact(f);
        for (const std::string &u : prompts) {
            exact.reset();
            fix.reset();
            std::vector<int32_t> toks = tok.encode(Tokenizer::chat_prompt(u));
            const size_t prompt = toks.size();
            for (size_t i = 0; i < toks.size() && i < prompt + static_cast<size_t>(gen) - 1; i++) {
                exact.step(toks[i], flog.data());
                fix.step(toks[i], ilog.data());
                if (i + 1 < prompt) continue;
                const int32_t want = argmax(flog);
                const int32_t wv = ilog[static_cast<size_t>(want)];
                uint64_t above = 0;
                for (int32_t k = 0; k < q2b::VOCAB; k++) above += ilog[static_cast<size_t>(k)] > wv;
                n++;
                same += above == 0;
                top5 += above < 5;
                rank_sum += static_cast<double>(above);
                toks.push_back(want);
                if (tok.is_eog(want)) break;
            }
            printf("  so far: %lu positions judged, top-1 %.2f %%\n", static_cast<unsigned long>(n), 100.0 * static_cast<double>(same) / static_cast<double>(n ? n : 1));
            fflush(stdout);
        }
    }
    printf("host, 2B in integers against the exact 2B, %zu prompts, %lu positions: top-1 %.2f %%, top-5 %.2f %%, misses %lu, mean rank %.3f\n", prompts.size(),
           static_cast<unsigned long>(n), 100.0 * static_cast<double>(same) / static_cast<double>(n), 100.0 * static_cast<double>(top5) / static_cast<double>(n),
           static_cast<unsigned long>(n - same), rank_sum / static_cast<double>(n));
    const q2b::FixCost c = fix.cost();
    const double pos = static_cast<double>(c.positions);
    printf("operations as trace tokens per inner token, at the unit's costs (a model, not a trace), over %lu positions:\n", static_cast<unsigned long>(c.positions));
    printf("  matrix rows %.3f M, quantize %.3f M, elementwise %.3f M, GDN state %.3f M, attention %.3f M, rms %.3f M; total %.3f M\n", static_cast<double>(c.dot) / pos / 1e6,
           static_cast<double>(c.quant) / pos / 1e6, static_cast<double>(c.elem) / pos / 1e6, static_cast<double>(c.state) / pos / 1e6, static_cast<double>(c.attn) / pos / 1e6,
           static_cast<double>(c.rms) / pos / 1e6, static_cast<double>(c.total()) / pos / 1e6);
    const q2b::FixWidths w = fix.widths();
    printf("largest magnitudes: matrix output %ld (Q.16), block product %ld, state %ld (Q.32)\n", static_cast<long>(w.y), static_cast<long>(w.t), static_cast<long>(w.state_q32));
    printf("32-bit wraps against the exact value: gemv acc %lu, kv24 %lu, v24 %lu, rms24 %lu\n", static_cast<unsigned long>(w.wrap_acc), static_cast<unsigned long>(w.wrap_kv24),
           static_cast<unsigned long>(w.wrap_v24), static_cast<unsigned long>(w.wrap_rms24));
    for (int a = 4; a < argc; a++) {
        fix.reset();
        std::vector<int32_t> t = tok.encode(Tokenizer::chat_prompt(argv[a]));
        const size_t p = t.size();
        std::string text;
        for (size_t i = 0; i < t.size() && i < p + 63; i++) {
            fix.step(t[i], ilog.data());
            if (i + 1 >= p) {
                const int32_t next = argmax(ilog);
                text += tok.decode(next);
                t.push_back(next);
                if (tok.is_eog(next)) break;
            }
        }
        printf("[%s]\n  integers: %s\n", argv[a], shown(text).c_str());
        fflush(stdout);
    }
    return 0;
}
