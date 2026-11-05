/* The 2B's host reference against real llama.cpp, tensor by tensor and token by token. Test only, on the host.
 *
 *   q2b_check MODEL ORACLE_TXT ORACLE_DUMP --prompt TEXT
 *
 * ORACLE_TXT and ORACLE_DUMP come from llama_dump. The reference is teacher-forced on llama.cpp's own token sequence, so
 * one near tie cannot end the comparison: at every position it reports whether the two models pick the same next token.
 * llama.cpp rounds its activations to 8 bits inside Q8_0 products and this reference does not, so tensors differ a
 * little and a near tie may flip: that is data, not a pass. It then writes its own free-running text.
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "../../src/text/q2b.h"
#include "../../src/text/tok.h"

namespace {

struct OracleRow {
    int32_t tok, next;
    float top, margin;
};

std::vector<OracleRow> read_oracle(const char *path) {
    std::vector<OracleRow> rows;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("pos ", 0) != 0) continue;
        std::istringstream ss(line);
        std::string w;
        size_t pos;
        OracleRow r{};
        ss >> w >> pos >> w >> r.tok >> w >> r.next >> w >> r.top >> w >> r.margin;
        if (ss) rows.push_back(r);
    }
    return rows;
}

using Tensors = std::map<std::string, std::vector<float>>;

std::map<uint32_t, Tensors> read_dump(const char *path) {
    std::map<uint32_t, Tensors> all;
    FILE *f = fopen(path, "rb");
    if (!f) return all;
    char magic[4];
    while (fread(magic, 1, 4, f) == 4) {
        uint32_t pos, nlen, type, nd;
        uint64_t dims[4], nb;
        if (fread(&pos, 4, 1, f) != 1 || fread(&nlen, 4, 1, f) != 1) break;
        std::string name(nlen, '\0');
        if (fread(&name[0], 1, nlen, f) != nlen) break;
        if (fread(&type, 4, 1, f) != 1 || fread(&nd, 4, 1, f) != 1 || fread(dims, 8, 4, f) != 4 || fread(&nb, 8, 1, f) != 1) break;
        std::vector<float> &d = all[pos][name];
        d.resize(nb / 4);
        if (fread(d.data(), 1, nb, f) != nb) break;
    }
    fclose(f);
    return all;
}

double nrmse(const float *a, const float *b, size_t n) {
    double e = 0, s = 0;
    for (size_t i = 0; i < n; i++) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        e += d * d;
        s += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    return s > 0 ? std::sqrt(e / s) : std::sqrt(e);
}

int32_t argmax(const std::vector<float> &v) {
    int32_t best = 0;
    for (int32_t k = 1; k < q2b::VOCAB; k++)
        if (v[static_cast<size_t>(k)] > v[static_cast<size_t>(best)]) best = k;
    return best;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 6 || strcmp(argv[4], "--prompt") != 0) {
        fputs("usage: q2b_check MODEL ORACLE_TXT ORACLE_DUMP --prompt TEXT\n", stderr);
        return 2;
    }
    gguf::File f;
    gguf::open(f, argv[1]);
    Tokenizer tok(f);
    const std::vector<OracleRow> oracle = read_oracle(argv[2]);
    auto dump = read_dump(argv[3]);
    q2b::Ref ref(f);
    std::vector<float> logits(q2b::VOCAB);
    /* teacher-forced on llama.cpp's sequence */
    int same = 0, differ = 0, seen_tensors = 0;
    double worst = 0, sum_top = 0;
    std::string worst_name;
    for (size_t i = 0; i < oracle.size(); i++) {
        auto it = dump.find(static_cast<uint32_t>(i));
        const Tensors *pos = it == dump.end() ? nullptr : &it->second;
        q2b::Tap tap = [&](const char *name, int il, const float *d, size_t n) {
            if (!pos) return;
            const std::string key = il >= 0 ? std::string(name) + "-" + std::to_string(il) : std::string(name);
            auto rec = pos->find(key);
            if (rec == pos->end() || rec->second.size() != n) return;
            seen_tensors++;
            const double e = nrmse(d, rec->second.data(), n);
            if (e > worst) {
                worst = e;
                worst_name = key + " at position " + std::to_string(i);
            }
        };
        ref.step(oracle[i].tok, logits.data(), &tap);
        const int32_t mine = argmax(logits);
        sum_top += std::fabs(static_cast<double>(logits[static_cast<size_t>(oracle[i].next)]) - static_cast<double>(oracle[i].top));
        if (mine == oracle[i].next) {
            same++;
        } else {
            differ++;
            printf("  position %zu: llama.cpp next %d (margin %.4f), reference next %d\n", i, oracle[i].next, static_cast<double>(oracle[i].margin), mine);
        }
    }
    printf("teacher-forced on llama.cpp's %zu positions: same next token %d, different %d; mean |top logit difference| %.4f\n", oracle.size(), same, differ,
           oracle.empty() ? 0.0 : sum_top / static_cast<double>(oracle.size()));
    printf("tensors compared with llama.cpp: %d, worst nrmse %.3e (%s)\n", seen_tensors, worst, worst_name.c_str());
    /* free-running, greedy */
    ref.reset();
    size_t prompt_n = 0;
    std::vector<int32_t> toks = tok.encode(argv[5]);
    prompt_n = toks.size();
    const size_t limit = oracle.size() + 1;
    std::string text;
    for (size_t i = 0; i < toks.size() && i < limit; i++) {
        ref.step(toks[i], logits.data());
        if (i + 1 >= prompt_n) {
            const int32_t next = argmax(logits);
            text += tok.decode(next);
            toks.push_back(next);
            if (tok.is_eog(next)) break;
        }
    }
    fputs("TEXT: ", stdout);
    for (char c : text) {
        if (c == '\n') fputs("\\n", stdout);
        else putchar(c);
    }
    putchar('\n');
    return differ != 0;
}
