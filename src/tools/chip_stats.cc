// Where a token's work goes in the merged file, on the host: what the card's forward is laid out from. Runs the real job (the same prompt protocol
// as chip_run) with a probe in the forward; it computes nothing of its own.
//
//   chip_stats MODEL.gguf "PROMPT" TOKENS [WINDOW]
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "../exec/tforward.h"
#include "../text/tok.h"

extern "C" {
#include "../guest64/q2b64_map.h"
}

struct ChipStats : TProbe {
    struct Sub {
        uint64_t hn_nz = 0, hn_lines = 0, chg_sub = 0, chg_tok = 0, chg_lines_sub = 0, x_touched = 0, y_lines = 0, active_nnz = 0, nnz = 0, slices = 0, slices_live = 0, chunks = 0, chunks_dirty = 0, units = 0, mixnz = 0;
        std::vector<float> last;
    };
    TForward &tf;
    std::vector<Sub> sub;
    std::vector<float> prev;
    uint64_t tokens = 0;
    explicit ChipStats(TForward &t) : tf(t), sub((size_t)t.layers() * 2), prev((size_t)t.hidden(), 0.0f) {
        for (Sub &s : sub) s.last.assign((size_t)t.hidden(), 0.0f);
    }
    std::map<const void *, std::vector<std::vector<uint32_t>>> cols_of;
    void chunks(Sub &s, const TForward::Sparse &m, const float *hn, const std::vector<char> &changed) {
        s.nnz += m.col.size();
        std::vector<std::vector<uint32_t>> &all = cols_of[&m];
        if (all.empty())
            for (size_t k = 0; k < m.row.size(); k += 16) {
                const size_t e = std::min(k + 16, m.row.size());
                std::vector<uint32_t> cols(m.col.begin() + m.start[k], m.col.begin() + m.start[e]);
                std::sort(cols.begin(), cols.end());
                cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
                all.push_back(std::move(cols));
            }
        for (const std::vector<uint32_t> &cols : all) {
            bool dirty = false;
            s.slices += cols.size();
            for (uint32_t c : cols) {
                s.slices_live += hn[c] != 0.0f;
                dirty = dirty || changed[c];
            }
            s.chunks++;
            s.chunks_dirty += dirty;
        }
        for (uint32_t c : m.col) s.active_nnz += hn[c] != 0.0f;
    }
    void on_hn(int il, int part, const float *hn, const float *) override {
        Sub &s = sub[(size_t)(il * 2 + part)];
        const int h = tf.hidden();
        std::vector<char> ch_tok((size_t)h, 0);
        std::vector<char> line((size_t)h / 16, 0), cline((size_t)h / 16, 0);
        for (int i = 0; i < h; i++) {
            if (hn[i] != 0.0f) {
                s.hn_nz++;
                line[(size_t)i / 16] = 1;
            }
            if (hn[i] != prev[(size_t)i]) {
                s.chg_sub++;
                cline[(size_t)i / 16] = 1;
            }
            if (hn[i] != s.last[(size_t)i]) {
                s.chg_tok++;
                ch_tok[(size_t)i] = 1;
            }
        }
        for (size_t i = 0; i < line.size(); i++) {
            s.hn_lines += (uint64_t)line[i];
            s.chg_lines_sub += (uint64_t)cline[i];
        }
        std::memcpy(s.last.data(), hn, (size_t)h * 4);
        std::memcpy(prev.data(), hn, (size_t)h * 4);
        const TForward::Layer &L = tf.ls_[(size_t)il];
        if (part == 0) {
            chunks(s, L.qkv, hn, ch_tok);
            chunks(s, L.gate, hn, ch_tok);
            chunks(s, L.alpha, hn, ch_tok);
            chunks(s, L.beta, hn, ch_tok);
        } else {
            chunks(s, L.f_gate, hn, ch_tok);
            chunks(s, L.f_up, hn, ch_tok);
        }
    }
    void on_delta(int il, int part, const float *y) override {
        Sub &s = sub[(size_t)(il * 2 + part)];
        const int h = tf.hidden();
        std::vector<char> line((size_t)h / 16, 0);
        for (int i = 0; i < h; i++)
            if (y[i] != 0.0f) {
                s.x_touched++;
                line[(size_t)i / 16] = 1;
            }
        for (char c : line) s.y_lines += (uint64_t)c;
    }
    void on_units(int il, const float *xs, int n) override {
        Sub &s = sub[(size_t)(il * 2 + 1)];
        for (int i = 0; i < n; i++) s.units += xs[i] != 0.0f;
    }
    void on_mix(int il, const float *mix, int n) override {
        Sub &s = sub[(size_t)(il * 2)];
        for (int i = 0; i < n; i++) s.mixnz += mix[i] != 0.0f;
    }
    void report(uint64_t from, uint64_t to) {
        const double n = (double)tokens;
        std::printf("tokens %llu to %llu; per token: sublayer, hn nonzero dims / lines, hn dims changed since the last sublayer / lines, changed since this sublayer last token, y dims / lines, units or mix nonzero, matrix nnz, live nnz, slices, live slices, chunks, chunks with a changed input\n", (unsigned long long)from, (unsigned long long)to);
        Sub t;
        for (size_t i = 0; i < sub.size(); i++) {
            const Sub &s = sub[i];
            std::printf("%3zu%c %6.0f %5.0f  %6.1f %5.1f  %7.1f  %6.1f %5.1f  %6.1f  %7.0f %7.0f %7.0f %7.0f %5.0f %6.1f\n", i / 2, i & 1 ? 'F' : 'A', s.hn_nz / n, s.hn_lines / n, s.chg_sub / n, s.chg_lines_sub / n, s.chg_tok / n, s.x_touched / n, s.y_lines / n, (i & 1 ? s.units : s.mixnz) / n, s.nnz / n, s.active_nnz / n, s.slices / n, s.slices_live / n, s.chunks / n, s.chunks_dirty / n);
            t.hn_nz += s.hn_nz; t.hn_lines += s.hn_lines; t.chg_sub += s.chg_sub; t.chg_lines_sub += s.chg_lines_sub; t.chg_tok += s.chg_tok; t.x_touched += s.x_touched; t.y_lines += s.y_lines; t.units += s.units; t.mixnz += s.mixnz; t.nnz += s.nnz; t.active_nnz += s.active_nnz; t.slices += s.slices; t.slices_live += s.slices_live; t.chunks += s.chunks; t.chunks_dirty += s.chunks_dirty;
        }
        std::printf("ALL  hn lines %.0f, hn lines changed since the last sublayer %.1f, hn dims changed since last token %.0f, y dims %.0f, y lines %.0f, units %.0f, mix %.0f, nnz %.0f, live nnz %.0f, slices %.0f, live %.0f, chunks %.0f, dirty %.0f\n", t.hn_lines / n, t.chg_lines_sub / n, t.chg_tok / n, t.x_touched / n, t.y_lines / n, (double)t.units / n, (double)t.mixnz / n, t.nnz / n, t.active_nnz / n, t.slices / n, t.slices_live / n, t.chunks / n, t.chunks_dirty / n);
        std::fflush(stdout);
        for (Sub &s : sub) {
            std::vector<float> keep = std::move(s.last);
            s = Sub();
            s.last = std::move(keep);
        }
        tokens = 0;
    }
};

int main(int argc, char **argv) {
    if (argc < 4) return 2;
    const long long total = std::atoll(argv[3]), window = argc > 4 ? std::atoll(argv[4]) : total;
    TForward tf(argv[1]);
    Tokenizer tok(tf.file());
    ChipStats st(tf);
    tf.probe(&st);
    const uint32_t go = (uint32_t)tf.meta("qwen35moe.chip.go_token"), byte0 = (uint32_t)tf.meta("qwen35moe.chip.byte_token");
    uint64_t pos = (uint64_t)tf.meta("qwen35moe.chip.system_tokens");
    uint32_t token = go;
    int turns = 0;
    long long last = 1;
    for (long long n = 1; n < total; n++) {
        tf.step(token);
        st.tokens++;
        const uint32_t next = tf.next_token();
        if (next >= byte0 && next < byte0 + 512 && next - byte0 == Q64_READY && !turns) {
            const std::vector<int32_t> toks = tok.encode(Tokenizer::chat_prompt(argv[2]));
            int32_t *vm = tf.bank_words(1);
            for (size_t i = 0; i < toks.size(); i++) vm[VM_TOKS + pos + i] = toks[i];
            vm[JOB_N] = (int32_t)(pos + toks.size());
            vm[JOB_GEN] = 64;
            vm[JOB_SEQ] += 1;
            turns++;
            std::printf("READY at token %lld\n", n);
        }
        if (n % window == 0) {
            st.report((uint64_t)last, (uint64_t)n);
            last = n;
        }
        token = next;
    }
    return 0;
}
