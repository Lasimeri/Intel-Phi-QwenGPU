// The model file running the 2B inside itself: each token goes through the computer's layers in the file, the output head
// picks the next one, and that is fed back. This program links the forward pass, the file reader and the tokenizer and
// nothing else: no machine, no circuit, no reference. The job (the prompt's tokens and the job words) is written into the
// words memory as the host writes it for the reference; the reply tokens come out as EMIT VLO VEND token triples.
//
//   gpu_run MODEL.gguf "PROMPT" [--tok TOKENIZER.gguf] [--gen N] [--max TOKENS] [--hashlog PATH]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../exec/tforward.h"
#include "../text/tok.h"
extern "C" {
#include "../guest/q2b_map.h"
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fputs("usage: gpu_run MODEL.gguf PROMPT [--tok TOKENIZER.gguf] [--gen N] [--max TOKENS] [--hashlog PATH]\n", stderr);
        return 2;
    }
    long long max_steps = 1LL << 60, gen = 8;
    std::string tokpath = argv[1];
    FILE *hlog = nullptr;
    for (int i = 3; i < argc; i++) {
        if (i + 1 < argc && !std::strcmp(argv[i], "--max")) max_steps = std::atoll(argv[++i]);
        else if (i + 1 < argc && !std::strcmp(argv[i], "--gen")) gen = std::atoll(argv[++i]);
        else if (i + 1 < argc && !std::strcmp(argv[i], "--tok")) tokpath = argv[++i];
        else if (i + 1 < argc && !std::strcmp(argv[i], "--hashlog")) hlog = std::fopen(argv[++i], "w");
    }
    const auto t_load = std::chrono::steady_clock::now();
    TForward tf(argv[1]);
    gguf::File tokf;
    gguf::open(tokf, tokpath.c_str());
    Tokenizer tok(tokf);
    const uint32_t next0 = (uint32_t)tf.meta("qwen35moe.chip.next0_token"), emit = (uint32_t)tf.meta("qwen35moe.chip.emit_token"), halt = (uint32_t)tf.meta("qwen35moe.chip.halt_token");
    const uint32_t vlo0 = (uint32_t)tf.meta("qwen35moe.chip.vlo0_token"), vhi0 = (uint32_t)tf.meta("qwen35moe.chip.vhi0_token"), vend0 = (uint32_t)tf.meta("qwen35moe.chip.vend0_token");
    std::printf("host: loaded the model in %.1f s: %d computer layers, hidden %d, %llu nonzero weights\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count(), tf.layers(), tf.hidden(),
                (unsigned long long)tf.nnz());
    std::fflush(stdout);
    // the job, as the host writes it for the reference (src/tools/q2b_gpu_check.cc)
    const std::vector<int32_t> toks = tok.encode(Tokenizer::chat_prompt(argv[2]), true);
    int32_t *m = reinterpret_cast<int32_t *>(tf.memory(MB_WORDS).data());
    for (size_t i = 0; i < toks.size(); i++) m[M_TOKS + i] = toks[i];
    m[JOB_SEQ] = 1;
    m[JOB_N] = (int32_t)toks.size();
    m[JOB_GEN] = (int32_t)gen;
    m[JOB_POS] = 0;
    m[JOB_PROMPT_N] = (int32_t)toks.size();
    m[JOB_MADE] = 0;
    m[JOB_NSTOPS] = 3;
    m[JOB_STOPS + 0] = tok.id_of("<|im_end|>");
    m[JOB_STOPS + 1] = tok.id_of("<|endoftext|>");
    m[JOB_STOPS + 2] = m[JOB_STOPS + 0];
    std::printf("host: prompt %zu tokens, %lld to generate\n", toks.size(), gen);
    std::fflush(stdout);
    uint64_t hash = 0xcbf29ce484222325ull;
    const auto mix = [&](uint32_t t) { hash = (hash ^ t) * 0x100000001b3ull; };
    uint32_t token = next0;
    mix(token);
    long long n = 1, last_mark = 1;
    int state = 0, positions = 0;
    uint32_t lo = 0;
    std::string text;
    const auto t0 = std::chrono::steady_clock::now();
    while (n < max_steps) {
        tf.step(token);
        const uint32_t next = tf.next_token();
        mix(next);
        n++;
        if (n % 4096 == 0 && hlog) {
            std::fprintf(hlog, "tokens %lld hash %016llx\n", n, (unsigned long long)hash);
            std::fflush(hlog);
        }
        if (n % 65536 == 0) {
            const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("host, inside the model: %lld trace tokens, %.0f tok/s, %.1f s, position %d\n", n, (double)n / el, el, m[JOB_POS]);
            std::fflush(stdout);
        }
        if (state == 0) {
            if (next == emit) state = 1;
        } else if (state == 1) {
            if (next < vhi0) lo = next - vlo0;
            state = 2;
        } else {
            const uint32_t id = lo | (next - vend0) << 16;
            state = 0;
            positions++;
            const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("host, inside the model: position %d done, token %u '%s', %lld trace tokens for it, %.0f tok/s\n", positions, id, tok.decode((int32_t)id).c_str(), n - last_mark, (double)n / el);
            std::fflush(stdout);
            last_mark = n;
            if (positions > (int)toks.size()) text += tok.decode((int32_t)id);
        }
        if (next == halt) break;
        token = next;
    }
    const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("host: %lld trace tokens, %.1f s, %.0f tok/s, hash %016llx, %d positions, %llu full scans of the output head\n%s\n", n, el, (double)n / el, (unsigned long long)hash, positions,
                (unsigned long long)tf.full_scans(), text.c_str());
    if (hlog) std::fclose(hlog);
    return 0;
}
