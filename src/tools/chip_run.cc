// The 35B model file, running the 2B inside itself: each token goes through the computer's layers in the file, the output head picks the next
// one, and that is fed back. This program links the forward pass, the file reader and the tokenizer and nothing else: no interpreter, no NPU,
// no circuit, no reference. The system prompt and its context are in the file; prompts come here, and the context stays through the turns.
//
//   chip_run MODEL.gguf "PROMPT" [--gen N] [--chat] [--hashlog PATH] [--max TOKENS]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "../exec/tforward.h"
#include "../text/tok.h"

extern "C" {
#include "../guest64/q2b64_map.h"
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fputs("usage: chip_run MODEL.gguf PROMPT [--gen N] [--chat] [--hashlog PATH] [--max TOKENS]\n", stderr);
        return 2;
    }
    long long max_steps = 1LL << 60, gen = 64;
    bool chat = false;
    FILE *hlog = nullptr;
    for (int i = 3; i < argc; i++) {
        if (!std::strcmp(argv[i], "--chat")) chat = true;
        else if (i + 1 < argc && !std::strcmp(argv[i], "--max")) max_steps = std::atoll(argv[++i]);
        else if (i + 1 < argc && !std::strcmp(argv[i], "--gen")) gen = std::atoll(argv[++i]);
        else if (i + 1 < argc && !std::strcmp(argv[i], "--hashlog")) hlog = std::fopen(argv[++i], "w");
    }
    const auto t_load = std::chrono::steady_clock::now();
    TForward tf(argv[1]);
    Tokenizer tok(tf.file());
    const uint32_t go = (uint32_t)tf.meta("qwen35moe.chip.go_token"), byte0 = (uint32_t)tf.meta("qwen35moe.chip.byte_token");
    uint64_t pos = (uint64_t)tf.meta("qwen35moe.chip.system_tokens"), prompt_end = pos;
    std::printf("host: loaded the model in %.1f s: %d computer layers, %llu nonzero weights, the system prompt's %llu tokens in the context, %d positions\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count(), tf.layers(), (unsigned long long)tf.nnz(), (unsigned long long)pos, Q64_CTX);
    std::fflush(stdout);
    uint64_t hash = 0xcbf29ce484222325ull;
    const auto mix = [&](uint32_t t) { hash = (hash ^ t) * 0x100000001b3ull; };
    mix(go);
    uint32_t token = go;
    long long n = 1, last_mark = 1;
    int turns = 0, got = 0;
    uint32_t bytes = 0;
    std::string prompt = argv[2];
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
            std::printf("host, inside the model: %lld trace tokens, %.0f tok/s, %.1f s, position %llu\n", n, (double)n / el, el, (unsigned long long)pos);
            std::fflush(stdout);
        }
        if (next >= byte0 && next < byte0 + 512) {
            const uint32_t b = next - byte0;
            if (b == Q64_READY) {
                if (turns && !chat) break;
                if (turns) {
                    std::printf("\n> ");
                    std::fflush(stdout);
                    if (!std::getline(std::cin, prompt)) break;
                }
                const std::string text = (turns ? std::string("<|im_end|>\n") : std::string()) + Tokenizer::chat_prompt(prompt);
                const std::vector<int32_t> toks = tok.encode(text);
                int32_t *vm = tf.bank_words(1);
                for (size_t i = 0; i < toks.size(); i++) vm[VM_TOKS + pos + i] = toks[i];
                prompt_end = pos + toks.size();
                vm[JOB_N] = (int32_t)prompt_end;
                vm[JOB_GEN] = (int32_t)gen;
                vm[JOB_SEQ] += 1;
                turns++;
                last_mark = n;
                std::printf("host: turn %d: %zu tokens in, positions %llu to %llu; the first reply token comes after that many positions of trace\n", turns, toks.size(), (unsigned long long)pos, (unsigned long long)prompt_end);
                std::fflush(stdout);
            } else {
                bytes |= b << (8 * got);
                if (++got == 3) {
                    const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    std::printf("host, inside the model: position %llu done, %lld trace tokens for it, %.0f tok/s\n", (unsigned long long)pos, n - last_mark, (double)n / el);
                    if (pos + 1 >= prompt_end) std::printf("%s", tok.decode((int32_t)bytes).c_str());
                    std::fflush(stdout);
                    last_mark = n;
                    pos++;
                    bytes = 0;
                    got = 0;
                }
            }
        }
        token = next;
    }
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\nhost, inside the model: %lld trace tokens, hash %016llx, %.2f s = %.0f tok/s, one thread, full output scans %llu\n", n, (unsigned long long)hash, dt, (double)n / dt, (unsigned long long)tf.full_scans());
    return 0;
}
