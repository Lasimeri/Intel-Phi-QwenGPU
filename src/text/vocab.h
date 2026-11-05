// The bridge's view of the outer model's vocabulary, from the tokenizer: bridge.Text's printable tokens.
#ifndef TEXT_VOCAB_H
#define TEXT_VOCAB_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "../circuit/core.h"
#include "tok.h"

inline TextVocab make_text_vocab(const Tokenizer &tok) {
    TextVocab v;
    for (int b = 0; b < 256; b++) {
        const int32_t id = tok.byte_token(b);
        if (id < 0) {
            std::fprintf(stderr, "vocabulary: no token for byte %d\n", b);
            std::exit(1);
        }
        v.printable.push_back((uint32_t)id);
    }
    for (const char *s : {"<tool_response>", "</tool_response>", "<think>", "</think>"}) {
        const int32_t id = tok.id_of(s);
        if (id < 0) {
            std::fprintf(stderr, "vocabulary: no token %s\n", s);
            std::exit(1);
        }
        v.printable.push_back((uint32_t)id);
    }
    for (size_t p = 0; p < v.printable.size(); p++) v.printable_index[v.printable[p]] = (int)p;
    v.tco = (uint32_t)tok.id_of("<tool_call>");
    v.tcc = (uint32_t)tok.id_of("</tool_call>");
    v.nl = v.printable[10];
    const Tokenizer *t = &tok;
    v.row_bytes = [t](uint32_t q, uint8_t *row) {
        const std::string s = t->decode((int32_t)q);
        std::memset(row, 0, S_TEXT_ROW);
        std::memcpy(row, s.data(), s.size() < (size_t)S_TEXT_ROW ? s.size() : (size_t)S_TEXT_ROW);
    };
    return v;
}

#endif
