#include "tok.h"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>

namespace {

enum { C_L = 1, C_M = 2, C_N = 4, C_S = 8 };

struct Range {
    uint32_t first, last;
    uint8_t cls;
};

const Range RANGES[] = {
#include "unicode_tab.inc"
};
const size_t NRANGES = sizeof RANGES / sizeof RANGES[0];

int class_of(uint32_t cp) {
    size_t lo = 0, hi = NRANGES;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (cp > RANGES[mid].last) lo = mid + 1;
        else hi = mid;
    }
    return lo < NRANGES && cp >= RANGES[lo].first ? RANGES[lo].cls : 0;
}

/* One code point of the text and its byte span. Bad UTF-8 stands alone as one byte, class 0. */
struct Cp {
    uint32_t cp;
    uint32_t off, len;
    int cls;
};

std::vector<Cp> decode_utf8(const std::string &s) {
    std::vector<Cp> v;
    v.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        uint8_t c = static_cast<uint8_t>(s[i]);
        uint32_t cp = c, len = 1;
        if (c >= 0xF0) cp = c & 7u, len = 4;
        else if (c >= 0xE0) cp = c & 15u, len = 3;
        else if (c >= 0xC0) cp = c & 31u, len = 2;
        bool ok = i + len <= s.size();
        for (uint32_t k = 1; ok && k < len; k++) {
            uint8_t d = static_cast<uint8_t>(s[i + k]);
            if ((d & 0xC0) != 0x80) ok = false;
            else cp = cp << 6 | (d & 63u);
        }
        if (!ok) cp = c, len = 1;
        v.push_back({cp, static_cast<uint32_t>(i), len, ok || len == 1 ? class_of(cp) : 0});
        i += len;
    }
    return v;
}

void append_utf8(std::string &out, uint32_t cp) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | cp >> 6);
        out += static_cast<char>(0x80 | (cp & 63));
    } else {
        out += static_cast<char>(0xE0 | cp >> 12);
        out += static_cast<char>(0x80 | (cp >> 6 & 63));
        out += static_cast<char>(0x80 | (cp & 63));
    }
}

bool is_newline(const Cp &c) { return c.cp == '\r' || c.cp == '\n'; }
bool is_letter_or_mark(const Cp &c) { return c.cls & (C_L | C_M); }
bool is_word(const Cp &c) { return c.cls & (C_L | C_M | C_N); }
bool is_space(const Cp &c) { return c.cls & C_S; }
int lower_ascii(uint32_t cp) { return cp < 128 && cp >= 'A' && cp <= 'Z' ? static_cast<int>(cp) + 32 : static_cast<int>(cp); }

/* The length in code points of the pre-tokenizer's match at i, this alternation in this order:
 *   (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
 *   [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
 *   \p{N}
 *    ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
 *   \s*[\r\n]+
 *   \s+(?!\S)
 *   \s+                                                                                       */
size_t match_at(const std::vector<Cp> &t, size_t i) {
    const size_t n = t.size();
    if (t[i].cp == '\'' && i + 1 < n) {
        int a = lower_ascii(t[i + 1].cp);
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
        if (i + 2 < n) {
            int b = lower_ascii(t[i + 2].cp);
            if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return 3;
        }
    }
    /* optional prefix first, as a greedy regex tries it, then no prefix */
    if (!is_newline(t[i]) && !(t[i].cls & (C_L | C_N)) && i + 1 < n && is_letter_or_mark(t[i + 1])) {
        size_t j = i + 2;
        while (j < n && is_letter_or_mark(t[j])) j++;
        return j - i;
    }
    if (is_letter_or_mark(t[i])) {
        size_t j = i + 1;
        while (j < n && is_letter_or_mark(t[j])) j++;
        return j - i;
    }
    if (t[i].cls & C_N) return 1;
    {
        size_t j = i + (t[i].cp == ' ' ? 1 : 0);
        size_t k = j;
        while (k < n && !is_space(t[k]) && !is_word(t[k])) k++;
        if (k > j) {
            while (k < n && is_newline(t[k])) k++;
            return k - i;
        }
    }
    size_t run = i;
    while (run < n && is_space(t[run])) run++;
    if (run == i) return 1;  /* unreachable for valid classes, but every position must advance */
    for (size_t k = run; k > i; k--)
        if (is_newline(t[k - 1])) return k - i;
    if (run == n) return run - i;
    if (run - i >= 2) return run - i - 1;
    return run - i;
}

}  // namespace

Tokenizer::Tokenizer(const gguf::File &f) {
    auto tk = f.str_arrays.find("tokenizer.ggml.tokens");
    auto mg = f.str_arrays.find("tokenizer.ggml.merges");
    auto ty = f.int_arrays.find("tokenizer.ggml.token_type");
    if (tk == f.str_arrays.end() || mg == f.str_arrays.end() || ty == f.int_arrays.end()) {
        fprintf(stderr, "tokenizer: the GGUF lacks tokens, merges or token types\n");
        exit(1);
    }
    pieces_ = tk->second;
    is_special_.assign(pieces_.size(), 0);
    vocab_.reserve(pieces_.size() * 2);
    for (size_t i = 0; i < pieces_.size(); i++) {
        vocab_[pieces_[i]] = static_cast<int32_t>(i);
        /* llama.cpp's token types: 3 control, 4 user defined */
        if (ty->second[i] == 3 || ty->second[i] == 4) {
            is_special_[i] = 1;
            specials_.push_back(pieces_[i]);
        }
    }
    std::stable_sort(specials_.begin(), specials_.end(), [](const std::string &a, const std::string &b) { return a.size() > b.size(); });
    ranks_.reserve(mg->second.size() * 2);
    for (size_t i = 0; i < mg->second.size(); i++) ranks_[mg->second[i]] = static_cast<int32_t>(i);
    /* GPT-2's byte alphabet */
    bool direct[256] = {};
    for (int b = '!'; b <= '~'; b++) direct[b] = true;
    for (int b = 0xA1; b <= 0xAC; b++) direct[b] = true;
    for (int b = 0xAE; b <= 0xFF; b++) direct[b] = true;
    uint32_t extra = 0;
    for (int b = 0; b < 256; b++) {
        uint32_t cp = direct[b] ? static_cast<uint32_t>(b) : 256 + extra++;
        std::string u;
        append_utf8(u, cp);
        byte_to_utf8_[b] = u;
        utf8_to_byte_[u] = static_cast<uint8_t>(b);
    }
    eos_ = static_cast<int32_t>(f.get_int("tokenizer.ggml.eos_token_id"));
    im_end_ = id_of("<|im_end|>");
    endoftext_ = id_of("<|endoftext|>");
}

int32_t Tokenizer::id_of(const std::string &piece) const {
    auto it = vocab_.find(piece);
    return it == vocab_.end() ? -1 : it->second;
}

void Tokenizer::bpe(const std::string &piece, std::vector<int32_t> &out) const {
    /* symbols are UTF-8 characters of the byte-alphabet form */
    std::vector<std::string> sym;
    sym.reserve(piece.size());
    for (unsigned char c : piece) sym.push_back(byte_to_utf8_[c]);
    while (sym.size() > 1) {
        int best = INT_MAX;
        size_t at = 0;
        for (size_t i = 0; i + 1 < sym.size(); i++) {
            auto it = ranks_.find(sym[i] + " " + sym[i + 1]);
            if (it != ranks_.end() && it->second < best) {
                best = it->second;
                at = i;
            }
        }
        if (best == INT_MAX) break;
        /* every occurrence of the winning pair merges in this pass, left to right */
        std::string a = sym[at], b = sym[at + 1];
        std::vector<std::string> next;
        next.reserve(sym.size());
        for (size_t i = 0; i < sym.size();) {
            if (i + 1 < sym.size() && sym[i] == a && sym[i + 1] == b) {
                next.push_back(a + b);
                i += 2;
            } else {
                next.push_back(sym[i]);
                i++;
            }
        }
        sym.swap(next);
    }
    for (const std::string &s : sym) {
        auto it = vocab_.find(s);
        if (it != vocab_.end()) {
            out.push_back(it->second);
        } else {
            /* a symbol with no token of its own falls apart into its bytes */
            for (size_t i = 0; i < s.size();) {
                size_t len = 1;
                unsigned char c = static_cast<unsigned char>(s[i]);
                if (c >= 0xC0) len = c >= 0xE0 ? 3 : 2;
                auto one = vocab_.find(s.substr(i, len));
                if (one != vocab_.end()) out.push_back(one->second);
                i += len;
            }
        }
    }
}

void Tokenizer::split_plain(const std::string &text, std::vector<int32_t> &out) const {
    if (text.empty()) return;
    std::vector<Cp> t = decode_utf8(text);
    for (size_t i = 0; i < t.size();) {
        size_t m = match_at(t, i);
        const Cp &end = t[i + m - 1];
        bpe(text.substr(t[i].off, end.off + end.len - t[i].off), out);
        i += m;
    }
}

std::vector<int32_t> Tokenizer::encode(const std::string &text, bool parse_special) const {
    std::vector<int32_t> out;
    if (!parse_special) {
        split_plain(text, out);
        return out;
    }
    size_t at = 0;
    while (at < text.size()) {
        size_t best_pos = std::string::npos;
        const std::string *best = nullptr;
        /* the earliest special wins; at one position, the longest (the list is longest first) */
        size_t lt = text.find('<', at);
        while (lt != std::string::npos && !best) {
            for (const std::string &s : specials_) {
                if (s.empty() || s[0] != '<') continue;
                if (text.compare(lt, s.size(), s) == 0) {
                    best = &s;
                    best_pos = lt;
                    break;
                }
            }
            if (!best) lt = text.find('<', lt + 1);
        }
        if (!best) {
            split_plain(text.substr(at), out);
            break;
        }
        split_plain(text.substr(at, best_pos - at), out);
        out.push_back(vocab_.at(*best));
        at = best_pos + best->size();
    }
    return out;
}

std::string Tokenizer::decode(int32_t id) const {
    if (id < 0 || id >= size()) return std::string();
    const std::string &p = pieces_[static_cast<size_t>(id)];
    if (is_special_[static_cast<size_t>(id)]) return p;
    std::string out;
    for (size_t i = 0; i < p.size();) {
        size_t len = 1;
        unsigned char c = static_cast<unsigned char>(p[i]);
        if (c >= 0xE0) len = 3;
        else if (c >= 0xC0) len = 2;
        auto it = utf8_to_byte_.find(p.substr(i, len));
        if (it != utf8_to_byte_.end()) out += static_cast<char>(it->second);
        else out += p.substr(i, len);
        i += len;
    }
    return out;
}

std::string Tokenizer::chat_prompt_tools(const std::string &user, const std::string &tool_json) {
    std::string p = "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n" + tool_json + "\n</tools>";
    p += "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n"
         "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\n"
         "multiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner "
         "<function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide "
         "optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, "
         "answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT><|im_end|>\n";
    return p + chat_prompt(user);
}

std::string Tokenizer::chat_prompt(const std::string &user, bool thinking) {
    std::string p = "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n";
    p += thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
    return p;
}
