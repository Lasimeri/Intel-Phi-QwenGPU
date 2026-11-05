/* The tokenizer of the GGUF's qwen35 vocabulary: GPT-2 byte-level BPE, the qwen35 pre-tokenizer written
 * out as a hand matcher instead of a regex engine, special tokens matched literally first. */
#ifndef TEXT_TOK_H
#define TEXT_TOK_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "gguf.h"

class Tokenizer {
public:
    explicit Tokenizer(const gguf::File &f);

    /* parse_special: control and user-defined tokens in the text become their single id. */
    std::vector<int32_t> encode(const std::string &text, bool parse_special = true) const;
    /* The token's raw bytes, which may end inside a UTF-8 sequence; specials give their text. */
    std::string decode(int32_t id) const;
    int32_t size() const { return static_cast<int32_t>(pieces_.size()); }
    int32_t id_of(const std::string &piece) const;
    /* The token that is exactly this one byte. */
    int32_t byte_token(int byte) const { return id_of(byte_to_utf8_[byte & 255]); }
    bool is_eog(int32_t id) const { return id == eos_ || id == im_end_ || id == endoftext_; }

    /* The GGUF's chat template with one user message and thinking turned off, as its Jinja source
     * renders enable_thinking=false. With thinking on it opens "<think>\n" instead. */
    static std::string chat_prompt(const std::string &user, bool thinking = false);
    /* The same with one tool offered, as the Jinja source renders its tools block. */
    static std::string chat_prompt_tools(const std::string &user, const std::string &tool_json);

private:
    void split_plain(const std::string &text, std::vector<int32_t> &out) const;
    void bpe(const std::string &piece, std::vector<int32_t> &out) const;

    std::vector<std::string> pieces_;
    std::vector<std::string> specials_;  /* longest first */
    std::vector<uint8_t> is_special_;
    std::unordered_map<std::string, int32_t> vocab_;
    std::unordered_map<std::string, int32_t> ranks_;
    std::string byte_to_utf8_[256];
    std::unordered_map<std::string, uint8_t> utf8_to_byte_;
    int32_t eos_ = -1, im_end_ = -1, endoftext_ = -1;
};

#endif
