/* Qwen3.8-2B-Distill Q8_0, arch qwen35: the model that is to run inside the 35B. Shapes read and checked against the
 * GGUF, and the host reference, a TEST program: it defines what the fixed-point guest must approximate and what every
 * later stage is measured against. The arithmetic is the 35B reference's (q35.h), elementwise and in row order; the
 * model differs in 16 value heads instead of 32, 8 query heads instead of 16, one dense FFN instead of routed experts,
 * and logits tied to the embedding. */
#ifndef TEXT_Q2B_H
#define TEXT_Q2B_H

#include <cstdint>
#include <functional>
#include <vector>

#include "gguf.h"

namespace q2b {

enum {
    D = 2048,        /* width */
    LAYERS = 24,     /* block_count 25, the last is the draft block */
    VHEADS = 16,     /* GDN value heads */
    KHEADS = 16,     /* GDN key heads */
    ST = 128,        /* GDN state width */
    KEY_DIM = KHEADS * ST,
    VAL_DIM = VHEADS * ST,
    QKV = 2 * KEY_DIM + VAL_DIM,
    CONV = 4,
    QHEADS = 8,      /* attention query heads */
    KVHEADS = 2,     /* attention key and value heads */
    HD = 256,        /* attention head width */
    FF = 6144,       /* width inside the FFN */
    ROT = 64,        /* dims RoPE rotates */
    VOCAB = 248320,
};

constexpr float RMS_EPS = 1e-6f;

inline bool is_full_attention(int il) { return (il + 1) % 4 == 0; }

/* Dies unless the file's metadata and tensor shapes are exactly the model this code was written for. */
void check_shapes(const gguf::File &f);

/* name, layer (-1 if none), data, count */
using Tap = std::function<void(const char *, int, const float *, size_t)>;

/* What a vector unit could afford, to measure what the model tolerates before any of it is built. All 0: exact.
 * act_bits: every matrix's input, GDN's q, k, v, attention's q, k, v and p are rounded to this many bits, with one
 *           scale per vector, or per 32 columns (per_block), as llama.cpp's Q8_0 products do.
 * state_bits: GDN's state after each update, per head and layer, one scale.
 * fn_bits: inputs and outputs of silu, sigmoid, softplus and exp on a grid of 2^-fn_bits, as a table would give. */
struct Approx {
    int act_bits = 0;
    bool per_block = false;
    int state_bits = 0;
    int fn_bits = 0;
};

class Ref {
public:
    explicit Ref(const gguf::File &f, Approx a = Approx());
    ~Ref();
    void reset();
    /* One token through the model at the next position; logits has VOCAB floats. */
    void step(int32_t token, float *logits, const Tap *tap = nullptr);

private:
    struct Layer;
    void round_vec(float *x, size_t n) const;
    float grid(float x) const;
    void layer_gdn(int il, const float *h, float *out, const Tap *tap);
    void layer_full(int il, const float *h, float *out, const Tap *tap);
    void ffn(int il, const float *h, float *out, const Tap *tap);
    void q8_gemv(const gguf::Tensor &t, uint64_t rows, const float *x, float *y) const;

    const gguf::File &f_;
    Approx a_;
    const uint8_t *map_ = nullptr;
    std::vector<Layer> layers_;
    std::vector<float> scale_tab_;
    std::vector<float> out_norm_;
    uint32_t pos_ = 0;
};

}  // namespace q2b

#endif
