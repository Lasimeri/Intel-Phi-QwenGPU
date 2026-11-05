/* Qwen3.8-35B-A3B, arch qwen35moe: shapes read and checked against the GGUF, and the host reference.
 *
 * The reference is a TEST program. It streams Q8_0 weights from disk with pread, and defines the exact
 * order of every operation the card's VPU must reproduce:
 *   a row of a Q8_0 matrix:   for each 32-column block { inner = fold fma(q[i], x[i], inner) ascending;
 *                                                        y = fma(scale, inner, y) }
 *   a row of an F32 matrix:   y = fold fma(w[j], x[j], y) ascending
 * Rows are independent, so the result does not depend on threads or lanes. */
#ifndef TEXT_Q35_H
#define TEXT_Q35_H

#include <cstdint>
#include <functional>
#include <vector>

#include "gguf.h"

namespace q35 {

enum {
    D = 2048,        /* width */
    LAYERS = 40,
    VHEADS = 32,     /* GDN value heads */
    KHEADS = 16,     /* GDN key heads */
    ST = 128,        /* GDN state width */
    KEY_DIM = KHEADS * ST,
    VAL_DIM = VHEADS * ST,
    QKV = 2 * KEY_DIM + VAL_DIM,
    CONV = 4,
    QHEADS = 16,     /* attention query heads */
    KVHEADS = 2,     /* attention key and value heads */
    HD = 256,        /* attention head width */
    NEXP = 256,
    USED = 8,
    FF = 512,        /* width inside an expert */
    ROT = 64,        /* dims RoPE rotates */
    VOCAB = 248320,
};

constexpr float RMS_EPS = 1e-6f;

inline bool is_full_attention(int il) { return (il + 1) % 4 == 0; }

/* Dies unless the file's metadata and tensor shapes are exactly the model this code was written for. */
void check_shapes(const gguf::File &f);

/* The table of rotations: cos then sin of ROT/2 angles, per position. Built on the host, and shipped
 * to the card rather than computed twice. */
void rope_row(uint32_t pos, float cos_out[ROT / 2], float sin_out[ROT / 2]);

/* name, layer (-1 if none), data, count */
using Tap = std::function<void(const char *, int, const float *, size_t)>;

/* What a card run reports for one position, and what the reference recomputes to check it. */
struct Check {
    uint64_t x_hash[LAYERS + 1];  /* the residual after each layer; the last is after the output norm */
    uint64_t logits_hash;
    int32_t next, second;
    uint32_t top_bits, second_bits;
};

class Ref {
public:
    explicit Ref(const gguf::File &f);
    ~Ref();
    void reset();
    /* One token through the model at the next position; logits has VOCAB floats. */
    void step(int32_t token, float *logits, const Tap *tap = nullptr);
    const Check &check() const { return check_; }
    /* The experts each layer chose on the last step, for routing statistics. */
    const int32_t *route(int il) const { return &routes_[static_cast<size_t>(il) * USED]; }
    uint64_t bytes_read() const { return bytes_read_; }

private:
    struct Layer;
    void layer_gdn(int il, const float *h, float *out, const Tap *tap);
    void layer_full(int il, const float *h, float *out, const Tap *tap);
    void moe_ffn(int il, const float *h, float *out, const Tap *tap);
    void q8_gemv(const gguf::Tensor &t, uint64_t skip_rows, uint64_t rows, const float *x, float *y);
    void read_weights(const gguf::Tensor &t, uint64_t off, uint64_t n);

    const gguf::File &f_;
    std::vector<Layer> layers_;
    std::vector<float> scale_tab_;
    std::vector<uint8_t> buf_;
    std::vector<int32_t> routes_;
    std::vector<float> out_norm_;
    uint32_t pos_ = 0;
    uint64_t bytes_read_ = 0;
    Check check_{};
};

}  // namespace q35

#endif
