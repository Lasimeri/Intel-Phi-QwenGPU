/* The 2B in integers: stage 2 of the goal. Host, a test program: it decides whether the model survives the arithmetic
 * a compiled vector unit can do, and counts that unit's operations, before any of the unit is built.
 *
 * No float touches an activation. Constants from the GGUF and the function tables are made once at load, as a chip's
 * tables are. Formats: activations Q.16 in int32; a matrix input is int8 with one Q.24 scale per 32 columns; a GDN
 * q, k, v and an attention q, k, v are int8 with one Q.24 scale per vector; GDN state is int16 with one power-of-two
 * scale per head and layer; probabilities are u8. */
#ifndef TEXT_Q2B_FIX_H
#define TEXT_Q2B_FIX_H

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "gguf.h"
#include "q2b.h"

namespace q2b {

/* Trace tokens the operations would take at upstream's costs (docs/vx.md), widened to Q8: a row costs 2 tokens per 32
 * columns, because Q8 has a scale per 32 columns. A model of the unit, not a machine's trace. */
struct FixCost {
    uint64_t positions = 0;
    uint64_t dot = 0;     /* matrix rows */
    uint64_t quant = 0;   /* VQ, 1 per 2 elements, and a guest division per scale */
    uint64_t elem = 0;    /* VEW: norms, conv, gated products */
    uint64_t state = 0;   /* GDN state reads and updates */
    uint64_t attn = 0;    /* attention scores, softmax, p.V */
    uint64_t rms = 0;     /* VRED and the guest's square root and division */
    uint64_t total() const { return dot + quant + elem + state + attn + rms; }
};

/* The largest magnitudes seen, to size the datapath from data, and how often a 32-bit lane value wrapped where the
 * exact value needed more (0 means the machine's registers lost nothing on these prompts). */
struct FixWidths {
    int64_t y = 0, t = 0, state_q32 = 0;
    uint64_t wrap_acc = 0, wrap_kv24 = 0, wrap_v24 = 0, wrap_rms24 = 0;
};

using FixTap = std::function<void(const char *, int, const int32_t *, size_t)>;

/* What the pack needs from this file: the same tables and constants, for the card. */
void fix_make_tables();
const std::vector<int32_t> &fix_table(int which);  /* 0 silu, 1 sigmoid, 2 softplus, 3 exp */
const std::vector<int64_t> &fix_scale_table();
std::vector<int32_t> fix_read_q16(const gguf::File &f, const std::string &name);

class Fix {
public:
    explicit Fix(const gguf::File &f);
    ~Fix();
    void reset();
    /* One token at the next position; logits has VOCAB entries in Q.16. The tap sees tensors in Q.16 under the exact
     * reference's names, to find where the two part. */
    void step(int32_t token, int32_t *logits, const FixTap *tap = nullptr);
    const FixCost &cost() const { return cost_; }
    const FixWidths &widths() const { return widths_; }
    /* A GDN layer's state, and a full-attention layer's caches, to hand a context on to the guest's memories. */
    struct GdnState {
        const int16_t *state;
        const int *se;
        const int64_t *smax;
        const int32_t *win;
    };
    struct AttnState {
        const int8_t *kc, *vc;
        const int32_t *ks, *vs;
        size_t slots;
    };
    GdnState gdn_state(int il) const;
    AttnState attn_state(int il) const;

private:
    struct Layer;
    struct BlockAct;
    void quantize_blocks(const int32_t *v, size_t n, BlockAct &a);
    void gemv(const gguf::Tensor &t, uint64_t rows, const BlockAct &x, int32_t *y);
    void norm(const int32_t *x, size_t n, const int32_t *w, int32_t *out);
    void layer_gdn(int il, const BlockAct &h, int32_t *out);
    void layer_full(int il, const BlockAct &h, int32_t *out);
    void ffn(int il, const BlockAct &h, int32_t *out);

    const gguf::File &f_;
    const uint8_t *map_ = nullptr;
    std::vector<Layer> layers_;
    std::vector<int64_t> scale_q24_;   /* a block's f16 scale, in Q.24 */
    std::vector<int32_t> out_norm_;
    uint32_t pos_ = 0;
    FixCost cost_;
    FixWidths widths_;
    const FixTap *tap_ = nullptr;
};

}  // namespace q2b

#endif
