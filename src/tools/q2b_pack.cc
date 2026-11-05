/* GGUF Q8_0 -> the 2B's pack for the card (src/text/q2b_pack.h). No weight changes: its int8 values only move into tiles.
 *
 *   q2b_pack MODEL.gguf OUT.pack
 */
#include <sys/mman.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../text/gguf.h"
#include "../text/q2b.h"
#include "../text/q2b_fix.h"
#include "../text/q2b_pack.h"
#include "../text/q35.h"

static_assert(Q2B_D == q2b::D && Q2B_LAYERS == q2b::LAYERS && Q2B_VHEADS == q2b::VHEADS && Q2B_KHEADS == q2b::KHEADS && Q2B_ST == q2b::ST && Q2B_QHEADS == q2b::QHEADS &&
                  Q2B_KVHEADS == q2b::KVHEADS && Q2B_HD == q2b::HD && Q2B_FF == q2b::FF && Q2B_VOCAB == q2b::VOCAB && Q2B_CONV == q2b::CONV,
              "the pack's shape is the model's");

namespace {

FILE *out;
uint64_t at = 0;

void put(const void *p, uint64_t n) {
    if (fwrite(p, 1, n, out) != n) {
        perror("q2b_pack: write");
        exit(1);
    }
    at += n;
}

void align64() {
    static const char zero[64] = {0};
    if (at % 64) put(zero, 64 - at % 64);
}

uint64_t put_array(const std::vector<int32_t> &v) {
    align64();
    const uint64_t off = at;
    put(v.data(), v.size() * sizeof(int32_t));
    return off;
}

uint64_t put_q16(const gguf::File &f, const std::string &name) {
    return put_array(q2b::fix_read_q16(f, name));
}

Q2bMat put_mat(const gguf::File &f, const uint8_t *map, const std::string &name) {
    const gguf::Tensor &t = f.tensor(name);
    const uint64_t cols = t.dims[0], rows = t.dims[1], blocks = cols / 32, row_bytes = gguf::row_bytes(gguf::T_Q8_0, cols);
    if (t.type != gguf::T_Q8_0 || cols % 32 || rows % 16) {
        fprintf(stderr, "q2b_pack: %s is not tileable\n", name.c_str());
        exit(1);
    }
    align64();
    const Q2bMat m{at, static_cast<uint32_t>(rows), static_cast<uint32_t>(cols)};
    const std::vector<int64_t> &scale = q2b::fix_scale_table();
    const uint8_t *raw = map + t.offset;
    std::vector<uint8_t> group(blocks * Q2B_TILE);
    for (uint64_t g = 0; g < rows / 16; g++) {
#pragma omp parallel for schedule(static)
        for (uint64_t b = 0; b < blocks; b++) {
            uint8_t *tile = &group[b * Q2B_TILE];
            for (uint64_t r = 0; r < 16; r++) {
                const uint8_t *blk = raw + (g * 16 + r) * row_bytes + b * 34;
                uint16_t h;
                memcpy(&h, blk, 2);
                const int64_t s = scale[h];
                if (s < 0 || s > INT32_MAX) {
                    fprintf(stderr, "q2b_pack: %s has a scale over the card's 32 bits\n", name.c_str());
                    exit(1);
                }
                const int32_t s32 = static_cast<int32_t>(s);
                memcpy(tile + 512 + 4 * r, &s32, 4);
                for (uint64_t i = 0; i < 32; i++) tile[16 * i + r] = blk[2 + i];
            }
        }
        put(group.data(), group.size());
    }
    return m;
}

std::string blk(int il, const char *suffix) { return "blk." + std::to_string(il) + "." + suffix; }

}  // namespace

int main(int argc, char **argv) {
    if (argc != 3) {
        fputs("usage: q2b_pack MODEL.gguf OUT.pack\n", stderr);
        return 2;
    }
    gguf::File f;
    gguf::open(f, argv[1]);
    q2b::check_shapes(f);
    void *mapped = mmap(nullptr, f.size, PROT_READ, MAP_SHARED, f.fd, 0);
    if (mapped == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    const uint8_t *map = static_cast<const uint8_t *>(mapped);
    out = fopen(argv[2], "wb");
    if (!out) {
        perror(argv[2]);
        return 1;
    }
    Q2bPack h;
    memset(&h, 0, sizeof h);
    h.magic = Q2B_MAGIC;
    put(&h, sizeof h);
    q2b::fix_make_tables();
    h.silu = put_array(q2b::fix_table(0));
    h.sig = put_array(q2b::fix_table(1));
    h.softplus = put_array(q2b::fix_table(2));
    h.exp = put_array(q2b::fix_table(3));
    {
        std::vector<int32_t> rope(static_cast<size_t>(Q2B_ROPE_POS) * Q2B_ROT);
        for (uint32_t pos = 0; pos < Q2B_ROPE_POS; pos++) {
            float c[Q2B_ROT / 2], s[Q2B_ROT / 2];
            q35::rope_row(pos, c, s);
            for (int i = 0; i < Q2B_ROT / 2; i++) {
                rope[static_cast<size_t>(pos) * Q2B_ROT + static_cast<size_t>(i)] = static_cast<int32_t>(std::lrint(static_cast<double>(c[i]) * 32768.0));
                rope[static_cast<size_t>(pos) * Q2B_ROT + Q2B_ROT / 2 + static_cast<size_t>(i)] = static_cast<int32_t>(std::lrint(static_cast<double>(s[i]) * 32768.0));
            }
        }
        h.rope = put_array(rope);
    }
    h.out_norm = put_q16(f, "output_norm.weight");
    h.emb = put_mat(f, map, "token_embd.weight");
    for (int il = 0; il < Q2B_LAYERS; il++) {
        Q2bLayer &L = h.layers[il];
        L.attn_norm = put_q16(f, blk(il, "attn_norm.weight"));
        L.post_norm = put_q16(f, blk(il, "post_attention_norm.weight"));
        L.ffn_g = put_mat(f, map, blk(il, "ffn_gate.weight"));
        L.ffn_u = put_mat(f, map, blk(il, "ffn_up.weight"));
        L.ffn_d = put_mat(f, map, blk(il, "ffn_down.weight"));
        if (q2b::is_full_attention(il)) {
            L.q_norm = put_q16(f, blk(il, "attn_q_norm.weight"));
            L.k_norm = put_q16(f, blk(il, "attn_k_norm.weight"));
            L.q = put_mat(f, map, blk(il, "attn_q.weight"));
            L.k = put_mat(f, map, blk(il, "attn_k.weight"));
            L.v = put_mat(f, map, blk(il, "attn_v.weight"));
            L.o = put_mat(f, map, blk(il, "attn_output.weight"));
        } else {
            L.conv = put_q16(f, blk(il, "ssm_conv1d.weight"));
            L.a = put_q16(f, blk(il, "ssm_a"));
            L.dt = put_q16(f, blk(il, "ssm_dt.bias"));
            L.ssm_norm = put_q16(f, blk(il, "ssm_norm.weight"));
            L.qkv = put_mat(f, map, blk(il, "attn_qkv.weight"));
            L.gate = put_mat(f, map, blk(il, "attn_gate.weight"));
            L.alpha = put_mat(f, map, blk(il, "ssm_alpha.weight"));
            L.beta = put_mat(f, map, blk(il, "ssm_beta.weight"));
            L.out = put_mat(f, map, blk(il, "ssm_out.weight"));
        }
    }
    h.total_bytes = at;
    if (fseek(out, 0, SEEK_SET)) return 1;
    if (fwrite(&h, 1, sizeof h, out) != sizeof h) return 1;
    fclose(out);
    printf("host: %s: %llu B\n", argv[2], static_cast<unsigned long long>(h.total_bytes));
    return 0;
}
