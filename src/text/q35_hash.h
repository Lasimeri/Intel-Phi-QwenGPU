/* The hashes that let the host check a card run afterwards. Plain C, integer only, on both sides. */
#ifndef TEXT_Q35_HASH_H
#define TEXT_Q35_HASH_H

#include <stdint.h>

#define Q35_FNV_INIT 0xCBF29CE484222325ULL

static inline uint64_t q35_fnv32(uint64_t h, uint32_t v) {
    for (int i = 0; i < 4; i++) {
        h ^= (v >> (8 * i)) & 255u;
        h *= 0x100000001B3ULL;
    }
    return h;
}

/* One 16-row group of logits: order between groups never matters, because their hashes are added. */
static inline uint64_t q35_group_hash(uint32_t group, const uint32_t bits[16]) {
    uint64_t h = q35_fnv32(Q35_FNV_INIT, group);
    for (int l = 0; l < 16; l++) h = q35_fnv32(h, bits[l]);
    return h;
}

/* Sorts like the float it holds, so the card never compares floats in x87. */
static inline uint32_t q35_order_key(uint32_t bits) { return (bits >> 31) ? ~bits : (bits | 0x80000000u); }

/* The rotating lane hash of a vector, 16 lanes: h = rol(h, 5) ^ bits, then FNV of the lanes. */
static inline uint64_t q35_lanes_final(const uint32_t h[16]) {
    uint64_t f = Q35_FNV_INIT;
    for (int l = 0; l < 16; l++) f = q35_fnv32(f, h[l]);
    return f;
}

static inline uint64_t q35_vector_hash(const uint32_t *bits, int n) {
    uint32_t h[16];
    for (int l = 0; l < 16; l++) h[l] = 0x811C9DC5u;
    for (int i = 0; i < n; i++) {
        uint32_t v = h[i % 16];
        h[i % 16] = ((v << 5) | (v >> 27)) ^ bits[i];
    }
    return q35_lanes_final(h);
}

#endif
