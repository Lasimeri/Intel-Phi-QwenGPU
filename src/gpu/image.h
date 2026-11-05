// The GPU image on disk and in memory: what q2b_build writes (out/gpu/) and what every runner loads. The ROM, M and
// the weights come from files; A and S start empty (A's V^T blocks carry the scale 1.0); the f16 table is computed.
#ifndef GPU_IMAGE_H
#define GPU_IMAGE_H

#include <cstdint>
#include <string>
#include <vector>

#include "sm_ref.h"

struct Image {
    std::vector<SmBlock> rom;
    std::vector<int32_t> m;
    std::vector<uint8_t> a;
    std::vector<int16_t> s;
    const uint8_t *w = nullptr;  // mmapped w.q8
    uint64_t w_bytes = 0;
    std::vector<int64_t> f16_q24;
    uint32_t descriptors = 0, a_blocks = 0, s_elems = 0;
    // the SmImage over these buffers
    SmImage img;
    void bind();
};

// Reads DIR/meta.txt, rom.bin, m.bin, maps w.q8, allocates A and S. Dies with a message on any problem.
void image_load(Image &im, const std::string &dir);

// A bare machine: an empty ROM, M of m_words zero words, a_blocks empty A blocks, s_elems zero state elements, no W,
// the f16 table. The OpenCL runtime's device when no 2B image is given.
void image_bare(Image &im, uint32_t m_words, uint32_t a_blocks, uint32_t s_elems);

// meta.txt's keys
struct ImageMeta {
    uint32_t nblk = 0, descriptors = 0, m_words = 0, a_blocks = 0, s_elems = 0;
    uint64_t w_blocks = 0;
    uint32_t desc_step_end = 0, desc_cont = 0, desc_halt = 0;
};
ImageMeta image_meta(const std::string &dir);

#endif
