// Builds src/exec/chip_image.h's image from a forward loaded from the file. Host only; bytes and layout, no value changes.
#ifndef QWENPU_EXEC_CHIP_IMAGE_HPP
#define QWENPU_EXEC_CHIP_IMAGE_HPP

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

#include "chip_image.h"
#include "tforward.h"

struct ChipImage {
    struct Piece {
        uint64_t to = 0, bytes = 0;
        int kind = 0;
        const uint8_t *from = nullptr;
    };

    explicit ChipImage(TForward &forward);
    const uint8_t *data() const { return blob_; }
    uint64_t size() const { return size_; }
    // The memories, each to go at its place behind the blob.
    const std::vector<Piece> &pieces() const { return pieces_; }
    const ChipHeader &header() const { return *reinterpret_cast<const ChipHeader *>(blob_); }

private:
    TForward &tf;
    uint8_t *blob_ = nullptr;
    uint64_t size_ = 0;
    std::vector<uint32_t> col, woff;
    std::vector<float> wtab;
    std::unordered_map<std::string, uint32_t> line_of;
    std::vector<Piece> pieces_;
    uint64_t image_bytes_ = 0;

    uint64_t alloc(uint64_t bytes, uint64_t align = 64);
    template <class T>
    T *at(uint64_t off) { return reinterpret_cast<T *>(blob_ + off); }
    uint32_t weight_line(const float *w);
    uint32_t slices(const std::vector<uint32_t> &cols, const std::vector<std::array<float, 16>> &w);
    void fan_table(const std::vector<std::vector<uint32_t>> &readers, uint64_t *o_start, uint64_t *o_item);
    static void add_reader(std::vector<std::vector<uint32_t>> &readers, const std::vector<uint32_t> &cols, uint32_t id);
    void y_lines(ChipSub &sub, const TForward::Sparse &m, uint32_t input_dims);
    void memo_tables(ChipSub &sub, const std::vector<float> &norm, const std::vector<float> *before);
    void reads_mask(ChipSub &sub, const std::vector<std::vector<uint32_t>> &readers);
    void one_layer(const TForward::Layer &L, ChipLayer &out, const std::vector<float> *norm_before);
    uint32_t bank_index(uint32_t il, uint32_t bi) const;
};

#endif
