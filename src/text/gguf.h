/* A read-only GGUF v3 reader for the host. The metadata goes into memory; tensor bytes stay on disk,
 * addressed by absolute file offset, because the model file is larger than this machine's free RAM. */
#ifndef TEXT_GGUF_H
#define TEXT_GGUF_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace gguf {

enum TensorType : uint32_t { T_F32 = 0, T_Q8_0 = 8, T_Q6_K = 14 };

struct Tensor {
    std::string name;
    uint32_t n_dims = 0;
    uint64_t dims[4] = {1, 1, 1, 1};  /* dims[0] is the fastest axis, as in ggml */
    uint32_t type = 0;
    uint64_t offset = 0;              /* absolute, in the file */
    uint64_t nbytes = 0;
};

struct File {
    int fd = -1;
    std::string path;
    uint64_t size = 0;
    std::map<std::string, uint64_t> ints;
    std::map<std::string, double> floats;
    std::map<std::string, std::string> strs;
    std::map<std::string, std::vector<std::string>> str_arrays;
    std::map<std::string, std::vector<int64_t>> int_arrays;
    std::vector<Tensor> tensors;
    std::map<std::string, size_t> by_name;

    const Tensor &tensor(const std::string &name) const;
    bool has(const std::string &name) const { return by_name.count(name) != 0; }
    uint64_t get_int(const std::string &key) const;
};

/* Exits with a message on any malformed input: a bad model file has no useful recovery here. */
void open(File &f, const char *path);
void close(File &f);

/* Bytes of `n` elements of a type laid out in a row; n must be a multiple of the block size. */
uint64_t row_bytes(uint32_t type, uint64_t n);

/* pread until done, or exit. */
void read_at(const File &f, uint64_t off, void *dst, uint64_t n);

}  // namespace gguf

#endif
