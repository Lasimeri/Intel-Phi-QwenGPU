// Prints a GGUF's metadata, without the tokenizer's tables, and its tensors whose names match a substring.
//
//   gguf_meta FILE [TENSOR_SUBSTRING]
#include <cstdio>
#include <string>

#include "../text/gguf.h"

int main(int argc, char **argv) {
    if (argc < 2) return 2;
    gguf::File f;
    gguf::open(f, argv[1]);
    for (const auto &p : f.ints) printf("int %s = %llu\n", p.first.c_str(), (unsigned long long)p.second);
    for (const auto &p : f.floats) printf("float %s = %.9g\n", p.first.c_str(), p.second);
    for (const auto &p : f.strs)
        if (p.first.rfind("tokenizer.", 0) != 0) printf("str %s = %.100s\n", p.first.c_str(), p.second.c_str());
    for (const auto &p : f.int_arrays)
        if (p.first.rfind("tokenizer.", 0) != 0) printf("array %s: %zu entries\n", p.first.c_str(), p.second.size());
    if (argc > 2)
        for (const gguf::Tensor &t : f.tensors)
            if (t.name.find(argv[2]) != std::string::npos)
                printf("tensor %s type %u dims %llu %llu %llu, %llu B at %llu\n", t.name.c_str(), t.type, (unsigned long long)t.dims[0], (unsigned long long)t.dims[1], (unsigned long long)t.dims[2], (unsigned long long)t.nbytes, (unsigned long long)t.offset);
    return 0;
}
