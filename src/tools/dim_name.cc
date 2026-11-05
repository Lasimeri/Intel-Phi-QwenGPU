// A residual index of the model file back to the circuit's name for that dim: a debugging aid, host only, never part of a run.
//
//   dim_name INDEX ...
#include <cstdio>
#include <cstdlib>

#include "../circuit/core.h"

int main(int argc, char **argv) {
    Target t;
    t.key_heads = 16;
    t.value_heads = 32;
    t.intermediate = 8192;
    RvCore *c = build_core64(t);
    for (int i = 1; i < argc; i++) {
        const int d = std::atoi(argv[i]) - 2048 - 8;
        if (d >= 0 && d < (int)c->dims.size()) std::printf("%s: %s, kind %c, level %d\n", argv[i], c->dims[(size_t)d].name.c_str(), c->dims[(size_t)d].kind, c->dims[(size_t)d].level);
        else std::printf("%s: outside the chip's dims\n", argv[i]);
    }
    return 0;
}
