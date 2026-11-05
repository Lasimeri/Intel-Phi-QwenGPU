// The driver: lower a PTX kernel to the machine's blocks at build time. Port of qgpu/lower.py (qwenGPU, MIT), widened
// for the machine of docs/machine.md: uniform registers, runtime arguments from M through u14, the banks and the warp
// instructions as inline-asm opcodes, no host at run time. Raises Unsupported for PTX it does not lower; the 2B's
// kernels are written within what it does.
#ifndef GPU_LOWER_H
#define GPU_LOWER_H

#include <cstdint>
#include <string>
#include <vector>

#include "blocks.h"
#include "ptx.h"

namespace lower {

// a kernel parameter at this launch: a constant (a scalar, or a buffer's base word for a pointer), or a runtime word
// M[u14 + k]
struct ParamSpec {
    enum Kind { CONST, PTR, ARG } kind = CONST;
    int64_t value = 0;  // the constant, the base word, or k
};

struct Options {
    uint8_t classes[SM_STAGES] = SM_STAGE_CLASSES_DEFAULT;
    uint32_t shared_base = 0;  // the M word where a kernel's .shared arrays go (thread blocks run one after another)
};

struct MBlock {
    std::vector<KInstr> instrs;             // physical registers after lower(); nops where padded
    std::vector<std::pair<int, int>> live;  // (stage, register or -1 for an stv), emission order
    int branch = SM_BR_NEXT, bstage = 0, target = 0, lwpb = 0;
    bool bneg = false, bany = false;
};

struct Launch {
    std::string name;
    std::vector<MBlock> blocks;
    std::vector<char> uni;  // per physical-register-space flag is per instruction: see KInstrU below
    uint32_t warps = 0;
    int lwpb = 0;
    int lane_regs = 0, uniform_regs = 0;  // how many of each the kernel uses
    // per block, per stage: the operand kinds (set by lower())
    struct Flags {
        uint8_t ua = 0, ub = 0, uc = 0, ud = 0;
    };
    std::vector<std::vector<Flags>> flags;
    uint64_t stats_instrs = 0;
};

Launch lower(const ptx::Kernel &k, int grid_x, int grid_y, int block_x, int block_y, const std::vector<ParamSpec> &params,
             const Options &opt);

// The launch's blocks as ROM rows at `base` (branch targets made absolute).
void encode(const Launch &l, std::vector<SmBlock> &rom, uint32_t base);

// A readable listing.
std::string listing(const Launch &l);

}  // namespace lower

#endif
