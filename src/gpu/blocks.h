// Load-time block formation: a straight-line list of machine instructions becomes blocks of up to S stages, each
// with its live-outs in emission order. Port of qgpu/circuit/blocks.py and kernel.placer (qwenGPU, MIT).
#ifndef GPU_BLOCKS_H
#define GPU_BLOCKS_H

#include <set>
#include <vector>

#include "kinstr.h"

namespace lower {

struct Block {
    std::vector<KInstr> instrs;             // stage slots, nops where padded
    std::vector<std::pair<int, int>> live;  // emission order: (stage, register or -1 for an stv)
};

// Stage slots for a candidate block, or false when it does not fit: an instruction goes to the next stage whose
// units hold its class (nops pad); no ld after an stv in one block (the store lands at the commit).
bool place(const std::vector<KInstr> &instrs, const uint8_t classes[SM_STAGES], std::vector<KInstr> &slots);

// live[i]: registers live after instruction i.
std::vector<std::set<int>> liveness(const std::vector<KInstr> &program, const std::set<int> &outputs);

// Emission order of a block's live-outs, or false if the block is not allowed.
bool plan(const std::vector<KInstr> &slots, const std::set<int> &live_after, std::vector<std::pair<int, int>> &order);

// Greedy blocks of up to S instructions. Throws Unsupported when an instruction fits no block.
std::vector<Block> split(const std::vector<KInstr> &program, const std::set<int> &outputs, const uint8_t classes[SM_STAGES]);

}  // namespace lower

#endif
