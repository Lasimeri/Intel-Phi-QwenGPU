// See blocks.h.
#include "blocks.h"

#include <algorithm>
#include <map>

namespace lower {

bool place(const std::vector<KInstr> &instrs, const uint8_t classes[SM_STAGES], std::vector<KInstr> &slots) {
    slots.clear();
    bool stored = false;
    // The record runs lane by lane: a reduction and a scalar (udiv, usqrt) are final only after the last lane ("late"),
    // so within the block only a uniform instruction may read them (it is computed in the final pass), and a scalar's
    // own operands must be final. A lane's part carries each reduction's running partial (2 tokens): at most two
    // reductions a block, so that the previous lane's partials are within the circuit's reach (4 tokens back).
    std::map<int, bool> late_reg;  // virtual register -> written in this block by a late stage
    bool supped = false;
    int reductions = 0;
    for (const KInstr &ins : instrs) {
        if (ins.op == SM_LD && stored) return false;  // its record must not read memory it is changing
        if (class_of(ins.op) == SM_CL_RED && ++reductions > 2) return false;
        // a sup's write lands after its lane's pass: a read of the state later in the block would see the old value
        if ((ins.op == SM_SDOT || (ins.op == SM_LD && ins.bank == SM_BANK_S)) && supped) return false;
        const auto rw = ins.rw();
        bool reads_late = false;
        for (int x : rw.second) {
            auto it = late_reg.find(x);
            if (it != late_reg.end() && it->second) reads_late = true;
        }
        const int cl = class_of(ins.op);
        if (reads_late && (!ins.ud || cl == SM_CL_SCALAR)) return false;
        const bool late = cl == SM_CL_RED || cl == SM_CL_SCALAR || reads_late;
        while ((int)slots.size() < SM_STAGES && !(classes[slots.size()] & cl)) {
            KInstr nop;
            slots.push_back(nop);
        }
        if ((int)slots.size() >= SM_STAGES) return false;
        slots.push_back(ins);
        if (rw.first >= 0 && rw.first != VZERO) late_reg[rw.first] = late;
        if (ins.op == SM_STV) stored = true;
        if (ins.op == SM_SUP) supped = true;
    }
    return true;
}

std::vector<std::set<int>> liveness(const std::vector<KInstr> &program, const std::set<int> &outputs) {
    std::set<int> live = outputs;
    std::vector<std::set<int>> out(program.size());
    for (int i = (int)program.size() - 1; i >= 0; i--) {
        out[(size_t)i] = live;
        auto [rd, srcs] = program[(size_t)i].rw();
        live.erase(rd);
        for (int s : srcs) live.insert(s);
    }
    return out;
}

bool plan(const std::vector<KInstr> &slots, const std::set<int> &live_after, std::vector<std::pair<int, int>> &order) {
    // deps[s]: the block inputs stage s depends on
    std::vector<std::set<int>> deps;
    std::map<int, int> last;  // register -> stage that last wrote it
    std::vector<int> stv_stages;
    for (size_t s = 0; s < slots.size(); s++) {
        auto [rd, srcs] = slots[s].rw();
        std::set<int> d;
        for (int x : srcs) {
            auto it = last.find(x);
            if (it != last.end()) d.insert(deps[(size_t)it->second].begin(), deps[(size_t)it->second].end());
            else d.insert(x);
        }
        deps.push_back(d);
        if (slots[s].op == SM_STV) stv_stages.push_back((int)s);
        if (rd != -1 && rd != VZERO) last[rd] = (int)s;
        if (slots[s].op == SM_NOP) continue;
    }
    std::vector<std::pair<int, int>> outs;  // (stage, register)
    for (const auto &p : last)
        if (live_after.count(p.first)) outs.emplace_back(p.second, p.first);
    for (int s : stv_stages) outs.emplace_back(s, -1);
    if ((int)outs.size() > SM_LMAX) return false;
    for (const auto &o : outs)
        if (o.second < -1) return false;  // a predicate would cross into another block
    // an stv's record is recomputed at every token while live-outs are committed: its address, value and guard
    // must not read a register a live-out overwrites
    std::set<int> written;
    for (const auto &o : outs)
        if (o.second >= 0) written.insert(o.second);
    for (int s : stv_stages)
        for (int d : deps[(size_t)s])
            if (written.count(d)) return false;
    std::sort(outs.begin(), outs.end());
    order.clear();
    std::vector<std::pair<int, int>> todo = outs;
    while (!todo.empty()) {
        // x may go first unless another pending live-out reads the register x overwrites
        int pick = -1;
        for (size_t i = 0; i < todo.size() && pick < 0; i++) {
            const auto &x = todo[i];
            bool ok = true;
            for (const auto &y : todo)
                if (y != x && x.second >= 0 && deps[(size_t)y.first].count(x.second)) ok = false;
            if (ok) pick = (int)i;
        }
        if (pick < 0) return false;
        order.push_back(todo[(size_t)pick]);
        todo.erase(todo.begin() + pick);
    }
    return true;
}

std::vector<Block> split(const std::vector<KInstr> &program, const std::set<int> &outputs, const uint8_t classes[SM_STAGES]) {
    std::vector<std::set<int>> live = liveness(program, outputs);
    std::vector<Block> blocks;
    size_t i = 0;
    while (i < program.size()) {
        bool have = false;
        size_t best_j = 0;
        std::vector<KInstr> best_slots;
        std::vector<std::pair<int, int>> best_order;
        for (size_t j = i + 1; j <= program.size(); j++) {
            std::vector<KInstr> slots;
            std::vector<KInstr> cand(program.begin() + (long)i, program.begin() + (long)j);
            if (!place(cand, classes, slots)) break;
            std::vector<std::pair<int, int>> order;
            if (plan(slots, live[j - 1], order)) {
                have = true;
                best_j = j;
                best_slots = slots;
                best_order = order;
            }
        }
        if (!have) throw Unsupported("instruction " + std::to_string(i) + " (" + op_name(program[i].op) + ") fits no block");
        Block b;
        b.instrs = best_slots;
        b.live = best_order;
        blocks.push_back(b);
        i = best_j;
    }
    return blocks;
}

}  // namespace lower
