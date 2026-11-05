// Machine instructions over virtual registers, before register allocation: what the driver emits and the block former
// reads. Port of KInstr (qgpu/circuit/kernel.py, MIT), with the machine's extensions (docs/machine.md).
#ifndef GPU_KINSTR_H
#define GPU_KINSTR_H

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sm.h"

namespace lower {

struct Unsupported : std::runtime_error {
    explicit Unsupported(const std::string &s) : std::runtime_error(s) {}
};

constexpr int VZERO = 1 << 20;   // the virtual name of the zero register (lane or uniform)
constexpr int VARGS = VZERO + 1;  // the args base, u14

struct KInstr {
    int op = SM_NOP;
    int rd = VZERO, ra = VZERO, rb = VZERO, rc = VZERO;
    uint32_t imm = 0;
    int pd = 0, cmp = SM_CMP_EQ;
    bool sgn = false, useimm = false;
    int guard = -1;  // predicate enabling each lane, or -1
    bool gneg = false;
    int bank = SM_BANK_M;
    int sh = 0;
    bool ud = false;  // the result (or the store) is uniform: set by the lowerer before block formation
    // (register written or -1 (predicates are -1 - p), registers read)
    std::pair<int, std::vector<int>> rw() const {
        std::vector<int> g;
        if (guard >= 0) g.push_back(-1 - guard);
        std::vector<int> two = useimm ? std::vector<int>{ra} : std::vector<int>{ra, rb};
        auto cat = [](std::vector<int> a, const std::vector<int> &b) {
            a.insert(a.end(), b.begin(), b.end());
            return a;
        };
        switch (op) {
        case SM_ADD: case SM_SUB: case SM_MAX: case SM_MIN: case SM_MULHI: case SM_AND: case SM_OR: case SM_XOR:
            return {rd, two};
        case SM_SHL: case SM_SHR: return {rd, sh == 63 ? std::vector<int>{ra, rc} : std::vector<int>{ra}};
        case SM_CLZ: return {rd, {ra}};
        case SM_SELP: return {rd, cat(two, g)};
        case SM_MAD: return {rd, useimm ? std::vector<int>{ra, rc} : std::vector<int>{ra, rb, rc}};
        case SM_MULSH: return {rd, sh == 63 ? cat(two, {rc}) : two};
        case SM_TID: case SM_SREG: return {rd, {}};
        case SM_LD: return {rd, {ra}};
        case SM_STV: return {-1, cat(two, g)};
        case SM_SETP: return {-1 - pd, two};
        case SM_Q8DOT: case SM_SDOT: return {rd, cat(two, {rc})};
        case SM_SUP: return {rd, cat(cat(two, {rc, rd}), g)};  // rd = max(rd, |the state element after the update|)
        case SM_SCFG: return {-1, {ra}};
        case SM_WSUM: case SM_WMAX: return {rd, cat({ra}, g)};
        case SM_UDIV: return {rd, useimm ? std::vector<int>{ra, rc} : std::vector<int>{ra, rb, rc}};
        case SM_USQRT: return {rd, {ra, rc}};
        case SM_NOP: return {-1, {}};
        default: throw Unsupported("rw of op " + std::to_string(op));
        }
    }
    bool writes_reg() const { return rw().first >= 0; }
};

inline int class_of(int op) {
    switch (op) {
    case SM_MAD: case SM_MULHI: case SM_MULSH: case SM_SUP: return SM_CL_MUL;
    case SM_Q8DOT: return SM_CL_DOT;
    case SM_SDOT: return SM_CL_STATE;
    case SM_AND: case SM_OR: case SM_XOR: case SM_SHL: case SM_SHR: case SM_CLZ: return SM_CL_LOGIC;
    case SM_WSUM: case SM_WMAX: return SM_CL_RED;
    case SM_UDIV: case SM_USQRT: return SM_CL_SCALAR;
    default: return SM_CL_ANY;
    }
}

inline const char *op_name(int op) {
    static const char *names[SM_NOPS] = {"nop", "add", "sub", "mad", "mulhi", "mulsh", "and", "or", "xor", "shl", "shr", "max", "min",
                                         "setp", "selp", "tid", "sreg", "ld", "stv", "q8dot", "sdot", "sup", "scfg", "wsum", "wmax",
                                         "udiv", "usqrt", "clz"};
    return op >= 0 && op < SM_NOPS ? names[op] : "?";
}

}  // namespace lower

#endif
