// Parser for the PTX subset clang's NVPTX backend emits for the 2B's kernels. Port of qgpu/ptx.py (qwenGPU, MIT).
// A module holds .entry kernels; for each one its parameters, register widths, .shared variables and a flat list of
// instructions with labels resolved to indices. Inline asm the kernels use for the machine's own instructions
// (q8dot, sdot, sup, scfg, smax, wsum, wmax, udiv, usqrt, mulsh, stv.*) comes through as ordinary opcodes.
#ifndef GPU_PTX_H
#define GPU_PTX_H

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ptx {

struct Param {
    std::string name, type;  // ".u64", ".u32", ".b32" ...
    int size = 0;            // bytes
};

struct Shared {
    std::string name;
    int align = 0, size = 0;
};

struct Operand;
using OperandP = std::shared_ptr<Operand>;

struct Operand {
    enum Kind { REG, SREG, IMM, SYM, LABEL, MEM, PAIR } kind = REG;
    std::string name;      // reg / sreg / sym / label
    int64_t value = 0;     // imm, or mem's byte offset
    OperandP base, second; // mem: the base operand; pair: the two operands (base, second)
};

struct Instr {
    std::string op;        // full opcode with modifiers, e.g. "setp.ge.s32"
    std::vector<Operand> args;
    std::string guard;     // predicate register guarding it, or empty
    bool neg = false;      // @!%p
    int line = 0;
    std::vector<std::string> parts;  // op split on '.'
    const std::string &base() const { return parts[0]; }
};

struct Kernel {
    std::string name;
    std::vector<Param> params;
    std::map<std::string, int> reg_width;  // prefix -> bits ("r" 32, "rd" 64, "p" 1)
    std::vector<Shared> shared;
    std::vector<Instr> instrs;
    std::map<std::string, int> labels;
    std::string stub_call;  // while parsing: the function a stub kernel calls (clang's OpenCL __clang_ocl_kern_imp_NAME)
};

// Dies with a message on anything it does not understand.
std::map<std::string, Kernel> parse(const std::string &text);
std::map<std::string, Kernel> load(const std::string &path);
Operand parse_operand(const std::string &s);

}  // namespace ptx

#endif
