// ptxdump FILE.ptx...: parses each file and prints its kernels, parameters, shared arrays, register widths, labels and
// instruction counts by opcode; with -v every instruction as parsed. Host, a check of the parser.
#include <cstdio>
#include <cstring>
#include <map>

#include "../gpu/ptx.h"

static void show(const ptx::Operand &o) {
    switch (o.kind) {
    case ptx::Operand::REG: std::printf("reg %s", o.name.c_str()); break;
    case ptx::Operand::SREG: std::printf("sreg %s", o.name.c_str()); break;
    case ptx::Operand::IMM: std::printf("imm %lld", (long long)o.value); break;
    case ptx::Operand::SYM: std::printf("sym %s", o.name.c_str()); break;
    case ptx::Operand::LABEL: std::printf("label %s", o.name.c_str()); break;
    case ptx::Operand::MEM:
        std::printf("mem [");
        show(*o.base);
        std::printf(" %+lld]", (long long)o.value);
        break;
    case ptx::Operand::PAIR:
        std::printf("pair ");
        show(*o.base);
        std::printf("|");
        show(*o.second);
        break;
    }
}

int main(int argc, char **argv) {
    bool verbose = false;
    int files = 0, kernels = 0, instrs = 0;
    std::map<std::string, int> by_op;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "-v")) {
            verbose = true;
            continue;
        }
        auto ks = ptx::load(argv[i]);
        files++;
        for (const auto &p : ks) {
            const ptx::Kernel &k = p.second;
            kernels++;
            std::printf("%s: kernel %s: %zu params, %zu shared, %zu instructions, %zu labels\n", argv[i], k.name.c_str(), k.params.size(), k.shared.size(),
                        k.instrs.size(), k.labels.size());
            for (const ptx::Param &pm : k.params) std::printf("  param %s %s %d\n", pm.name.c_str(), pm.type.c_str(), pm.size);
            for (const ptx::Shared &s : k.shared) std::printf("  shared %s align %d size %d\n", s.name.c_str(), s.align, s.size);
            for (const auto &rw : k.reg_width) std::printf("  reg %%%s %d\n", rw.first.c_str(), rw.second);
            for (const ptx::Instr &ins : k.instrs) {
                instrs++;
                by_op[ins.base()]++;
                if (!verbose) continue;
                std::printf("  %4d %s%s%s %s", ins.line, ins.guard.empty() ? "" : (ins.neg ? "@!" : "@"), ins.guard.c_str(), ins.guard.empty() ? "" : " ", ins.op.c_str());
                for (size_t a = 0; a < ins.args.size(); a++) {
                    std::printf(a ? ", " : " ");
                    show(ins.args[a]);
                }
                std::printf("\n");
            }
        }
    }
    std::printf("host: %d files, %d kernels, %d instructions\n", files, kernels, instrs);
    for (const auto &p : by_op) std::printf("  %-8s %d\n", p.first.c_str(), p.second);
    return 0;
}
