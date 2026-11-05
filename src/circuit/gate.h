// The gate language of the SIMT circuit: the token features of the machine's stream, and the exact gadgets every
// datapath is built from (nibble adders, products as nibble columns, bit selects, counters, tables, banks).
//
// Port of the generic parts of QwenPU's src/circuit/{core.h, core.cc, core64.cc, vxparts.h, vxcirc.cc, vx64.cc} (MIT),
// which were proven exact against the RV64 machine, with nothing of RISC-V left in them. A circuit is a Graph of
// FFN units (c * a * relu(b)), DeltaNet heads (one-hot keyed rows of 126 data columns) and banks (memories written
// compactly, src/circuit/graph.h); every weight is a dyadic rational, every activation an exact bf16 value.
#ifndef GPU_CIRCUIT_GATE_H
#define GPU_CIRCUIT_GATE_H

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "graph.h"

extern "C" {
#include "../gpu/sm.h"
}

// The token families of docs/machine.md. The exit code has NFAM family bits and NPAY payload bits.
enum { F_VLO, F_VHI, F_VEND, F_EXEC, F_NEXT, F_WARP, F_EMIT, F_HALT, F_GO, F_LSTEP, F_TEXT, F_COUNT };
constexpr int NFAM = 4, NPAY = 16;
constexpr int CODE_AMP = 128;
constexpr int PAY_BITS[F_COUNT] = {16, 16, 16, SM_BBITS + 1, SM_QBITS, SM_WBITS, 0, 0, 0, 0, 0};

struct Fam {
    int id;  // F_*, or -1 for a token outside the chip's vocabulary
    uint32_t p;
};
Fam famof(uint32_t tok);
uint32_t tok_of(int fam, uint32_t p);  // the inverse: a token from its family and payload

struct Counter {
    Expr x, reset;
};

// One next-token rule of the exit: when the literals hold, the family and payload.
struct SelCase {
    L lits;
    bool is_int;
    long long ival;
    L bits;
};

class Circ : public Graph {
public:
    explicit Circ(Target t);

    Expr fam[F_COUNT];           // is<family>, one per token
    std::vector<L> pay;          // pay[f]: the payload bits of a token of family f (0 on other tokens)
    L code;                      // the exit's code dims: CODE_AMP * (2 bit - 1) of the token's own code
    std::map<std::string, int> tables;  // table name -> node index
    L out_fam, out_pay;          // the next token's family and payload bits, written by the exit
    std::vector<int> keep;       // dims that survive the clear layer

    std::vector<Terms> fn_bits(const Expr &z, long long lo, long long hi, const std::function<long long(long long)> &f, int n, const L &lits = {});
    L zeros(int n, const std::string &name);
    L table(const std::string &name, const std::map<int, Chan> &queries, const std::map<int, Chan> &keys, const std::vector<VSpec> &vheads,
            char levels = 'b', double tiny = 0.0, int min_level = 0);
    L int_bits(const Expr &z, long long hi, const std::string &name, int n);
    Expr counter(const std::string &name, const Expr &reset, const Expr &step);
    L thermo(const Expr &x, const Expr &reset, int n, const std::string &name);
    Expr split(const Expr &z, long long lo, long long hi, const std::string &name);
    void tag_table(const std::string &name) { tables[name] = (int)nodes.size() - 1; }
    // a register of n bits held across passes: written with `bits` when `we` is 1, else it keeps its value (through the
    // previous pass's output); reset to 0 when `clear` is 1 (clear wins)
    L reg(const std::string &name, const L &bits, const Expr &we, const Expr &clear, int n);
    // the previous pass's value of a linear form (through a one-column head with a lag tap)
    L lag1(const L &bits, const std::string &name);
};

// Pieces every datapath uses.
L lits_of(const L &bits, long long v);  // the literals that say bits == v
Expr andpre(const L &lits);             // a linear threshold that is 1 when every literal is 1, <= 0 otherwise (as a gate input)
L nots(const L &bits);
std::vector<int> col_offsets(const std::vector<std::pair<int, int>> &bounds, int top = 255);
Expr nib(const L &bits, int k, bool inv = false);  // nibble k of a bit list as a 0..15 form (inv: 15 - it)
L select_bits(Circ &c, const std::vector<SelCase> &cases, int n, const std::string &name);
L mux_bits(Circ &c, const Expr &sel, const L &when1, const L &when0, const std::string &name);  // sel ? when1 : when0, per bit
L and_bits(Circ &c, const L &bits, const Expr &lit, const std::string &name);

struct Adder {
    L r;
    Expr cout;
};
struct Stage1 {
    L g, p;
    std::vector<L> d;
};
using SumForms = std::vector<std::vector<std::pair<Expr, L>>>;  // per nibble: (a sum of nibble forms, its literals)
Stage1 stage1_digits(Circ &c, const SumForms &sums, const std::string &name, long long hi);
Stage1 stage1_cols(Circ &c, const std::vector<std::pair<Expr, long long>> &cols, const L &lits, const std::string &name);
Adder adder_stage2(Circ &c, const L &g, const L &p, const std::vector<L> &d, int n_nib, const std::string &name);
L add_nibbles(Circ &c, const L &a_bits, const L &b_bits, int n_nib, const L &lits, const std::string &name);
// a + b with a carry-in form (0/1) and the carry out, n_nib nibbles; b may be inverted (a - b - 1 + cin)
Adder add_full(Circ &c, const L &a_bits, const L &b_bits, const Expr &cin, bool inv_b, int n_nib, const std::string &name);

// Nibble columns: terms with bounds, reduced to bits by rounds of carries until every column fits a bf16 input.
struct CT {
    Expr e;
    long long lo, hi;
};
using Cols = std::vector<std::vector<CT>>;
using IntF = std::function<int(int)>;
L bits_of(const L &th, int n_bits, int top);
Expr count_fn(Circ &c, const Counter &x, const IntF &f, int top, const std::string &name, const Expr *a_in = nullptr, const L &lits = {});
Expr count_bit(Circ &c, const Counter &x, int b, int top, const std::string &name, const Expr *base = nullptr, int shift = 0);
L sum_columns(Circ &c, const Cols &cols_in, int n_nib, const std::string &name);
// x * y into nibble columns: nx nibbles of x as bits (its top one signed if sx), y as nibble forms with their ranges
void product_cols(Circ &c, Cols &cols, const L &x, int nx, bool sx, const L &y, const std::vector<std::pair<long long, long long>> &yr, const std::string &name);
// (x * y + 2^15) >> 16, bits 0..31, for x of nx nibbles (top signed) and 8 unsigned nibbles of y
L mul_round(Circ &c, const L &x, int nx, const L &y, const std::string &name);
std::pair<L, std::vector<std::pair<long long, long long>>> signed_nibs(const L &bits, int n);
// the window of a 48-bit signed x shifted right by u - 33 (u = 0..63): bits 0..n-1; u below 33 is a left shift
L barrel(Circ &c, const L &x48, const L &u, int n, const std::string &name);
// bit planes of a dot product q . w over 32 lanes: qs[i] = 8 bits of lane i's q (signed), ws[i] = a nibble form of lane i's w
void plane_cols(Circ &c, Cols &cols, long long &cst, const std::vector<L> &qs, const L &ws, long long wmin, long long wmax, int nib, const std::string &name);
// a < b, a == b over n-bit unsigned values given as bits (nibble compares)
std::pair<Expr, Expr> cmp_bits(Circ &c, const L &a, const L &b, int n_nib, const std::string &name);

#endif
