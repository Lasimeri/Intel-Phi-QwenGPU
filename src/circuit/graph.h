// The circuit graph for the Qwen3.8-35B target: SwiGLU units, Gated DeltaNet heads and banks (memories written
// compactly), plus the exact gadgets built from them.
//
// Port of chipformer/q35/ir.py (Graph, FFN, Head, Chan, placement) and chipformer/q35/prims.py (Circ) from QwenPU
// (MIT), with banks added for the SIMT machine's memories. Node and dim creation order is semantic: it feeds the
// scheduler's tie-break and every matrix layout, so nodes and dims live in vectors in creation order. Maps below
// (std::map, sorted) replace Python dicts only where upstream's order has no effect on a value.
#ifndef GPU_CIRCUIT_GRAPH_H
#define GPU_CIRCUIT_GRAPH_H

#include <array>
#include <functional>
#include <map>
#include <optional>
#include <string>

#include "expr.h"

constexpr int K_GATE = 128;           // silu gate scale: silu(K n) = K relu(n) for integers n
constexpr long long A_REF = 1 << 24;  // ballast value of a state row (two columns)
constexpr int A_BIT = 1 << 7;         // value of a set bit in a state row
constexpr int HD = 128;               // key/value head dim
constexpr int REF_COL0 = 0, REF_COL1 = 1;
constexpr double OUT_W = 128.0;       // head output weight of a one-hot read

using Taps = std::array<double, 4>;   // x_t, x_{t-1}, x_{t-2}, x_{t-3}
constexpr Taps LAG0{1, 0, 0, 0}, LAG1{0, 1, 0, 0}, LAG2{0, 0, 1, 0}, LAG3{0, 0, 0, 1};

struct Chan {
    Expr expr;
    Taps taps{1.0, 0.0, 0.0, 0.0};
    Chan() = default;
    Chan(const Expr &e) : expr(e) {}  // NOLINT: a bare form is a channel with the default taps
    Chan(const Expr &e, Taps tp) : expr(e), taps(tp) {}
};

struct Dim {
    std::string name;
    char kind;  // 'o' one, 't' token, 'f' ffn, 'h' head
    int level;
};

struct Term {
    Dy c;
    Expr a, b;  // c * a * relu(b)
};
using Terms = std::vector<Term>;

struct Out {
    int col, dim;
    double w;
};

struct VHead {
    std::map<int, Chan> v;
    Expr a;
    std::map<int, Expr> z;
    std::vector<Out> out;
    std::optional<Expr> beta;
    bool into = false;
};

// --- banks: the machine's memories, written compactly -------------------------------------------------------------
// A bank is a memory the file holds in one piece. A read port's address bits select an element whose bits land in its
// output dims (a read costs no unit). A write port fires when its enable is 1: its address and data channels may be
// lagged (the value of a previous pass), so a value computed in one pass can land in the next; writes land before the
// node's reads in the same pass. Element kinds by memory: see MemKind.
enum MemKind {
    MEM_M,     // int32 words, addressed by word
    MEM_A,     // quantized blocks of 36 bytes (32 int8 then an int32 scale): addresses as block << 6 | j (j < 32 a byte, j = 32 the scale)
    MEM_S,     // int16 elements
    MEM_C,     // int8 bytes
    MEM_W,     // read-only Q8_0 blocks of 34 bytes (an f16 scale then 32 int8): block << 6 | j
    MEM_ROM,   // read-only blocks of 544 bits: six 80-bit rows then the 64-bit LIVE word
    MEM_REG,   // the lane registers: words, address slot << 9 | lane << 4 | r
    MEM_UREG,  // the uniform registers: words, address slot << 4 | u
    MEM_F16,   // read-only: an f16's value in Q.24 as an int32 word, addressed by the f16's 16 bits
    MEM_SCR,   // the circuit's scratch words (its own state between passes): words
    MEM_COUNT
};
// what a read port returns (width in bits): a word (32), a half (16), a byte (8), or a whole block (A: 288 = 32 bytes
// then the scale; W: 272 = the f16 then 32 bytes; ROM: 544)
enum PortWidth { PW_BYTE = 8, PW_HALF = 16, PW_WORD = 32, PW_ABLK = 288, PW_WBLK = 272, PW_ROMBLK = 544 };

struct BankRead {
    L addr;                 // address bits, low first (0/1 forms)
    std::vector<int> out;   // the element's bits land here (width dims), low first
    int width = PW_WORD;
};
struct BankWrite {
    L addr;                 // address bits
    Taps addr_taps = LAG0;  // the pass the address comes from
    L data;                 // data bits, low first
    std::vector<Taps> data_taps;  // per data bit
    Expr we;                // write enable, a 0/1 form of the current pass
    int width = PW_WORD;    // 32, 16 or 8 bits written
};
struct Bank {
    int mem = MEM_M;
    std::vector<BankRead> reads;
    std::vector<BankWrite> writes;
};

struct Node {
    bool is_ffn = true;
    bool is_bank = false;
    std::string name;
    int level = 0, min_level = 0;
    // FFN
    Terms terms;
    int out = -1;
    bool into = false;
    // Head: the first entry of vh is the first value head
    std::map<int, Chan> q, k;
    std::vector<VHead> vh;
    double tiny = 0;
    int qhot = 2;
    char levels = 'b';  // 'b' bit rows, 'y' byte rows, 'd' decay rows
    // Bank
    Bank bank;
};

struct Target {
    int key_heads = 16, value_heads = 32, intermediate = 9216;
};

// What head() takes for a value head.
struct VSpec {
    std::map<int, Chan> v;
    Expr a;
    std::map<int, Expr> z;
    std::vector<std::pair<int, double>> outs;
    std::optional<Expr> beta;
    std::optional<L> into;
};

// What thead() takes for a value head: 0/1 tables.
struct TSpec {
    std::map<int, Chan> data;
    Expr ref;
    std::optional<Expr> gate, beta;
    Expr a;
    std::optional<L> into;
};

class Graph {
public:
    Target target;
    std::vector<Dim> dims;
    std::vector<std::function<double(uint32_t)>> feat_fn;  // parallel to feat_dim
    std::vector<int> feat_dim;
    std::vector<Node> nodes;
    Expr one;
    std::map<int, int> units, heads;  // sublayer -> FFN units, key heads used
    int floor_ = 0;                   // no FFN goes below this level

    explicit Graph(Target t);

    Expr token_feature(const std::string &name, std::function<double(uint32_t)> fn);
    int new_dim(const std::string &name, char kind, int level);
    int level_of(const Expr &e) const;

    Expr ffn(const Terms &terms, const std::string &name, const Expr *into = nullptr, int min_level = 0);
    L head(const std::string &name, std::map<int, Chan> q, std::map<int, Chan> k, const VSpec &first,
           const std::vector<VSpec> &rest, int min_level = 0, double tiny = 0.0);
    // a bank node: its read ports' output dims must exist (token features of value 0); placed on the attention side
    void bank(const std::string &name, const Bank &spec, int min_level = 0);

    int n_layers() const;

    // --- prims.Circ --------------------------------------------------------------------------
    Terms t_const(const Dy &c) const;
    Terms t_and(const L &lits, const Dy &coef = Dy(1), const Expr *a = nullptr) const;
    Terms t_ge(const Expr &z, long long theta, long long lo, long long hi, const Dy &coef = Dy(1),
               const L &lits = {}) const;
    Terms t_lt(const Expr &z, long long theta, long long lo, long long hi, const Dy &coef = Dy(1),
               const L &lits = {}) const;
    Terms t_in(const Expr &z, long long a, long long b, long long lo, long long hi,
               const Dy &coef = Dy(1), const L &lits = {}) const;
    Terms t_eq(const Expr &z, long long v, long long lo, long long hi, const Dy &coef = Dy(1),
               const L &lits = {}) const;
    Expr node(const Terms &terms, const std::string &name);
    Expr and_(const L &lits, const std::string &name);

private:
    int place(int lvl, bool odd, int need, std::map<int, int> &used, int cap);
};

// prims.py's free functions.
Expr from_bits(const L &bits);
Chan sel(const Graph &c, const Chan &e);
Chan chan(const Chan &e, const Dy &scale);
std::vector<L> thead(Graph &c, const std::string &name, const std::map<int, Chan> &keys,
                     const std::map<int, Chan> &queries, const std::vector<TSpec> &vheads,
                     double tiny = 0.0);

// sched.py
void reschedule(Graph &g, int unit_cap, int head_cap, const char *last = "exit.");
void sched_check(const Graph &g);
std::vector<int> node_reads(const Node &n);
std::vector<int> node_outputs(const Node &n);

// emit.py allocate(): residual slots shared by dims whose lifetimes do not overlap.
// A handover subtracts the old value with one unit (-x * relu(ONE), exact for a bf16 x): at the new dim's FFN level, or at
// the FFN level before a head's. The emitter writes those units; the evaluator never sees them (its dims are not shared).
struct Alloc {
    std::vector<int> slot;                   // dim -> slot (-1: never placed)
    std::map<int, int> subtract;             // new dim -> old dim: the defining FFN level subtracts the old value
    std::map<int, std::vector<int>> clears;  // FFN level -> old dims cleared there for head dims born at level + 1
    std::vector<int> final_owner;            // slot -> the dim holding it at the end
    int n_slots = 0;
};
Alloc allocate(const Graph &g, const std::vector<int> &keep, int unit_cap);
int allocate_slots(const Graph &g, const std::vector<int> &keep, int unit_cap);  // n_slots only

#endif
