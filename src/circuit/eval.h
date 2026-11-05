// The circuit's exact semantics, pass by pass: the reference the layers in the model file are compared with, and the
// gate of the circuit against the machine's token stream.
//
// Port of ir.evaluate / eval_ffn / eval_head (chipformer/q35/ir.py, QwenPU, MIT), with banks (the SIMT machine's
// memories, written compactly) and the machine's image as the starting state. With check on, every intermediate the
// hardware materializes must be an exact bf16 value, silu inputs must lie in its exact range, keys must be one-hot,
// queries multi-hot, data must be absorbed by the ballast, and every bank address, data and enable form must be 0 or 1:
// this is what makes the fp32 executors exact, and it is the builder's strongest check.
//
// Host only: the reference the executors are compared with, never the port's result.
#ifndef GPU_CIRCUIT_EVAL_H
#define GPU_CIRCUIT_EVAL_H

#include <array>
#include <cmath>
#include <vector>

#include "gate.h"

extern "C" {
#include "../gpu/sm_ref.h"
}

struct LinF {
    uint32_t off = 0, n = 0;
};

class Evaluator {
public:
    Evaluator(const Circ &c, bool check, int threads = 0);  // threads: 0 = the host's hardware threads less two, at most 14
    ~Evaluator();

    // the memories from the machine's image (M, A, S, C, W, ROM, the f16 table); registers start at 0
    void load(const SmImage &img);

    // one pass over a token; residual() holds the stream after every node
    void step(uint32_t token);
    const std::vector<double> &residual() const { return m_; }

    // the next token from the exit's code; -1 if the code names none
    int64_t predict() const;

    uint64_t nnz_total() const { return fi_.size(); }
    uint64_t units() const { return units_; }
    uint64_t passes() const { return pos_; }

    // the value heads' states, in creation order, 128 x 128 F32 each (the emitter's state file); dies on an inexact value
    std::vector<float> export_state() const;

    // a memory's bytes (the emitter writes them into the file; the host's GO writes M through here)
    std::vector<uint8_t> &memory(int mem) { return mem_[(size_t)mem]; }
    // a store trace: every write of a word of M at this address is printed with the pass and the node
    void watch(int64_t addr) { watch_ = addr; }
    const uint8_t *w_bytes() const { return w_; }
    uint64_t w_blocks() const { return w_blocks_; }

private:
    struct EChan {
        int col;
        LinF f;
        Taps taps;
        bool lagged;
        double h[3];
    };
    struct EVHead {
        std::vector<EChan> v;
        LinF a, beta;
        bool has_beta;
        std::vector<std::pair<int, LinF>> z;
        std::vector<Out> out;
        bool into;
        std::vector<double> S;   // 128 x 128 rows
        std::vector<char> live;  // rows that may be nonzero
    };
    struct EUnit {
        double c;
        LinF a, b;
    };
    struct ELag {
        LinF f;
        Taps taps;
        double h[3];
    };
    struct ERead {
        std::vector<LinF> addr;
        std::vector<int> out;
        int width;
    };
    struct EWrite {
        std::vector<ELag> addr, data;
        LinF we;
        int width;
    };
    struct ENode {
        int kind;  // 0 ffn, 1 head, 2 bank
        const char *name;
        std::vector<EUnit> units;
        int out;
        bool into;
        std::vector<EChan> q, k;
        std::vector<EVHead> vh;
        double tiny;
        int mem;
        std::vector<ERead> reads;
        std::vector<EWrite> writes;
    };

    LinF flat(const Expr &e);
    double ev(LinF f) const;
    double bit(LinF f, const char *node, const char *what) const;  // 0 or 1, else exact_fail (with check)
    double lagged(ELag &l) const;
    double conv(EChan &c, const char *node);
    void eval_ffn(ENode &n);
    void eval_head(ENode &n);
    void eval_bank(ENode &n);
    uint64_t read_bits(int mem, int width, uint32_t addr, uint8_t *bits) const;  // fills `width` bits, returns the count
    void write_bits(int mem, int width, uint32_t addr, uint32_t value);

    // the unit nodes of one level run on the worker threads (they write distinct dims and read lower levels only);
    // heads, banks and the `into` nodes run in creation order on the main thread
    struct Pool;
    void run_ffn_level(const std::vector<int> &nodes);
    Pool *pool_ = nullptr;

    const Circ &c_;
    bool check_;
    std::vector<ENode> nodes_;
    std::vector<int> order_;  // node indices by level, then creation
    std::vector<std::vector<int>> levels_;      // per level in order: the unit nodes that may run in parallel
    std::vector<std::vector<int>> levels_ser_;  // per level in order: heads, banks and `into` nodes, in creation order
    std::vector<uint32_t> fi_;
    std::vector<double> fc_;
    std::vector<double> m_;
    uint64_t pos_ = 0, units_ = 0;
    LinF out_fam_[NFAM], out_pay_[NPAY];
    // memories
    std::vector<uint8_t> mem_[MEM_COUNT];
    const uint8_t *w_ = nullptr;
    uint64_t w_blocks_ = 0;
    const int64_t *f16_ = nullptr;  // the full table, for the range check at a read
    int64_t watch_ = -1;
};

#endif
