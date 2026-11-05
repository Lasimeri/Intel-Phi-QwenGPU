// The forward pass of layers read from a model file, and nothing else: no circuit, no machine, no knowledge of what the layers compute.
// The arithmetic is the 35B's (src/text/elem.h, as src/text/q35_ref.cc uses it): RMS norms, DeltaNet heads, the shared expert, and the
// banks (memories held in one piece, addressed by rows of the layer's input: src/exec/membits.h says how an address selects bytes). It
// loads a tensor's nonzeros from the file and leaves its zeros out, which changes no value: a zero weight adds exactly zero. Shortcuts
// here hold for any tensors, never for a particular program.
#ifndef QWENPU_EXEC_TFORWARD_H
#define QWENPU_EXEC_TFORWARD_H

#include <cstdint>
#include <string>
#include <set>
#include <unordered_map>
#include <vector>

#include "../text/gguf.h"
#include "membits.h"

// What a tool may watch in a forward, without changing it.
struct TProbe {
    virtual ~TProbe() {}
    virtual void on_hn(int, int, const float *, const float *) {}   // layer, 0 attention or 1 FFN, the normed residual, the residual
    virtual void on_delta(int, int, const float *) {}
    virtual void on_units(int, const float *, int) {}
    virtual void on_mix(int, const float *, int) {}
};

class TForward {
    friend struct ChipStats;
    friend struct ChipImage;
public:
    explicit TForward(const std::string &path);
    ~TForward();

    // One token through every layer. The residual afterwards: hidden() values.
    const std::vector<float> &step(uint32_t token);
    // The argmax of the output, tied to the embedding, on the residual step() left.
    uint32_t next_token();
    // The residual before layer k (k = layers(): after the last), kept when asked; -1 keeps none.
    void keep_before(int k) { keep_at_ = k; }
    const std::vector<float> &before() const { return before_; }
    void probe(TProbe *p) { probe_ = p; }
    uint64_t full_scans() const { return full_scans_; }
    int hidden() const { return h_; }
    int layers() const { return (int)ls_.size(); }
    uint64_t nnz() const { return nnz_; }
    uint64_t file_data_bytes() const { return data_bytes_; }
    int64_t meta(const std::string &key) const { return (int64_t)f_.get_int(key); }
    const gguf::File &file() const { return f_; }
    // A writable memory's bytes (chip.mem.K): the host writes a job into the words memory through here.
    std::vector<uint8_t> &memory(int kind);

private:
    // rows with nonzeros only
    struct Sparse {
        std::vector<uint32_t> row, start, col;
        std::vector<float> val;
        uint32_t rows = 0;
    };
    struct VState {
        std::vector<float> S;  // ST x ST
        int hd = 0;
    };
    // a bank: ports over one of the memories; a read port's address rows select an element whose bits add onto its
    // output dims, a write port's address and data rows (each bit through its own taps over the last passes) land a
    // value when its enable row is nonzero; writes land before the reads, ports in their order
    struct BRead {
        int nbits, width, aoff, ooff;
    };
    struct BWrite {
        int nabits, ndbits, width, aoff, doff;
        float ataps[4];
    };
    struct Bank {
        int kind = 0;
        std::vector<BRead> reads;
        std::vector<BWrite> writes;
        Sparse ra, wa, wd, we;
        std::vector<float> taps;                 // 4 per data bit
        std::vector<float> vra, vwa, vd, vwe;    // this pass's row values
        std::vector<float> hwa[3], hd[3];        // the last three passes' address and data row values, newest first
        std::vector<int> out;
    };
    struct Layer {
        Sparse qkv, gate, alpha, beta, out, f_gate, f_up, f_down;
        std::vector<float> attn_norm, post_norm, conv, a, dt, ssm_norm, gate_inp;
        std::vector<VState> st;       // value heads that can ever hold or read anything
        std::vector<float> win[3];    // the last projections, oldest first
        std::vector<uint32_t> convrows;
        std::vector<Bank> banks;
    };

    Sparse load_sparse(const std::string &name);
    std::vector<float> load_dense(const std::string &name, uint64_t n);
    void gemv(const Sparse &m, const float *x, float *y) const;
    void one_layer(Layer &L);
    void run_bank(Bank &b);

    gguf::File f_;
    std::vector<Layer> ls_;
    int h_ = 0, nk_ = 0, nv_ = 0, st_ = 0, w_ = 0;
    uint64_t nnz_ = 0, data_bytes_ = 0;
    std::vector<float> x_, hn_, qkv_, z_, conv_, mix_, y_, g_, u_, xs_, q_, k_, alpha_, beta_;
    std::vector<float> row_;
    int keep_at_ = -1;
    std::vector<float> before_;
    std::vector<char> row_used_;
    std::set<int> strays_seen_;
    std::vector<int> out_dims_, out_dims_all_;
    std::vector<float> out_max_;
    std::unordered_map<uint64_t, uint32_t> out_table_;
    uint64_t full_scans_ = 0;
    TProbe *probe_ = nullptr;
    // the memories, one piece each: writable ones in mem_, read-only ones mapped from the file
    std::vector<uint8_t> mem_[MB_COUNT];
    const uint8_t *mem_ro_[MB_COUNT] = {};
    uint64_t mem_n_[MB_COUNT] = {};
};

#endif
