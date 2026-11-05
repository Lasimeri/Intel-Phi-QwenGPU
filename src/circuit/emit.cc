// The GPU as tensors in one model file: the SIMT machine's circuit (src/circuit/sm_circ.cc) becomes layers of the 35B's
// shape, every gate a weight, the memories (ROM, M, A, S, W, the f16 table, the registers, the scratch words) as banks,
// the 2B's image as the state those layers start from. Port of QwenPU's src/circuit/emit64.cc (itself a port of
// chipformer/q35/emit.py and merge.py, MIT), with residual slots shared by lifetime and banks with any number of ports.
//
// The residual: [Qwen's 2048 dims][ballast x m][slots ...], hidden a power of 2 and m B^2 / hidden a power of 4, so every
// RMS norm on a chip token divides by exactly a power of 2 and a residual value x reaches the projections as x c; weights
// carry 1 / c. A slot is shared by dims whose lifetimes do not overlap; a handover subtracts the old value with one unit.
// The units live in the shared expert (gate = K b, up = a / K, down = c; its router gate rides on the ballast: sigmoid(128)
// = 1); the heads are the 35B's DeltaNet heads; a bank is a memory in one piece with its ports' address, data and enable
// forms as rows. The file is sparse: a tensor's zero pages are holes.
//
//   gpu_emit OUT.gguf IMAGE_DIR [--base MODEL_35B.gguf] [--hidden N]
//
// Without --base the file holds the computer's tensors and keys only (the forward and its gate need nothing else); with
// it, the 35B's metadata, tensor table and tensor bytes come first, copied byte for byte.
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "../exec/membits.h"
#include "../gpu/image.h"
#include "eval.h"
#include "sm_circ.h"

namespace {

[[noreturn]] void die(const std::string &what) {
    std::fprintf(stderr, "gpu_emit: %s\n", what.c_str());
    std::exit(2);
}

struct Layout {
    int hidden = 16384, off = 2048, m = 16, log2_ballast = 40;
    std::vector<int> slot;  // dim -> slot
    double ballast() const { return std::ldexp(1.0, log2_ballast); }
    double scale() const {  // c: the normed value of a residual value 1
        const double mean = (double)m * ballast() * ballast() / hidden;
        const int log2m = std::ilogb(mean);
        if (std::ldexp(1.0, log2m) != mean || log2m % 2) die("m B^2 / hidden is not a power of 4");
        return std::ldexp(1.0, -log2m / 2);
    }
    int dim(int d) const {
        if (d <= 0 || d >= (int)slot.size() || slot[(size_t)d] < 0) die("dim " + std::to_string(d) + " has no slot");
        return off + m + slot[(size_t)d];
    }
};

// A tensor's nonzeros: position -> value (they add), or dense values, or raw bytes (a memory, its zero pages left as holes).
struct T {
    std::string name;
    uint64_t cols = 0, rows = 0, third = 1;
    std::map<uint64_t, double> v;
    std::vector<float> dense;
    const uint8_t *raw = nullptr;
    uint64_t raw_bytes = 0;
    T(const std::string &n, uint64_t c, uint64_t r, uint64_t th = 1) : name(n), cols(c), rows(r), third(th) {}
    void add(uint64_t r, uint64_t c, double x) { v[r * cols + c] += x; }
    uint64_t count() const { return cols * rows * third; }
};

void row(const Expr &e, const Layout &lay, T &t, uint64_t r, double w = 1.0) {
    const double c = lay.scale();
    for (const auto &p : e.t) {
        const double coef = p.second.to_double();
        if (p.first == ONE) t.add(r, (uint64_t)lay.off, w * coef / (lay.ballast() * c));
        else t.add(r, (uint64_t)lay.dim(p.first), w * coef / c);
    }
}

void put_str(std::vector<uint8_t> &o, const std::string &s) {
    const uint64_t n = s.size();
    o.insert(o.end(), (const uint8_t *)&n, (const uint8_t *)&n + 8);
    o.insert(o.end(), s.begin(), s.end());
}
template <typename V>
void put(std::vector<uint8_t> &o, V v) {
    o.insert(o.end(), (const uint8_t *)&v, (const uint8_t *)&v + sizeof v);
}
void kv_int(std::vector<uint8_t> &o, const std::string &k, uint32_t v) {
    put_str(o, k);
    put<uint32_t>(o, 4);
    put<uint32_t>(o, v);
}
void kv_text(std::vector<uint8_t> &o, const std::string &k, const std::string &v) {
    put_str(o, k);
    put<uint32_t>(o, 8);
    put_str(o, v);
}

// The 35B's header, walked only to find where its parts end: they are copied byte for byte.
struct Src {
    std::vector<uint8_t> h;
    size_t at = 0;
    uint64_t n_tensors = 0, n_kv = 0, kv_end = 0, info_end = 0, align = 32;
    template <typename V>
    V get() {
        V v;
        std::memcpy(&v, &h[at], sizeof v);
        at += sizeof v;
        return v;
    }
    void skip_value(uint32_t ty) {
        static const size_t sz[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
        if (ty == 8) {
            const uint64_t n = get<uint64_t>();
            at += n;
        } else if (ty == 9) {
            const uint32_t et = get<uint32_t>();
            const uint64_t n = get<uint64_t>();
            if (et == 8) {
                for (uint64_t i = 0; i < n; i++) skip_value(8);
            } else {
                at += n * sz[et];
            }
        } else {
            at += sz[ty];
        }
    }
    void read(int fd) {
        h.resize(256u << 20);
        const ssize_t got = pread(fd, h.data(), h.size(), 0);
        if (got < 1024 || std::memcmp(h.data(), "GGUF", 4)) die("the base file is not a GGUF");
        at = 8;
        n_tensors = get<uint64_t>();
        n_kv = get<uint64_t>();
        for (uint64_t i = 0; i < n_kv; i++) {
            const uint64_t n = get<uint64_t>();
            const std::string key((const char *)&h[at], n);
            at += n;
            const uint32_t ty = get<uint32_t>();
            if (key == "general.alignment") {
                align = get<uint32_t>();
            } else {
                skip_value(ty);
            }
        }
        kv_end = at;
        for (uint64_t i = 0; i < n_tensors; i++) {
            at += get<uint64_t>();
            at += 8 * (uint64_t)get<uint32_t>();
            at += 4 + 8;
        }
        info_end = at;
        if (info_end >= (uint64_t)got) die("the base file's header is larger than expected");
    }
};

constexpr uint64_t PAGE = 4096;

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fputs("usage: gpu_emit OUT.gguf IMAGE_DIR [--base MODEL_35B.gguf] [--hidden N]\n", stderr);
        return 2;
    }
    const std::string path = argv[1], dir = argv[2];
    std::string base;
    Layout lay;
    for (int i = 3; i < argc; i++) {
        if (!std::strcmp(argv[i], "--base") && i + 1 < argc) base = argv[++i];
        else if (!std::strcmp(argv[i], "--hidden") && i + 1 < argc) lay.hidden = std::atoi(argv[++i]);
    }
    uint8_t classes[SM_STAGES] = SM_STAGE_CLASSES_DEFAULT;
    sm_classes_env(classes);
    Target target;
    target.key_heads = 16;
    target.value_heads = 32;
    target.intermediate = 8192;
    SmCircStats st;
    Circ *cp = build_sm_circuit(classes, target, &st);
    Circ &c = *cp;
    const int nk = 16, nv = 32, r_ = nv / nk, W = target.intermediate, ST = HD;
    const uint64_t qkv_n = (uint64_t)(2 * nk + nv) * (uint64_t)ST, val_n = (uint64_t)nv * (uint64_t)ST;

    // the image: the memories' starting bytes, through the evaluator's loader (the same packing the gate ran on)
    Image im;
    image_load(im, dir);
    Evaluator ev(c, false, 1);
    ev.load(im.img);
    const std::vector<float> state = ev.export_state();

    // the slots: shared by lifetime; the code dims are never handed over
    Alloc al = allocate(c, c.keep, W);
    lay.slot = al.slot;
    // m B^2 / hidden must be a power of 4: m = hidden / 1024 keeps the mean at 2^70 for every power-of-2 hidden
    lay.m = std::max(1, lay.hidden / 1024);
    if (lay.off + lay.m + al.n_slots > lay.hidden) die("the chip's slots do not fit: " + std::to_string(al.n_slots) + " slots, hidden " + std::to_string(lay.hidden));
    // the last layers only clear: the next token's code is all that reaches the output; a layer clears at most W slots
    const int keep_n = (int)c.keep.size();
    const int exit_layers = (int)((al.n_slots - keep_n + lay.m + W - 1) / W);
    const int layers = c.n_layers() + std::max(1, exit_layers);
    std::printf("host: %d computer layers, hidden %d (Qwen's %d + %d ballast + %d slots for %zu dims, %d spare), shared expert %d units, c = 2^%d; circuit %llu units, %llu nodes, %llu banks\n", layers,
                lay.hidden, lay.off, lay.m, al.n_slots, c.dims.size(), lay.hidden - lay.off - lay.m - al.n_slots, W, std::ilogb(lay.scale()), (unsigned long long)st.units, (unsigned long long)st.nodes,
                (unsigned long long)st.banks);
    std::fflush(stdout);
    std::vector<char> keep(c.dims.size(), 0);
    for (int d : c.keep) keep[(size_t)d] = 1;

    std::vector<T> ts;
    std::vector<std::pair<const Node *, int>> bank_nodes;
    uint64_t units_total = 0, clear_units = 0, zero_units = 0;
    int heads_total = 0, vheads_total = 0;
    const uint64_t H = (uint64_t)lay.hidden;
    for (int L = 0; L < layers; L++) {
        const std::string p = "chip.blk." + std::to_string(L) + ".";
        T qkv(p + "attn_qkv.weight", H, qkv_n);
        T zw(p + "attn_gate.weight", H, val_n);
        T aw(p + "ssm_alpha.weight", H, (uint64_t)nv);
        T bw(p + "ssm_beta.weight", H, (uint64_t)nv);
        T ow(p + "ssm_out.weight", val_n, H);
        T gate(p + "ffn_gate_shexp.weight", H, (uint64_t)W);
        T up(p + "ffn_up_shexp.weight", H, (uint64_t)W);
        T down(p + "ffn_down_shexp.weight", (uint64_t)W, H);
        T conv(p + "ssm_conv1d.weight", 4, qkv_n);
        conv.dense.assign(qkv_n * 4, 0.0f);
        T stt(p + "ssm_state", (uint64_t)ST, (uint64_t)ST, (uint64_t)nv);
        stt.dense.assign(val_n * (uint64_t)ST, 0.0f);
        int j = 0;
        for (const Node &n : c.nodes) {
            if (n.is_ffn || n.level != 2 * L + 1) continue;
            if (n.is_bank) {
                bank_nodes.push_back({&n, L});
                continue;
            }
            if ((int)n.vh.size() > r_) die(n.name + ": too many value heads for a key head");
            if (j >= nk) die("too many heads in a layer");
            const auto chans = [&](uint64_t base, const std::map<int, Chan> &m) {
                for (const auto &e : m) {
                    row(e.second.expr, lay, qkv, base + (uint64_t)e.first);
                    for (int lag = 0; lag < 4; lag++) conv.dense[(base + (uint64_t)e.first) * 4 + (uint64_t)(3 - lag)] = (float)e.second.taps[(size_t)lag];
                }
            };
            chans((uint64_t)j * (uint64_t)ST, n.q);
            chans((uint64_t)(nk + j) * (uint64_t)ST, n.k);
            for (size_t x = 0; x < n.vh.size(); x++) {
                const VHead &vh = n.vh[x];
                const uint64_t h = (uint64_t)j * (uint64_t)r_ + x;
                chans((uint64_t)(2 * nk) * (uint64_t)ST + h * (uint64_t)ST, vh.v);
                const Expr beta = vh.beta ? Dy(256) * *vh.beta - Dy(128) * c.one : Dy(128) * c.one;
                row(beta, lay, bw, h);
                row(vh.a, lay, aw, h);
                for (const auto &e : vh.z) row(e.second, lay, zw, h * (uint64_t)ST + (uint64_t)e.first);
                for (const Out &o : vh.out) ow.add((uint64_t)lay.dim(o.dim), h * (uint64_t)ST + (uint64_t)o.col, o.w);
                vheads_total++;
            }
            j++;
            heads_total++;
        }
        uint64_t u = 0;
        // a unit that subtracts a slot's old value: -x relu(K) / K
        const auto clear_unit = [&](int old_dim) {
            if (u >= (uint64_t)W) die("too many units in a layer (clearing)");
            row(c.one, lay, gate, u, (double)K_GATE);
            up.add(u, (uint64_t)lay.dim(old_dim), 1.0 / K_GATE / lay.scale());
            down.add((uint64_t)lay.dim(old_dim), u, -1.0);
            u++;
            clear_units++;
        };
        for (const Node &n : c.nodes) {
            if (!n.is_ffn || n.level != 2 * L + 2) continue;
            for (const Term &t : n.terms) {
                if (t.b.empty()) {
                    zero_units++;  // relu of an empty form: the unit is 0 for ever, so it has no row
                    continue;
                }
                if (u >= (uint64_t)W) die("too many units in a layer");
                row(t.b, lay, gate, u, (double)K_GATE);
                row(t.a, lay, up, u, 1.0 / K_GATE);
                down.add((uint64_t)lay.dim(n.out), u, t.c.to_double());
                u++;
            }
            if (!n.into) {
                const auto hs = al.subtract.find(n.out);
                if (hs != al.subtract.end()) clear_unit(hs->second);
            }
        }
        const auto cl = al.clears.find(2 * L + 2);
        if (cl != al.clears.end())
            for (int o : cl->second) clear_unit(o);
        if (L >= c.n_layers()) {
            // the exit: every slot but the code's, and the ballast, leave the residual; slot s is cleared by exit layer s / W
            const int ex = L - c.n_layers();
            for (int s = 0; s < al.n_slots; s++) {
                const int o = al.final_owner[(size_t)s];
                if (keep[(size_t)o] || s / W != ex) continue;
                clear_unit(o);
            }
            for (int i = 0; i < lay.m; i++) {
                if (L != layers - 1) break;
                if (u >= (uint64_t)W) die("too many units in the exit layer");
                row(c.one, lay, gate, u, (double)K_GATE);
                up.add(u, (uint64_t)(lay.off + i), (lay.ballast() / K_GATE) / (lay.ballast() * lay.scale()));
                down.add((uint64_t)(lay.off + i), u, -1.0);
                u++;
            }
        }
        units_total += u;
        ts.push_back(std::move(qkv));
        ts.push_back(std::move(zw));
        ts.push_back(std::move(aw));
        ts.push_back(std::move(bw));
        ts.push_back(std::move(ow));
        ts.push_back(std::move(gate));
        ts.push_back(std::move(up));
        ts.push_back(std::move(down));
        ts.push_back(std::move(conv));
        ts.push_back(std::move(stt));
        const auto gain = [&](const std::string &nm, float v, uint64_t n) {
            T g(p + nm, n, 1);
            g.dense.assign(n, v);
            ts.push_back(std::move(g));
        };
        gain("attn_norm.weight", 1.0f, H);
        gain("post_attention_norm.weight", 1.0f, H);
        gain("ssm_a", -1.0f, (uint64_t)nv);
        gain("ssm_dt.bias", -110.0f, (uint64_t)nv);
        gain("ssm_norm.weight", 1.0f, (uint64_t)ST);
        // the shared expert's gate: sigmoid(128) = 1 on a chip token, from the ballast
        T inp(p + "ffn_gate_inp_shexp.weight", H, 1);
        inp.dense.assign(H, 0.0f);
        inp.dense[(size_t)lay.off] = (float)(128.0 / (lay.ballast() * lay.scale()));
        ts.push_back(std::move(inp));
    }
    // the banks: a memory's ports as rows (every address bit, data bit and enable a row of the layer's input)
    for (size_t K = 0; K < bank_nodes.size(); K++) {
        const Node &n = *bank_nodes[K].first;
        const Bank &b = n.bank;
        const std::string p = "chip.bank." + std::to_string(K) + ".";
        uint64_t nra = 0, nwa = 0, nd = 0, nout = 0;
        std::vector<float> rinfo, winfo, outd;
        for (const BankRead &r : b.reads) {
            rinfo.insert(rinfo.end(), {(float)r.addr.size(), (float)r.width, (float)nout, (float)nra});
            nra += r.addr.size();
            if ((int)r.out.size() != r.width) die(n.name + ": a read port's outputs do not match its width");
            for (int d : r.out) outd.push_back((float)lay.dim(d));
            nout += (uint64_t)r.width;
        }
        for (const BankWrite &w : b.writes) {
            if (w.data_taps.size() != w.data.size()) die(n.name + ": a write port's taps do not match its data");
            winfo.insert(winfo.end(), {(float)w.addr.size(), (float)w.data.size(), (float)w.width, (float)nwa, (float)nd, (float)w.addr_taps[0], (float)w.addr_taps[1], (float)w.addr_taps[2], (float)w.addr_taps[3]});
            nwa += w.addr.size();
            nd += w.data.size();
        }
        T ra(p + "ra", H, std::max<uint64_t>(nra, 1)), wa(p + "wa", H, std::max<uint64_t>(nwa, 1)), wd(p + "wd", H, std::max<uint64_t>(nd, 1)), we(p + "we", H, std::max<uint64_t>(b.writes.size(), 1));
        std::vector<float> taps(std::max<uint64_t>(nd, 1) * 4, 0.0f);
        uint64_t ai = 0, wai = 0, di = 0;
        for (const BankRead &r : b.reads)
            for (const Expr &e : r.addr) row(e, lay, ra, ai++);
        for (size_t wi = 0; wi < b.writes.size(); wi++) {
            const BankWrite &w = b.writes[wi];
            for (const Expr &e : w.addr) row(e, lay, wa, wai++);
            for (size_t i = 0; i < w.data.size(); i++) {
                row(w.data[i], lay, wd, di);
                for (int lag = 0; lag < 4; lag++) taps[di * 4 + (uint64_t)lag] = (float)w.data_taps[i][(size_t)lag];
                di++;
            }
            row(w.we, lay, we, wi);
        }
        ts.push_back(std::move(ra));
        ts.push_back(std::move(wa));
        ts.push_back(std::move(wd));
        ts.push_back(std::move(we));
        T tp(p + "taps", 4, std::max<uint64_t>(nd, 1));
        tp.dense = taps;
        ts.push_back(std::move(tp));
        T ou(p + "out", std::max<uint64_t>(nout, 1), 1);
        ou.dense = outd;
        if (ou.dense.empty()) ou.dense.assign(1, 0.0f);
        ts.push_back(std::move(ou));
        T ri(p + "rinfo", 4, std::max<uint64_t>(b.reads.size(), 1));
        ri.dense = rinfo;
        if (ri.dense.empty()) ri.dense.assign(4, 0.0f);
        ts.push_back(std::move(ri));
        T wi(p + "winfo", 9, std::max<uint64_t>(b.writes.size(), 1));
        wi.dense = winfo;
        if (wi.dense.empty()) wi.dense.assign(9, 0.0f);
        ts.push_back(std::move(wi));
        T info(p + "info", 6, 1);
        info.dense = {(float)b.mem, (float)bank_nodes[K].second, (float)b.reads.size(), (float)b.writes.size(), 0.0f, 0.0f};
        ts.push_back(std::move(info));
    }
    // the memories, one piece each, as the evaluator starts with them (W from the image's file)
    {
        T mi("chip.meminfo", MB_COUNT, 1);
        mi.dense.assign(MB_COUNT, 0.0f);
        for (int k = 0; k < MB_COUNT; k++) {
            const uint8_t *bytes = k == MEM_W ? im.w : ev.memory(k).data();
            const uint64_t nbytes = k == MEM_W ? im.w_bytes : ev.memory(k).size();
            mi.dense[(size_t)k] = (float)nbytes;
            if ((double)mi.dense[(size_t)k] != (double)nbytes) die("a memory's length is not exact in F32");
            if (!nbytes) continue;
            T mem("chip.mem." + std::to_string(k), (nbytes + 3) / 4, 1);
            mem.raw = bytes;
            mem.raw_bytes = nbytes;
            ts.push_back(std::move(mem));
        }
        ts.push_back(std::move(mi));
    }
    // the value heads' starting states, in creation order
    {
        std::map<std::string, T *> by;
        for (T &t : ts) by[t.name] = &t;
        std::map<int, int> next;
        size_t state_at = 0;
        for (const Node &n : c.nodes) {
            if (n.is_ffn || n.is_bank) continue;
            const int L = (n.level - 1) / 2, jj = next[L]++;
            for (size_t x = 0; x < n.vh.size(); x++) {
                T &stt = *by["chip.blk." + std::to_string(L) + ".ssm_state"];
                std::memcpy(&stt.dense[((uint64_t)jj * (uint64_t)r_ + x) * (uint64_t)(ST * ST)], &state[state_at], (size_t)(ST * ST) * 4);
                state_at += (size_t)(ST * ST);
            }
        }
        if (state_at != state.size()) die("the state does not match the value heads");
    }
    // the slot map, for the gate: dim -> residual index (-1 none), slot -> final dim
    {
        T sl("chip.slot", c.dims.size(), 1), fo("chip.final", std::max<uint64_t>((uint64_t)al.n_slots, 1), 1);
        for (size_t d = 0; d < c.dims.size(); d++) sl.dense.push_back(al.slot[d] < 0 ? -1.0f : (float)(lay.off + lay.m + al.slot[d]));
        for (int s = 0; s < al.n_slots; s++) fo.dense.push_back((float)al.final_owner[(size_t)s]);
        if (fo.dense.empty()) fo.dense.assign(1, -1.0f);
        ts.push_back(std::move(sl));
        ts.push_back(std::move(fo));
    }
    T emb("chip.token_embd.weight", (uint64_t)lay.hidden, (uint64_t)SM_CHIP_VOCAB);

    // ------------------------------------------------------------------ the file
    int sfd = -1;
    Src src;
    uint64_t src_data = 0, src_bytes = 0;
    if (!base.empty()) {
        sfd = open(base.c_str(), O_RDONLY);
        if (sfd < 0) die("cannot open the base file");
        struct stat sst;
        if (fstat(sfd, &sst)) die("stat");
        src.read(sfd);
        src_data = ((src.info_end + src.align - 1) / src.align) * src.align;
        src_bytes = (uint64_t)sst.st_size - src_data;
    }
    const uint64_t align = base.empty() ? PAGE : src.align;
    std::vector<T *> all;
    all.push_back(&emb);
    for (T &t : ts) all.push_back(&t);
    std::vector<uint8_t> kvs;
    uint32_t nkeys = 0;
    const auto ki = [&](const std::string &k, uint32_t v) {
        kv_int(kvs, k, v);
        nkeys++;
    };
    if (base.empty()) {
        kv_text(kvs, "general.architecture", "qwen35moe");
        nkeys++;
        kv_text(kvs, "general.name", "Intel-Phi-QwenGPU chip only");
        nkeys++;
        ki("general.alignment", (uint32_t)align);
    }
    ki("qwen35moe.chip.block_count", (uint32_t)layers);
    ki("qwen35moe.chip.exit_layers", (uint32_t)(layers - c.n_layers()));
    ki("qwen35moe.chip.embedding_length", (uint32_t)lay.hidden);
    ki("qwen35moe.chip.ssm.group_count", (uint32_t)nk);
    ki("qwen35moe.chip.ssm.time_step_rank", (uint32_t)nv);
    ki("qwen35moe.chip.ssm.state_size", (uint32_t)ST);
    ki("qwen35moe.chip.expert_shared_feed_forward_length", (uint32_t)W);
    ki("qwen35moe.chip.first_dim", (uint32_t)lay.off);
    ki("qwen35moe.chip.ballast_dims", (uint32_t)lay.m);
    ki("qwen35moe.chip.ir_dims", (uint32_t)al.n_slots);
    ki("qwen35moe.chip.banks", (uint32_t)bank_nodes.size());
    ki("qwen35moe.chip.mems", (uint32_t)MB_COUNT);
    ki("qwen35moe.chip.vocab", SM_CHIP_VOCAB);
    ki("qwen35moe.chip.go_token", SM_GO);
    ki("qwen35moe.chip.halt_token", SM_HALT);
    ki("qwen35moe.chip.emit_token", SM_EMIT);
    ki("qwen35moe.chip.lstep_token", SM_LSTEP);
    ki("qwen35moe.chip.text_token", SM_TEXT);
    ki("qwen35moe.chip.next0_token", SM_NEXT0);
    ki("qwen35moe.chip.exec0_token", SM_EXEC0);
    ki("qwen35moe.chip.warp0_token", SM_WARP0);
    ki("qwen35moe.chip.vlo0_token", SM_VLO0);
    ki("qwen35moe.chip.vhi0_token", SM_VHI0);
    ki("qwen35moe.chip.vend0_token", SM_VEND0);
    ki("qwen35moe.chip.job_words", SM_M_JOB);
    ki("qwen35moe.chip.stage_classes", (uint32_t)classes[0] | (uint32_t)classes[1] << 8 | (uint32_t)classes[2] << 16 | (uint32_t)classes[3] << 24);
    ki("qwen35moe.chip.stage_classes_hi", (uint32_t)classes[4] | (uint32_t)classes[5] << 8);
    // the chip's tensor table: its size depends on the names and shapes, not on the offsets
    const auto infos = [&](uint64_t rel0, std::vector<uint64_t> *rels) {
        std::vector<uint8_t> o;
        uint64_t rel = rel0;
        for (T *t : all) {
            put_str(o, t->name);
            const uint32_t nd = t->third > 1 ? 3 : (t->rows > 1 ? 2 : 1);
            put<uint32_t>(o, nd);
            put<uint64_t>(o, t->cols);
            if (nd > 1) put<uint64_t>(o, t->rows);
            if (nd > 2) put<uint64_t>(o, t->third);
            put<uint32_t>(o, 0);  // F32
            put<uint64_t>(o, rel);
            if (rels) rels->push_back(rel);
            rel += (t->count() * 4 + PAGE - 1) & ~(PAGE - 1);
        }
        return std::make_pair(o, rel);
    };
    const uint64_t hdr_bytes = base.empty() ? 24 + kvs.size() + infos(0, nullptr).first.size() : 24 + (src.kv_end - 24) + kvs.size() + (src.info_end - src.kv_end) + infos(0, nullptr).first.size();
    const uint64_t data_at = ((hdr_bytes + align - 1) / align) * align;
    const uint64_t rel0 = ((data_at + src_bytes + PAGE - 1) & ~(PAGE - 1)) - data_at;
    std::vector<uint64_t> rels;
    const auto ci = infos(rel0, &rels);
    std::vector<uint8_t> hdr;
    hdr.insert(hdr.end(), {'G', 'G', 'U', 'F'});
    put<uint32_t>(hdr, 3);
    put<uint64_t>(hdr, src.n_tensors + all.size());
    put<uint64_t>(hdr, src.n_kv + nkeys);
    if (!base.empty()) hdr.insert(hdr.end(), src.h.begin() + 24, src.h.begin() + (long)src.kv_end);
    hdr.insert(hdr.end(), kvs.begin(), kvs.end());
    if (!base.empty()) hdr.insert(hdr.end(), src.h.begin() + (long)src.kv_end, src.h.begin() + (long)src.info_end);
    hdr.insert(hdr.end(), ci.first.begin(), ci.first.end());
    if (hdr.size() != hdr_bytes) die("the header's size");
    const int fd = open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (fd < 0) die("cannot create " + path);
    if (ftruncate(fd, (off_t)(data_at + ci.second))) die("ftruncate");
    if (pwrite(fd, hdr.data(), hdr.size(), 0) != (ssize_t)hdr.size()) die("header");
    if (!base.empty()) {
        std::vector<uint8_t> buf(64u << 20);
        for (uint64_t at = 0; at < src_bytes;) {
            const size_t n = (size_t)std::min<uint64_t>(buf.size(), src_bytes - at);
            if (pread(sfd, buf.data(), n, (off_t)(src_data + at)) != (ssize_t)n || pwrite(fd, buf.data(), n, (off_t)(data_at + at)) != (ssize_t)n) die("copying the base file");
            at += n;
        }
        std::printf("host: the base's %.2f GB copied unchanged\n", (double)src_bytes / 1e9);
        std::fflush(stdout);
    }
    uint64_t nonzero = 0, not_exact = 0, pages = 0;
    std::vector<float> page(PAGE / 4);
    for (size_t i = 0; i < all.size(); i++) {
        T *t = all[i];
        const uint64_t off = data_at + rels[i];
        if (t == &emb) continue;
        if (t->raw || !t->dense.empty()) {
            const uint8_t *src_b = t->raw ? t->raw : reinterpret_cast<const uint8_t *>(t->dense.data());
            const uint64_t bytes = t->raw ? t->raw_bytes : t->count() * 4;
            if (!t->raw && t->dense.size() != t->count()) die(t->name + ": dense size");
            for (uint64_t k = 0; k < bytes; k += PAGE) {
                const size_t n = (size_t)std::min<uint64_t>(PAGE, bytes - k);
                bool any = false;
                for (size_t q = 0; q < n && !any; q++) any = src_b[k + q] != 0;
                if (!any) continue;
                if (pwrite(fd, src_b + k, n, (off_t)(off + k)) != (ssize_t)n) die("write");
                pages++;
            }
            continue;
        }
        uint64_t cur = ~0ull;
        const auto flush = [&]() {
            if (cur == ~0ull) return;
            if (pwrite(fd, page.data(), PAGE, (off_t)(off + cur * PAGE)) != (ssize_t)PAGE) die("write");
            pages++;
        };
        for (const auto &e : t->v) {
            if (e.second == 0.0) continue;
            const uint64_t pg = e.first / (PAGE / 4);
            if (pg != cur) {
                flush();
                cur = pg;
                std::fill(page.begin(), page.end(), 0.0f);
            }
            const float fv = (float)e.second;
            if ((double)fv != e.second) not_exact++;
            page[e.first % (PAGE / 4)] = fv;
            nonzero++;
        }
        flush();
    }
    if (not_exact) die("weights not exact in F32: " + std::to_string(not_exact));
    // the embedding: B in the ballast, the token's features in their slots; only the pages touched are written
    {
        const uint64_t off = data_at + rels[0];
        std::vector<float> rowv((size_t)lay.hidden, 0.0f);
        for (int i = 0; i < lay.m; i++) rowv[(size_t)(lay.off + i)] = (float)lay.ballast();
        std::vector<int> fdim;
        std::set<uint64_t> pgs;
        pgs.insert((uint64_t)lay.off * 4 / PAGE);
        for (int d : c.feat_dim) {
            fdim.push_back(lay.dim(d));
            pgs.insert((uint64_t)fdim.back() * 4 / PAGE);
        }
        for (uint32_t tk = 0; tk < SM_CHIP_VOCAB; tk++) {
            for (size_t i = 0; i < fdim.size(); i++) rowv[(size_t)fdim[i]] = (float)c.feat_fn[i](tk);
            for (uint64_t pg : pgs) {
                const float *src_r = &rowv[pg * (PAGE / 4)];
                bool any = false;
                for (size_t q = 0; q < PAGE / 4 && !any; q++) any = src_r[q] != 0.0f;
                if (any && pwrite(fd, src_r, PAGE, (off_t)(off + (uint64_t)tk * (uint64_t)lay.hidden * 4 + pg * PAGE)) != (ssize_t)PAGE) die("write");
                pages += any;
            }
        }
    }
    close(fd);
    std::printf("host: wrote %s: %d key heads, %d value heads, %zu banks, %llu units (%llu of them handovers and the exit's clears; %llu constant-zero units left out), %llu nonzero weights, %llu pages written, logical %.2f GB\n", path.c_str(),
                heads_total, vheads_total, bank_nodes.size(), (unsigned long long)units_total, (unsigned long long)clear_units, (unsigned long long)zero_units, (unsigned long long)nonzero, (unsigned long long)pages, (double)(data_at + ci.second) / 1e9);
    return 0;
}
