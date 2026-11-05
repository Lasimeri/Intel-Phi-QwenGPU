#include "chip_image.hpp"

#include <sys/mman.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <unordered_map>

#include "../text/elem.h"

namespace {

[[noreturn]] void dies(const std::string &what) {
    std::fprintf(stderr, "chip_image: %s\n", what.c_str());
    std::exit(2);
}

constexpr uint64_t BLOB_MOST = 2ULL << 30;  // a reservation: pages are touched only as the image grows

// A sparse matrix by lines of 16 rows: each line's columns ascending, with their 16 weights.
struct LineForm {
    std::vector<uint32_t> cols;
    std::vector<std::array<float, 16>> w;
};

std::map<uint32_t, LineForm> by_lines(const std::vector<uint32_t> &row, const std::vector<uint32_t> &start, const std::vector<uint32_t> &col, const std::vector<float> &val) {
    std::map<uint32_t, std::map<uint32_t, std::array<float, 16>>> gather;
    for (size_t k = 0; k < row.size(); k++) {
        const uint32_t line = row[k] / 16, lane = row[k] % 16;
        for (uint32_t i = start[k]; i < start[k + 1]; i++) {
            auto it = gather[line].try_emplace(col[i]);
            if (it.second) it.first->second.fill(0.0f);
            it.first->second[lane] = val[i];
        }
    }
    std::map<uint32_t, LineForm> out;
    for (auto &l : gather) {
        LineForm &f = out[l.first];
        for (auto &c : l.second) {
            f.cols.push_back(c.first);
            f.w.push_back(c.second);
        }
    }
    return out;
}

}  // namespace

uint64_t ChipImage::alloc(uint64_t bytes, uint64_t align) {
    const uint64_t at = (size_ + align - 1) / align * align;
    if (at + bytes > BLOB_MOST) dies("the image is past its reservation");
    size_ = at + bytes;
    return at;
}

uint32_t ChipImage::weight_line(const float *w) {
    const std::string key(reinterpret_cast<const char *>(w), 64);
    const auto it = line_of.find(key);
    if (it != line_of.end()) return it->second;
    const uint32_t off = (uint32_t)(wtab.size() * 4);
    wtab.insert(wtab.end(), w, w + 16);
    line_of.emplace(key, off);
    return off;
}

// The line's slices into the shared arrays. Returns the first; the length is the form's.
uint32_t ChipImage::slices(const std::vector<uint32_t> &cols, const std::vector<std::array<float, 16>> &w) {
    const uint32_t s0 = (uint32_t)col.size();
    for (size_t i = 0; i < cols.size(); i++) {
        col.push_back(cols[i]);
        woff.push_back(weight_line(w[i].data()));
    }
    return s0;
}

// who reads which input dim, as start[] and item[]
void ChipImage::fan_table(const std::vector<std::vector<uint32_t>> &readers, uint64_t *o_start, uint64_t *o_item) {
    uint64_t items = 0;
    for (const std::vector<uint32_t> &r : readers) items += r.size();
    *o_start = alloc((readers.size() + 1) * 4);
    *o_item = alloc(std::max<uint64_t>(items, 1) * 4);
    uint32_t *st = at<uint32_t>(*o_start), *it = at<uint32_t>(*o_item), n = 0;
    for (size_t d = 0; d < readers.size(); d++) {
        st[d] = n;
        for (uint32_t x : readers[d]) it[n++] = x;
    }
    st[readers.size()] = n;
}

void ChipImage::add_reader(std::vector<std::vector<uint32_t>> &readers, const std::vector<uint32_t> &cols, uint32_t id) {
    for (uint32_t c : cols) {
        std::vector<uint32_t> &r = readers[c];
        if (r.empty() || r.back() != id) r.push_back(id);
    }
}

void ChipImage::y_lines(ChipSub &sub, const TForward::Sparse &m, uint32_t input_dims) {
    sub.n_y = (uint32_t)m.row.size();
    sub.o_yv = alloc((uint64_t)CHIP_HIDDEN * 4);
    sub.o_yrow_dim = alloc(((uint64_t)sub.n_y + 1) * 4);
    sub.o_yrow_start = alloc(((uint64_t)sub.n_y + 1) * 4);
    sub.o_yrow_col = alloc(std::max<uint64_t>(m.col.size(), 1) * 4);
    sub.o_yrow_w = alloc(std::max<uint64_t>(m.val.size(), 1) * 4);
    std::vector<std::vector<uint32_t>> readers(input_dims);
    for (uint32_t r = 0; r < sub.n_y; r++) {
        at<uint32_t>(sub.o_yrow_dim)[r] = m.row[r];
        at<uint32_t>(sub.o_yrow_start)[r] = m.start[r];
        for (uint32_t i = m.start[r]; i < m.start[r + 1]; i++) readers[m.col[i]].push_back(r);
    }
    at<uint32_t>(sub.o_yrow_start)[sub.n_y] = sub.n_y ? m.start[sub.n_y] : 0;
    if (!m.col.empty()) {
        std::memcpy(at<uint32_t>(sub.o_yrow_col), m.col.data(), m.col.size() * 4);
        std::memcpy(at<float>(sub.o_yrow_w), m.val.data(), m.val.size() * 4);
    }
    fan_table(readers, &sub.o_yfan_start, &sub.o_yfan_item);
    sub.o_ydirty = alloc(((uint64_t)sub.n_y + 7) / 8 * 8 + 8);
}

void ChipImage::reads_mask(ChipSub &sub, const std::vector<std::vector<uint32_t>> &readers) {
    sub.o_reads = alloc((uint64_t)CHIP_LINES * 2);
    for (size_t d = 0; d < readers.size(); d++)
        if (!readers[d].empty()) at<uint16_t>(sub.o_reads)[d / 16] |= (uint16_t)(1u << (d % 16));
}

void ChipImage::memo_tables(ChipSub &sub, const std::vector<float> &norm, const std::vector<float> *before) {
    sub.o_norm = alloc((uint64_t)CHIP_HIDDEN * 4);
    std::memcpy(at<float>(sub.o_norm), norm.data(), (size_t)CHIP_HIDDEN * 4);
    sub.o_dirty = alloc(((uint64_t)sub.n_work + 7) / 8 * 8 + 8);
    sub.o_list = alloc(((uint64_t)sub.n_work + 1) * 4);
    if (before)
        for (int i = 0; i < CHIP_HIDDEN; i++)
            if (q_bits((*before)[(size_t)i]) != q_bits(norm[(size_t)i])) {
                sub.norm_diff[i / 16 / 64] |= 1ULL << (i / 16 % 64);
                sub.any_norm_diff = 1;
            }
}

void ChipImage::one_layer(const TForward::Layer &L, ChipLayer &out, const std::vector<float> *norm_before) {
    const int NK = tf.nk_, NV = tf.nv_, ST = tf.st_, group = NV / NK;
    if (ST != CHIP_ST || tf.h_ != CHIP_HIDDEN || tf.w_ != CHIP_HIDDEN) dies("this image's shapes are 8192 and 128");

    // ------------------------------------------------------------------------- the attention side
    std::vector<ChipSell> work;
    std::vector<std::vector<uint32_t>> readers((size_t)CHIP_HIDDEN);
    uint32_t form_who = 0;
    const auto form_line = [&](const std::vector<uint32_t> &cols, const std::vector<std::array<float, 16>> &w, uint64_t dst) {
        ChipSell s;
        std::memset(&s, 0, sizeof s);
        s.s0 = slices(cols, w);
        s.len = (uint32_t)cols.size();
        s.dst = dst;
        s.changed_at = 0xFFFF0000u;
        s.who = form_who;
        const uint32_t id = (uint32_t)work.size();
        add_reader(readers, cols, id);
        work.push_back(s);
        return (int32_t)id;
    };

    // the q, k and v lines that have both a projection and taps: any other line of conv stays zero
    const std::map<uint32_t, LineForm> qkv = by_lines(L.qkv.row, L.qkv.start, L.qkv.col, L.qkv.val);
    std::vector<int> keys_used((size_t)NK, 0);
    for (const TForward::VState &vs : L.st) keys_used[(size_t)(vs.hd / group)] = 1;
    std::map<uint32_t, int32_t> conv_of;
    std::vector<uint32_t> conv_lines;
    const auto want_line = [&](uint32_t line) {
        if (!qkv.count(line)) return;
        bool any = false;
        for (uint32_t i = 0; i < 16 * 4; i++) any = any || L.conv[(uint64_t)line * 64 + i] != 0.0f;
        if (!any) return;
        conv_of[line] = (int32_t)conv_lines.size();
        conv_lines.push_back(line);
    };
    for (int kh = 0; kh < NK; kh++) {
        if (!keys_used[(size_t)kh]) continue;
        for (int j = 0; j < CHIP_ST_LINES; j++) want_line((uint32_t)(kh * CHIP_ST_LINES + j));
        for (int j = 0; j < CHIP_ST_LINES; j++) want_line((uint32_t)((NK + kh) * CHIP_ST_LINES + j));
    }
    for (const TForward::VState &vs : L.st)
        for (int j = 0; j < CHIP_ST_LINES; j++) want_line((uint32_t)((2 * NK + vs.hd) * CHIP_ST_LINES + j));
    out.n_conv = (uint32_t)conv_lines.size();
    out.o_act = alloc(((uint64_t)out.n_conv + 1) * 2);
    out.o_in = alloc((uint64_t)out.n_conv + 1);
    out.o_conv = alloc(out.n_conv * sizeof(ChipConv));
    for (uint32_t i = 0; i < out.n_conv; i++) {
        ChipConv &C = at<ChipConv>(out.o_conv)[i];
        const uint32_t line = conv_lines[i];
        const LineForm &f = qkv.at(line);
        C.conv_at = 0xFFFF0000u;
        form_who = 1u << 24 | i;
        C.work = form_line(f.cols, f.w, (uint64_t)((const uint8_t *)C.cur - blob_));
        form_who = 0;
        for (int t = 0; t < 4; t++)
            for (int lane = 0; lane < 16; lane++) C.w[t][lane] = L.conv[((uint64_t)line * 16 + (uint64_t)lane) * 4 + (uint64_t)t];
    }
    const auto conv_id = [&](uint32_t line) {
        const auto it = conv_of.find(line);
        return it == conv_of.end() ? (int32_t)-1 : it->second;
    };

    // the value heads with something in them, under their key heads
    const std::map<uint32_t, LineForm> gate = by_lines(L.gate.row, L.gate.start, L.gate.col, L.gate.val);
    out.n_heads = (uint32_t)L.st.size();
    out.o_heads = alloc(out.n_heads * sizeof(ChipHead));
    uint32_t n_keys = 0;
    for (int u : keys_used) n_keys += (uint32_t)u;
    out.n_keys = n_keys;
    out.o_keys = alloc(n_keys * sizeof(ChipKey));
    const auto row_form = [&](const TForward::Sparse &m, uint32_t row, uint64_t dst) {
        for (size_t k = 0; k < m.row.size(); k++)
            if (m.row[k] == row) {
                std::vector<uint32_t> cols;
                std::vector<std::array<float, 16>> w;
                for (uint32_t i = m.start[k]; i < m.start[k + 1]; i++) {
                    std::array<float, 16> a;
                    a.fill(0.0f);
                    a[0] = m.val[i];
                    cols.push_back(m.col[i]);
                    w.push_back(a);
                }
                return form_line(cols, w, dst);
            }
        return (int32_t)-1;
    };
    uint32_t kid = 0, hid = 0;
    std::vector<uint32_t> sorted_heads;
    for (int kh = 0; kh < NK; kh++) {
        if (!keys_used[(size_t)kh]) continue;
        ChipKey &K = at<ChipKey>(out.o_keys)[kid++];
        K.q_at = 0xFFFF0000u;
        for (int j = 0; j < CHIP_ST_LINES; j++) {
            K.q[j] = conv_id((uint32_t)(kh * CHIP_ST_LINES + j));
            K.k[j] = conv_id((uint32_t)((NK + kh) * CHIP_ST_LINES + j));
        }
        K.head0 = hid;
        const auto reads_of_key = [&](int32_t id) {
            if (id >= 0) at<ChipConv>(out.o_conv)[id].key = kid - 1;
        };
        for (int j = 0; j < CHIP_ST_LINES; j++) {
            reads_of_key(K.q[j]);
            reads_of_key(K.k[j]);
        }
        for (const TForward::VState &vs : L.st) {
            if (vs.hd / group != kh) continue;
            ChipHead &H = at<ChipHead>(out.o_heads)[hid++];
            H.touched_at = 0xFFFF0000u;
            form_who = 2u << 24 | (hid - 1);
            K.nheads++;
            H.hd = (uint32_t)vs.hd;
            H.o_at = 0xFFFF0000u;
            H.fresh = 1;
            H.mix_line = (uint32_t)vs.hd * CHIP_ST_LINES;
            H.dt_bits = q_bits(L.dt[(size_t)vs.hd]);
            H.a_bits = q_bits(L.a[(size_t)vs.hd]);
            for (int j = 0; j < CHIP_ST_LINES; j++) H.v[j] = conv_id((uint32_t)((2 * NK + vs.hd) * CHIP_ST_LINES + j));
            for (int j = 0; j < CHIP_ST_LINES; j++) reads_of_key(H.v[j]);
            for (int j = 0; j < CHIP_ST_LINES; j++) {
                const auto g = gate.find((uint32_t)(vs.hd * CHIP_ST_LINES + j));
                H.z[j] = g == gate.end() ? -1 : form_line(g->second.cols, g->second.w, (uint64_t)((const uint8_t *)(H.z_v + 16 * j) - blob_));
            }
            H.alpha = row_form(L.alpha, (uint32_t)vs.hd, (uint64_t)((const uint8_t *)H.alpha_v - blob_));
            H.beta = row_form(L.beta, (uint32_t)vs.hd, (uint64_t)((const uint8_t *)H.beta_v - blob_));
            form_who = 0;
            std::memcpy(H.S, vs.S.data(), sizeof H.S);
            for (int r = 0; r < CHIP_ST; r++) {
                bool any = false;
                for (int c = 0; c < CHIP_ST; c++) any = any || vs.S[(size_t)(r * CHIP_ST + c)] != 0.0f;
                if (any) H.rows_nz[r / 64] |= 1ULL << (r % 64);
            }
        }
    }

    // the memories of this layer
    out.n_banks = (uint32_t)L.banks.size();
    out.o_banks = alloc(out.n_banks * sizeof(ChipBank));
    for (uint32_t bi = 0; bi < out.n_banks; bi++) {
        const TForward::Bank &b = L.banks[bi];
        ChipBank &B = at<ChipBank>(out.o_banks)[bi];
        B.kind = b.kind;
        B.nra = b.nra;
        B.nwa = b.nwa;
        B.nd = b.nd;
        B.nout = (uint32_t)b.out.size();
        if (b.nra > 16 * CHIP_BANK_LINES || b.nwa > 16 * CHIP_BANK_LINES || b.nd > 16 * CHIP_BANK_LINES || b.out.size() > CHIP_BANK_OUTS) dies("a memory is wider than the image allows");
        for (size_t i = 0; i < b.out.size(); i++) B.out[i] = (uint16_t)b.out[i];
        for (int i = 0; i < CHIP_BANK_LINES; i++) B.ra[i] = B.wa[i] = B.data[i] = -1;
        B.we = -1;
        const auto forms = [&](const TForward::Sparse &m, int32_t *ids, float *dst, int lines) {
            for (const auto &l : by_lines(m.row, m.start, m.col, m.val)) {
                if ((int)l.first >= lines) dies("a memory's form has too many rows");
                ids[l.first] = form_line(l.second.cols, l.second.w, (uint64_t)((const uint8_t *)(dst + 16 * l.first) - blob_));
            }
        };
        forms(b.ra, B.ra, B.vra, CHIP_BANK_LINES);
        forms(b.wa, B.wa, B.vwa, CHIP_BANK_LINES);
        forms(b.data, B.data, B.cur, CHIP_BANK_LINES);
        forms(b.we, &B.we, B.we_v, 1);
        for (int i = 0; i < b.nd; i++)
            for (int t = 0; t < 4; t++) B.taps[t][i] = b.taps[(size_t)i * 4 + (size_t)t];
    }

    out.a.n_work = (uint32_t)work.size();
    out.a.o_work = alloc(work.size() * sizeof(ChipSell));
    if (!work.empty()) std::memcpy(at<ChipSell>(out.a.o_work), work.data(), work.size() * sizeof(ChipSell));
    fan_table(readers, &out.a.o_fan_start, &out.a.o_fan_item);
    reads_mask(out.a, readers);
    memo_tables(out.a, L.attn_norm, norm_before);
    out.a.o_input = alloc((uint64_t)NV * CHIP_ST * 4);  // the mix
    y_lines(out.a, L.out, (uint32_t)NV * CHIP_ST);

    out.o_ssm_norm = alloc((uint64_t)CHIP_ST * 4);
    std::memcpy(at<float>(out.o_ssm_norm), L.ssm_norm.data(), (size_t)CHIP_ST * 4);

    // ------------------------------------------------------------------------- the FFN side
    const std::map<uint32_t, LineForm> fg = by_lines(L.f_gate.row, L.f_gate.start, L.f_gate.col, L.f_gate.val), fu = by_lines(L.f_up.row, L.f_up.start, L.f_up.col, L.f_up.val);
    std::vector<std::vector<uint32_t>> ureaders((size_t)CHIP_HIDDEN);
    out.f.n_work = (uint32_t)fg.size();  // a line with no gate row has silu(0) = 0, so nothing it could give
    out.f.o_work = alloc(out.f.n_work * sizeof(ChipUnit));
    uint32_t uid = 0;
    for (const auto &l : fg) {
        ChipUnit &U = at<ChipUnit>(out.f.o_work)[uid];
        U.line = l.first;
        U.glen = (uint32_t)l.second.cols.size();
        U.g0 = slices(l.second.cols, l.second.w);
        const auto up = fu.find(l.first);
        std::vector<uint32_t> all = l.second.cols;
        if (up != fu.end()) {
            U.ulen = (uint32_t)up->second.cols.size();
            U.u0 = slices(up->second.cols, up->second.w);
            all.insert(all.end(), up->second.cols.begin(), up->second.cols.end());
            std::sort(all.begin(), all.end());
            all.erase(std::unique(all.begin(), all.end()), all.end());
        } else {
            U.u0 = U.g0;
        }
        add_reader(ureaders, all, uid);
        uid++;
    }
    fan_table(ureaders, &out.f.o_fan_start, &out.f.o_fan_item);
    reads_mask(out.f, ureaders);
    memo_tables(out.f, L.post_norm, &L.attn_norm);
    out.f.o_input = alloc((uint64_t)CHIP_HIDDEN * 4);  // the units
    y_lines(out.f, L.f_down, CHIP_HIDDEN);

    std::vector<uint32_t> gi_idx;
    std::vector<float> gi_w;
    for (int i = 0; i < CHIP_HIDDEN; i++)
        if (L.gate_inp[(size_t)i] != 0.0f) {
            gi_idx.push_back((uint32_t)i);
            gi_w.push_back(L.gate_inp[(size_t)i]);
        }
    out.n_gi = (uint32_t)gi_idx.size();
    out.gi_fresh = 1;
    out.o_gi_idx = alloc(std::max<uint64_t>(gi_idx.size(), 1) * 4);
    out.o_gi_w = alloc(std::max<uint64_t>(gi_w.size(), 1) * 4);
    if (!gi_idx.empty()) {
        std::memcpy(at<uint32_t>(out.o_gi_idx), gi_idx.data(), gi_idx.size() * 4);
        std::memcpy(at<float>(out.o_gi_w), gi_w.data(), gi_w.size() * 4);
    }
}

ChipImage::ChipImage(TForward &forward) : tf(forward) {
    void *mapped = mmap(nullptr, BLOB_MOST, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);  // page aligned, zeros, touched as it grows
    if (mapped == MAP_FAILED) dies("no address space for the image");
    blob_ = static_cast<uint8_t *>(mapped);
    const uint64_t o_h = alloc(sizeof(ChipHeader));
    const uint32_t layers = (uint32_t)tf.ls_.size();
    const uint64_t o_layers = alloc(layers * sizeof(ChipLayer));
    for (uint32_t il = 0; il < layers; il++) {
        const TForward::Layer &L = tf.ls_[il];
        const std::vector<float> *before = il ? &tf.ls_[il - 1].post_norm : &tf.ls_[layers - 1].post_norm;
        one_layer(L, at<ChipLayer>(o_layers)[il], before);
    }

    const float EPS = 1e-6f;
    const uint64_t o_byte = alloc(256 * 4);
    for (int i = 0; i < 256; i++) at<float>(o_byte)[i] = (float)i / 256.0f;
    const uint64_t o_x = alloc((uint64_t)CHIP_HIDDEN * 4 + 64), o_hn = alloc((uint64_t)CHIP_HIDDEN * 4 + 64), o_ytmp = alloc((uint64_t)CHIP_HIDDEN * 4 + 64);
    const uint64_t o_hp = alloc((uint64_t)CHIP_HIDDEN * 4 + 64);

    // the embedding, by rows
    const TForward::Sparse emb = tf.load_sparse("chip.token_embd.weight");
    const gguf::Tensor &et = tf.f_.tensor("chip.token_embd.weight");
    const uint32_t vocab = (uint32_t)et.dims[1];
    const uint64_t o_used = alloc((uint64_t)CHIP_HIDDEN);
    for (uint32_t c : emb.col) at<uint8_t>(o_used)[c] = 1;
    const uint64_t o_start = alloc(((uint64_t)vocab + 1) * 4), o_ecol = alloc(emb.col.size() * 2, 2), o_eval = alloc(emb.val.size() * 4);
    uint32_t *st = at<uint32_t>(o_start);
    {
        size_t k = 0;
        for (uint32_t t = 0; t < vocab; t++) {
            st[t] = emb.start[k];  // a row with nothing in it is empty, at the next one's start
            if (k < emb.row.size() && emb.row[k] == t) k++;
        }
        st[vocab] = (uint32_t)emb.col.size();
    }
    uint16_t *ec = at<uint16_t>(o_ecol);
    for (size_t i = 0; i < emb.col.size(); i++) ec[i] = (uint16_t)emb.col[i];
    std::memcpy(at<float>(o_eval), emb.val.data(), emb.val.size() * 4);

    // the shared arrays last: every layer has added to them
    const uint64_t o_col = alloc(col.size() * 4), o_woff = alloc(woff.size() * 4), o_wtab = alloc(wtab.size() * 4);
    std::memcpy(at<uint32_t>(o_col), col.data(), col.size() * 4);
    std::memcpy(at<uint32_t>(o_woff), woff.data(), woff.size() * 4);
    std::memcpy(at<float>(o_wtab), wtab.data(), wtab.size() * 4);

    // the memories follow the blob, each from where the forward holds it
    uint64_t next = (size_ + 4095) / 4096 * 4096;
    image_bytes_ = next;
    for (uint32_t il = 0; il < layers; il++) {
        ChipLayer &CL = at<ChipLayer>(o_layers)[il];
        const TForward::Layer &L = tf.ls_[il];
        for (uint32_t bi = 0; bi < CL.n_banks; bi++) {
            ChipBank &B = at<ChipBank>(CL.o_banks)[bi];
            const TForward::Bank &b = L.banks[bi];
            const std::string name = "chip.bank." + std::to_string(bank_index(il, bi)) + ".mem";
            const gguf::Tensor &m = tf.f_.tensor(name);
            B.o_mem = next;
            B.mem_bytes = m.nbytes;
            Piece p;
            p.to = next;
            p.bytes = m.nbytes;
            p.kind = b.kind;
            p.from = b.kind == 0 ? b.bytes : b.mem.data();
            pieces_.push_back(p);
            next += (m.nbytes + 4095) / 4096 * 4096;
        }
    }

    ChipHeader &h = *at<ChipHeader>(o_h);
    h.magic = CHIP_IMAGE_MAGIC;
    h.image_bytes = image_bytes_;
    h.total_bytes = next;
    h.layers = layers;
    h.vocab = vocab;
    h.go_token = (uint32_t)tf.meta("qwen35moe.chip.go_token");
    h.byte_token = (uint32_t)tf.meta("qwen35moe.chip.byte_token");
    h.system_tokens = (uint32_t)tf.meta("qwen35moe.chip.system_tokens");
    h.o_layers = o_layers;
    h.o_col = o_col;
    h.o_woff = o_woff;
    h.o_wtab = o_wtab;
    h.nnz = tf.nnz_;
    h.slices = col.size();
    h.wlines = wtab.size() / 16;
    h.o_emb_start = o_start;
    h.o_emb_col = o_ecol;
    h.o_emb_val = o_eval;
    h.emb_nnz = emb.col.size();
    h.o_byte_f = o_byte;
    h.o_x = o_x;
    h.o_hn = o_hn;
    h.o_ytmp = o_ytmp;
    h.o_hp = o_hp;
    h.o_row_used = o_used;
    h.eps_bits = q_bits(EPS);
    h.e2_bits = q_bits(EPS * EPS);
    h.over_h_bits = q_bits(1.0f / (float)CHIP_HIDDEN);
    h.over_st_bits = q_bits(1.0f / (float)CHIP_ST);
    h.q_scale_bits = q_bits(0.0883883461f);
    h.one_bits = q_bits(1.0f);
    size_ = image_bytes_;
}

// The file numbers its memories in the order the forward attaches them to layers, so a layer's K is found by counting.
uint32_t ChipImage::bank_index(uint32_t il, uint32_t bi) const {
    const int nb = (int)tf.f_.get_int("qwen35moe.chip.banks");
    uint32_t seen = 0;
    for (int K = 0; K < nb; K++) {
        const std::string p = "chip.bank." + std::to_string(K) + ".info";
        float info[6];
        gguf::read_at(tf.f_, tf.f_.tensor(p).offset, info, sizeof info);
        if ((uint32_t)info[1] != il) continue;
        if (seen == bi) return (uint32_t)K;
        seen++;
    }
    dies("a layer's memory is not in the file");
}
