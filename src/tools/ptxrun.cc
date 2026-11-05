// ptxrun KERNEL.ptx NAME GX GY BX BY PARAM... [-v] [-a word,word,...]
//   PARAM: c:VALUE (a constant), p:N[,v0,v1,...] (a buffer of N words, zero past the values given), a:K (runtime word
//   M[u14 + K]; -a gives the args block; -base N puts the kernel behind N filler blocks). Lowers the kernel, runs one
//   launch on the reference machine and prints every buffer, the tokens and the hash. Host, a check of the driver
//   against known answers.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../gpu/lower.h"
#include "../gpu/sm_ref.h"

static std::vector<int64_t> split_ints(const std::string &s) {
    std::vector<int64_t> out;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        if (j > i) out.push_back(std::strtoll(s.substr(i, j - i).c_str(), nullptr, 0));
        i = j + 1;
    }
    return out;
}

int main(int argc, char **argv) {
    if (argc < 7) {
        std::fputs("usage: ptxrun KERNEL.ptx NAME GX GY BX BY PARAM... [-v] [-a w,w,...]\n", stderr);
        return 2;
    }
    auto ks = ptx::load(argv[1]);
    if (!ks.count(argv[2])) {
        std::fprintf(stderr, "no kernel %s\n", argv[2]);
        return 2;
    }
    const ptx::Kernel &k = ks[argv[2]];
    int gx = std::atoi(argv[3]), gy = std::atoi(argv[4]), bx = std::atoi(argv[5]), by = std::atoi(argv[6]);
    bool verbose = false;
    uint32_t base = 0;  // -base N: the kernel behind N filler blocks that end a thread block at once
    std::vector<int64_t> args;
    std::vector<lower::ParamSpec> params;
    struct Buf {
        uint32_t base, n;
    };
    std::vector<Buf> bufs;
    std::vector<int32_t> m(1u << 20, 0);
    uint32_t top = SM_M_USER + 1024;  // buffers start here; the args block at SM_M_USER
    for (int i = 7; i < argc; i++) {
        std::string s = argv[i];
        if (s == "-v") {
            verbose = true;
            continue;
        }
        if (s == "-a") {
            args = split_ints(argv[++i]);
            continue;
        }
        if (s == "-base") {
            base = (uint32_t)std::atoi(argv[++i]);
            continue;
        }
        lower::ParamSpec ps;
        if (s.compare(0, 2, "c:") == 0) {
            ps.kind = lower::ParamSpec::CONST;
            ps.value = std::strtoll(s.c_str() + 2, nullptr, 0);
        } else if (s.compare(0, 2, "a:") == 0) {
            ps.kind = lower::ParamSpec::ARG;
            ps.value = std::strtoll(s.c_str() + 2, nullptr, 0);
        } else if (s.compare(0, 2, "p:") == 0) {
            std::vector<int64_t> v = split_ints(s.substr(2));
            uint32_t n = (uint32_t)v[0];
            Buf b{top, n};
            for (size_t q = 1; q < v.size() && q - 1 < n; q++) m[top + q - 1] = (int32_t)v[q];
            top += n;
            bufs.push_back(b);
            ps.kind = lower::ParamSpec::PTR;
            ps.value = b.base;
        } else {
            std::fprintf(stderr, "bad param %s\n", s.c_str());
            return 2;
        }
        params.push_back(ps);
    }
    for (size_t i = 0; i < args.size(); i++) m[SM_M_USER + i] = (int32_t)args[i];
    lower::Options opt;
    opt.shared_base = top;
    lower::Launch L;
    try {
        L = lower::lower(k, gx, gy, bx, by, params, opt);
    } catch (const lower::Unsupported &e) {
        std::printf("unsupported: %s\n", e.what());
        return 3;
    }
    if (verbose) std::fputs(lower::listing(L).c_str(), stdout);
    std::vector<SmBlock> rom;
    {
        // filler blocks before the kernel: nops that END the thread block, so a warp sent to block 0 by mistake does
        // nothing and the known answer comes out wrong
        SmBlock fill;
        std::memset(&fill, 0, sizeof fill);
        SmLive lv;
        std::memset(&lv, 0, sizeof lv);
        lv.branch = SM_BR_END;
        fill.live = sm_live_encode(&lv);
        rom.assign(base, fill);
    }
    lower::encode(L, rom, base);
    // the queue: one launch, then halt
    m[SM_M_QUEUE + 0] = (int32_t)sm_qword(SM_Q_LAUNCH, base, L.warps, (uint32_t)L.lwpb, 0);
    m[SM_M_QUEUE + 1] = (int32_t)SM_M_USER;
    m[SM_M_QUEUE + 2] = (int32_t)sm_qword(SM_Q_HALT, 0, 0, 0, 0);
    SmImage img;
    std::memset(&img, 0, sizeof img);
    uint8_t classes[SM_STAGES] = SM_STAGE_CLASSES_DEFAULT;
    std::memcpy(img.classes, classes, sizeof classes);
    sm_classes_env(img.classes);
    img.rom = rom.data();
    img.nblk = (uint32_t)rom.size();
    img.m = m.data();
    img.m_words = (uint32_t)m.size();
    std::vector<int64_t> tab(65536, 0);
    tab[0x3C00] = 1 << 24;  // f16 1.0
    img.f16_q24 = tab.data();
    // small banks for the probes: A block 0 = ones with scale 1.0; W block j = all (j + 1), scale 1.0; S and C empty
    std::vector<uint8_t> abank(256 * 36, 0), wbank(64 * 34, 0);
    for (int j = 0; j < 32; j++) abank[j] = 1;
    {
        int32_t one = 1 << 24;
        std::memcpy(&abank[32], &one, 4);
    }
    for (int j = 0; j < 64; j++) {
        uint16_t h = 0x3C00;
        std::memcpy(&wbank[j * 34], &h, 2);
        for (int q = 0; q < 32; q++) wbank[j * 34 + 2 + q] = (uint8_t)(int8_t)(j + 1);
    }
    std::vector<int16_t> sbank(4096, 0);
    std::vector<int8_t> cbank(4096, 0);
    img.a = abank.data();
    img.a_blocks = 256;
    img.w = wbank.data();
    img.w_blocks = 64;
    img.s = sbank.data();
    img.s_elems = 4096;
    img.c = cbank.data();
    img.c_bytes = 4096;
    SmMachine *mc = (SmMachine *)std::calloc(1, sizeof *mc);
    sm_init(mc, &img);
    int rc = sm_run(mc, 10000000);
    std::printf("%s: %zu blocks, %u warps, %llu records, %llu tokens, hash %016llx, rc %d %s\n", k.name.c_str(), L.blocks.size(), L.warps,
                (unsigned long long)mc->records, (unsigned long long)mc->sink.n, (unsigned long long)mc->sink.hash, rc, mc->fault_what);
    for (size_t b = 0; b < bufs.size(); b++) {
        std::printf("buffer %zu:", b);
        for (uint32_t i = 0; i < bufs[b].n; i++) std::printf(" %d", m[bufs[b].base + i]);
        std::printf("\n");
    }
    return rc == 0 ? 0 : 1;
}
