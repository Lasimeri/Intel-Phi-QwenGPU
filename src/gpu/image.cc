// See image.h.
#include "image.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>


#include "../guest/q2b_map.h"

namespace {

[[noreturn]] void die(const std::string &what) {
    std::fprintf(stderr, "image: %s\n", what.c_str());
    std::exit(2);
}

double half_value(uint16_t h) {
    const int sign = h >> 15, exp = h >> 10 & 31, man = h & 1023;
    double m = exp ? 1.0 + man / 1024.0 : man / 1024.0;
    double v = std::ldexp(m, (exp ? exp : 1) - 15);
    return sign ? -v : v;
}

template <typename T>
void read_all(const std::string &path, std::vector<T> &out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) die("cannot open " + path);
    f.seekg(0, std::ios::end);
    std::streamoff n = f.tellg();
    f.seekg(0);
    if (n % (std::streamoff)sizeof(T)) die(path + ": size is not a whole number of elements");
    out.resize((size_t)n / sizeof(T));
    f.read((char *)out.data(), n);
    if (!f) die("short read of " + path);
}

}  // namespace

ImageMeta image_meta(const std::string &dir) {
    std::ifstream f(dir + "/meta.txt");
    if (!f) die("cannot open " + dir + "/meta.txt");
    ImageMeta m;
    std::string key;
    uint64_t v;
    while (f >> key >> v) {
        if (key == "nblk") m.nblk = (uint32_t)v;
        else if (key == "descriptors") m.descriptors = (uint32_t)v;
        else if (key == "m_words") m.m_words = (uint32_t)v;
        else if (key == "a_blocks") m.a_blocks = (uint32_t)v;
        else if (key == "s_elems") m.s_elems = (uint32_t)v;
        else if (key == "w_blocks") m.w_blocks = v;
        else if (key == "desc_step_end") m.desc_step_end = (uint32_t)v;
        else if (key == "desc_cont") m.desc_cont = (uint32_t)v;
        else if (key == "desc_halt") m.desc_halt = (uint32_t)v;
    }
    return m;
}

void Image::bind() {
    std::memset(&img, 0, sizeof img);
    uint8_t classes[SM_STAGES] = SM_STAGE_CLASSES_DEFAULT;
    std::memcpy(img.classes, classes, sizeof classes);
    sm_classes_env(img.classes);
    img.rom = rom.data();
    img.nblk = (uint32_t)rom.size();
    img.m = m.data();
    img.m_words = (uint32_t)m.size();
    img.a = a.data();
    img.a_blocks = a_blocks;
    img.s = s.data();
    img.s_elems = s_elems;
    img.w = w;
    img.w_blocks = (uint32_t)(w_bytes / 34);
    img.c = nullptr;
    img.c_bytes = 0;
    img.f16_q24 = f16_q24.data();
}

void image_bare(Image &im, uint32_t m_words, uint32_t a_blocks, uint32_t s_elems) {
    im.rom.clear();
    im.m.assign(m_words, 0);
    im.a_blocks = a_blocks;
    im.a.assign((size_t)a_blocks * 36, 0);
    im.s_elems = s_elems;
    im.s.assign(s_elems, 0);
    im.w = nullptr;
    im.w_bytes = 0;
    im.f16_q24.resize(65536);
    for (uint32_t h = 0; h < 65536; h++) im.f16_q24[h] = (int64_t)std::llrint(half_value((uint16_t)h) * 16777216.0);
    im.descriptors = 0;
    im.bind();
}

void image_load(Image &im, const std::string &dir) {
    ImageMeta meta = image_meta(dir);
    read_all(dir + "/rom.bin", im.rom);
    if (im.rom.size() != meta.nblk) die("rom.bin holds " + std::to_string(im.rom.size()) + " blocks, meta says " + std::to_string(meta.nblk));
    read_all(dir + "/m.bin", im.m);
    if (im.m.size() != meta.m_words) die("m.bin size differs from meta");
    {
        std::string wp = dir + "/w.q8";
        int fd = open(wp.c_str(), O_RDONLY);
        if (fd < 0) die("cannot open " + wp);
        struct stat st;
        if (fstat(fd, &st)) die("stat " + wp);
        im.w_bytes = (uint64_t)st.st_size;
        if (im.w_bytes != meta.w_blocks * 34) die("w.q8 size differs from meta");
        void *p = mmap(nullptr, im.w_bytes, PROT_READ, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) die("mmap " + wp);
        im.w = (const uint8_t *)p;
        close(fd);
    }
    im.a_blocks = meta.a_blocks;
    im.a.assign((size_t)meta.a_blocks * 36, 0);
    // V transposed carries the scale 1.0 in every block: the activation side carries the probabilities' scale
    {
        int32_t one = 1 << 24;
        for (uint32_t b = A_VT; b < (uint32_t)A_END && b < meta.a_blocks; b++) std::memcpy(&im.a[(size_t)b * 36 + 32], &one, 4);
    }
    im.s_elems = meta.s_elems;
    im.s.assign(meta.s_elems, 0);
    im.f16_q24.resize(65536);
    for (uint32_t h = 0; h < 65536; h++) im.f16_q24[h] = (int64_t)std::llrint(half_value((uint16_t)h) * 16777216.0);
    im.descriptors = meta.descriptors;
    im.bind();
}
