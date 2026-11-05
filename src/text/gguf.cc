#include "gguf.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace gguf {

namespace {

[[noreturn]] void die(const std::string &path, const char *what) {
    fprintf(stderr, "gguf: %s: %s\n", path.c_str(), what);
    exit(1);
}

/* A buffered forward reader over the file's header. */
struct Reader {
    int fd;
    const std::string &path;
    std::vector<uint8_t> buf;
    size_t have = 0, at = 0;
    uint64_t consumed = 0;

    Reader(int d, const std::string &p) : fd(d), path(p), buf(1u << 20) {}

    void bytes(void *dst, size_t n) {
        uint8_t *out = static_cast<uint8_t *>(dst);
        consumed += n;
        while (n) {
            if (at == have) {
                ssize_t got = ::read(fd, buf.data(), buf.size());
                if (got <= 0) die(path, "truncated header");
                have = static_cast<size_t>(got);
                at = 0;
            }
            size_t take = have - at < n ? have - at : n;
            memcpy(out, buf.data() + at, take);
            at += take;
            out += take;
            n -= take;
        }
    }
    template <class T> T get() {
        T v;
        bytes(&v, sizeof v);
        return v;
    }
    std::string str() {
        uint64_t n = get<uint64_t>();
        if (n > (1ull << 30)) die(path, "absurd string length");
        std::string s(n, '\0');
        if (n) bytes(&s[0], n);
        return s;
    }
};

enum : uint32_t { V_U8, V_I8, V_U16, V_I16, V_U32, V_I32, V_F32, V_BOOL, V_STR, V_ARR, V_U64, V_I64, V_F64 };

uint64_t scalar_int(Reader &r, uint32_t t) {
    switch (t) {
        case V_U8: case V_BOOL: return r.get<uint8_t>();
        case V_I8: return static_cast<uint64_t>(static_cast<int64_t>(r.get<int8_t>()));
        case V_U16: return r.get<uint16_t>();
        case V_I16: return static_cast<uint64_t>(static_cast<int64_t>(r.get<int16_t>()));
        case V_U32: return r.get<uint32_t>();
        case V_I32: return static_cast<uint64_t>(static_cast<int64_t>(r.get<int32_t>()));
        case V_U64: return r.get<uint64_t>();
        case V_I64: return static_cast<uint64_t>(r.get<int64_t>());
        default: die(r.path, "not an integer value");
    }
}

}  // namespace

const Tensor &File::tensor(const std::string &name) const {
    auto it = by_name.find(name);
    if (it == by_name.end()) die(path, ("no tensor " + name).c_str());
    return tensors[it->second];
}

uint64_t File::get_int(const std::string &key) const {
    auto it = ints.find(key);
    if (it == ints.end()) die(path, ("no key " + key).c_str());
    return it->second;
}

uint64_t row_bytes(uint32_t type, uint64_t n) {
    switch (type) {
        case T_F32: return n * 4;
        case T_Q8_0:
            if (n % 32) break;
            return n / 32 * 34;
        case T_Q6_K:
            if (n % 256) break;
            return n / 256 * 212;
        default: break;
    }
    fprintf(stderr, "gguf: no row size for type %u, %llu elements\n", type, static_cast<unsigned long long>(n));
    exit(1);
}

void open(File &f, const char *path) {
    f.path = path;
    f.fd = ::open(path, O_RDONLY);
    if (f.fd < 0) die(f.path, "cannot open");
    struct stat st;
    if (fstat(f.fd, &st)) die(f.path, "cannot stat");
    f.size = static_cast<uint64_t>(st.st_size);
    Reader r(f.fd, f.path);
    char magic[4];
    r.bytes(magic, 4);
    if (memcmp(magic, "GGUF", 4)) die(f.path, "not GGUF");
    if (r.get<uint32_t>() != 3) die(f.path, "not version 3");
    uint64_t nt = r.get<uint64_t>(), nk = r.get<uint64_t>();
    for (uint64_t i = 0; i < nk; i++) {
        std::string key = r.str();
        uint32_t t = r.get<uint32_t>();
        if (t == V_STR) {
            f.strs[key] = r.str();
        } else if (t == V_F32) {
            f.floats[key] = static_cast<double>(r.get<float>());
        } else if (t == V_F64) {
            f.floats[key] = r.get<double>();
        } else if (t == V_ARR) {
            uint32_t e = r.get<uint32_t>();
            uint64_t n = r.get<uint64_t>();
            if (e == V_STR) {
                std::vector<std::string> a;
                a.reserve(n);
                for (uint64_t j = 0; j < n; j++) a.push_back(r.str());
                f.str_arrays[key] = std::move(a);
            } else if (e == V_F32) {
                for (uint64_t j = 0; j < n; j++) r.get<float>();
            } else {
                std::vector<int64_t> a;
                a.reserve(n);
                for (uint64_t j = 0; j < n; j++) a.push_back(static_cast<int64_t>(scalar_int(r, e)));
                f.int_arrays[key] = std::move(a);
            }
        } else {
            f.ints[key] = scalar_int(r, t);
        }
    }
    f.tensors.resize(nt);
    for (uint64_t i = 0; i < nt; i++) {
        Tensor &T = f.tensors[i];
        T.name = r.str();
        T.n_dims = r.get<uint32_t>();
        if (T.n_dims > 4) die(f.path, "tensor with more than 4 dims");
        uint64_t rest = 1;
        for (uint32_t d = 0; d < T.n_dims; d++) {
            T.dims[d] = r.get<uint64_t>();
            if (d) rest *= T.dims[d];
        }
        T.type = r.get<uint32_t>();
        T.offset = r.get<uint64_t>();
        T.nbytes = row_bytes(T.type, T.dims[0]) * rest;
        f.by_name[T.name] = i;
    }
    uint64_t align = f.ints.count("general.alignment") ? f.ints["general.alignment"] : 32;
    uint64_t data = (r.consumed + align - 1) / align * align;
    for (Tensor &T : f.tensors) {
        T.offset += data;
        if (T.offset + T.nbytes > f.size) die(f.path, ("tensor past the end: " + T.name).c_str());
    }
}

void close(File &f) {
    if (f.fd >= 0) ::close(f.fd);
    f.fd = -1;
}

void read_at(const File &f, uint64_t off, void *dst, uint64_t n) {
    uint8_t *p = static_cast<uint8_t *>(dst);
    while (n) {
        ssize_t got = pread(f.fd, p, n, static_cast<off_t>(off));
        if (got <= 0) die(f.path, "read failed");
        p += got;
        off += static_cast<uint64_t>(got);
        n -= static_cast<uint64_t>(got);
    }
}

}  // namespace gguf
