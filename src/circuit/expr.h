// Exact linear forms over the residual stream's dims.
//
// Port of Expr in QwenPU chipformer/dsl/graph.py (MIT). Upstream's coefficients are Python ints
// and Fractions; every one the circuit produces is dyadic, so Dy holds n * 2^e exactly in 64 + 32
// bits and aborts on anything else: a non-dyadic value or an overflow is a defect in the port, never
// a value to round.
#ifndef QWENPU_CIRCUIT_EXPR_H
#define QWENPU_CIRCUIT_EXPR_H

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

struct Dy {
    int64_t n = 0;
    int32_t e = 0;

    Dy() = default;
    Dy(int v) : n(v) { norm(); }
    Dy(long v) : n(v) { norm(); }
    Dy(long long v) : n(v) { norm(); }
    Dy(double v) {
        int ex = 0;
        double m = std::frexp(v, &ex);
        n = (int64_t)std::ldexp(m, 53);
        e = ex - 53;
        if (std::ldexp((double)n, e) != v) die("double is not exactly dyadic in 64 bits");
        norm();
    }
    static Dy pow2(int k) {
        Dy d;
        d.n = 1;
        d.e = k;
        return d;
    }
    [[noreturn]] static void die(const char *what) {
        std::fprintf(stderr, "Dy: %s\n", what);
        std::abort();
    }
    void norm() {
        if (n == 0) {
            e = 0;
            return;
        }
        int tz = __builtin_ctzll((uint64_t)n);
        n >>= tz;
        e += tz;
    }
    bool zero() const { return n == 0; }
    double to_double() const {
        double v = std::ldexp((double)n, e);
        if ((int64_t)std::ldexp(v, -e) != n) die("value does not fit a double");
        return v;
    }
    friend bool operator==(const Dy &a, const Dy &b) { return a.n == b.n && a.e == b.e; }
    friend bool operator!=(const Dy &a, const Dy &b) { return !(a == b); }
    // A total order on the representation, for use as a key; not the numeric order.
    friend bool operator<(const Dy &a, const Dy &b) { return a.n != b.n ? a.n < b.n : a.e < b.e; }
    friend Dy operator-(const Dy &a) {
        Dy r = a;
        r.n = -a.n;
        return r;
    }
    friend Dy operator+(const Dy &a, const Dy &b) {
        if (a.n == 0) return b;
        if (b.n == 0) return a;
        const Dy &lo = a.e <= b.e ? a : b, &hi = a.e <= b.e ? b : a;
        int s = hi.e - lo.e;
        if (s > 62) die("sum spans more than 62 bits");
        int64_t shifted, sum;
        if (__builtin_mul_overflow(hi.n, (int64_t)1 << s, &shifted)
            || __builtin_add_overflow(lo.n, shifted, &sum))
            die("sum overflows");
        Dy r;
        r.n = sum;
        r.e = lo.e;
        r.norm();
        return r;
    }
    friend Dy operator-(const Dy &a, const Dy &b) { return a + (-b); }
    friend Dy operator*(const Dy &a, const Dy &b) {
        Dy r;
        if (__builtin_mul_overflow(a.n, b.n, &r.n)) die("product overflows");
        r.e = a.e + b.e;
        r.norm();
        return r;
    }
    Dy half() const {
        Dy r = *this;
        r.e -= 1;
        if (r.n == 0) r.e = 0;
        return r;
    }
};

constexpr int ONE = 0;  // dim id of the constant-one feature

struct Expr {
    std::vector<std::pair<int, Dy>> t;  // sorted by dim, no zero coefficients

    Expr() = default;
    static Expr dim(int d, Dy c = Dy(1)) {
        Expr e;
        if (!c.zero()) e.t.emplace_back(d, c);
        return e;
    }
    bool empty() const { return t.empty(); }
    int only_dim() const {
        if (t.size() != 1) Dy::die("Expr is not a single dim");
        return t[0].first;
    }
    friend Expr operator+(const Expr &a, const Expr &b) {
        Expr r;
        r.t.reserve(a.t.size() + b.t.size());
        size_t i = 0, j = 0;
        while (i < a.t.size() || j < b.t.size()) {
            if (j == b.t.size() || (i < a.t.size() && a.t[i].first < b.t[j].first)) {
                r.t.push_back(a.t[i++]);
            } else if (i == a.t.size() || b.t[j].first < a.t[i].first) {
                r.t.push_back(b.t[j++]);
            } else {
                Dy s = a.t[i].second + b.t[j].second;
                if (!s.zero()) r.t.emplace_back(a.t[i].first, s);
                i++;
                j++;
            }
        }
        return r;
    }
    friend Expr operator-(const Expr &a) {
        Expr r = a;
        for (auto &p : r.t) p.second = -p.second;
        return r;
    }
    friend Expr operator-(const Expr &a, const Expr &b) { return a + (-b); }
    friend Expr operator*(const Expr &a, const Dy &c) {
        Expr r;
        if (c.zero()) return r;
        r.t.reserve(a.t.size());
        for (const auto &p : a.t) r.t.emplace_back(p.first, p.second * c);
        return r;
    }
    friend Expr operator*(const Dy &c, const Expr &a) { return a * c; }
    Expr &operator+=(const Expr &o) { return *this = *this + o; }
    Expr &operator-=(const Expr &o) { return *this = *this - o; }

    // Constants live on dim ONE, as as_expr does.
    friend Expr operator+(const Expr &a, const Dy &c) { return a + Expr::dim(ONE, c); }
    friend Expr operator+(const Dy &c, const Expr &a) { return a + Expr::dim(ONE, c); }
    friend Expr operator-(const Expr &a, const Dy &c) { return a + Expr::dim(ONE, -c); }
    friend Expr operator-(const Dy &c, const Expr &a) { return Expr::dim(ONE, c) - a; }
    Expr &operator+=(const Dy &c) { return *this = *this + c; }
    Expr &operator-=(const Dy &c) { return *this = *this - c; }
};

using L = std::vector<Expr>;

// Python list concatenation for the literal and term lists.
template <class T>
std::vector<T> operator+(std::vector<T> a, const std::vector<T> &b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}
template <class T>
std::vector<T> &operator+=(std::vector<T> &a, const std::vector<T> &b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

// v[a:b], clamped as Python slices are (for the non-negative bounds used here).
template <class T>
std::vector<T> sl(const std::vector<T> &v, int a, int b) {
    int n = (int)v.size();
    a = a < n ? a : n;
    b = b < n ? b : n;
    if (b < a) b = a;
    return std::vector<T>(v.begin() + a, v.begin() + b);
}

inline L rep(const Expr &e, int n) { return L((size_t)n, e); }

inline Expr sum(const L &v) {
    // Upstream's sum() rebuilds the dict per term; a k-way merge by repeated doubling is the same
    // exact value without the quadratic cost on long sums.
    if (v.empty()) return Expr();
    L cur = v;
    while (cur.size() > 1) {
        L nxt;
        nxt.reserve((cur.size() + 1) / 2);
        for (size_t i = 0; i + 1 < cur.size(); i += 2) nxt.push_back(cur[i] + cur[i + 1]);
        if (cur.size() % 2) nxt.push_back(cur.back());
        cur.swap(nxt);
    }
    return cur[0];
}

// Python floor division and modulo.
inline long long fdiv_py(long long a, long long b) {
    long long q = a / b;
    return ((a % b != 0) && ((a < 0) != (b < 0))) ? q - 1 : q;
}
inline long long fmod_py(long long a, long long b) { return a - fdiv_py(a, b) * b; }
inline int bit_length(long long v) { return v ? 64 - __builtin_clzll((unsigned long long)v) : 0; }

#endif
