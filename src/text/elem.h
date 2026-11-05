/* The text path's elementwise functions, defined once as exact sequences of IEEE F32 operations.
 *
 * KNC has no vector divide or square root, and its estimate instructions (vexp223ps, vrcp23ps,
 * vrsqrt23ps) cannot be reproduced on another machine. These functions use only add, multiply,
 * fused multiply-add, round-to-nearest-even and integer operations on the bits, so a card lane and
 * this scalar code give the same bits. That is what lets a generated token stream be checked bit for
 * bit afterwards, by a program that shares nothing with the card but this definition.
 *
 * Compiled for the host with -mfma: __builtin_fmaf is the hardware instruction, never libm. The card's
 * vector forms live in src/card/q35_elem.h and follow this file operation by operation. */
#ifndef TEXT_ELEM_H
#define TEXT_ELEM_H

#include <math.h>
#include <stdint.h>
#include <string.h>

#define Q_FMA(a, b, c) __builtin_fmaf((a), (b), (c))

static inline uint32_t q_bits(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}
static inline float q_float(uint32_t u) {
    float f;
    memcpy(&f, &u, 4);
    return f;
}

/* e^x. Below -87 the result is exactly 0, so no lane ever makes a subnormal. */
static inline float q_exp(float x) {
    if (x < -87.0f) return 0.0f;
    if (x > 88.0f) x = 88.0f;
    /* round to nearest, even on ties, by adding 1.5 x 2^23: the sum's low mantissa bits are the integer */
    float t = x * 1.44269504088896341f + 12582912.0f;
    int32_t ni = (int32_t)(q_bits(t) - 0x4B400000u);
    float n = t - 12582912.0f;
    float r = Q_FMA(-n, 0.693359375f, x);
    r = Q_FMA(-n, -2.12194440054690583e-4f, r);
    float p = 1.98412698e-4f;
    p = Q_FMA(p, r, 1.38888889e-3f);
    p = Q_FMA(p, r, 8.33333333e-3f);
    p = Q_FMA(p, r, 4.16666667e-2f);
    p = Q_FMA(p, r, 1.66666667e-1f);
    p = Q_FMA(p, r, 0.5f);
    p = Q_FMA(p, r, 1.0f);
    p = Q_FMA(p, r, 1.0f);
    return q_float(q_bits(p) + ((uint32_t)ni << 23));
}

/* 1/a for a normal and positive. */
static inline float q_rcp(float a) {
    float y = q_float(0x7EF311C7u - q_bits(a));
    for (int i = 0; i < 4; i++) {
        float e = Q_FMA(-a, y, 1.0f);
        y = Q_FMA(y, e, y);
    }
    return y;
}

/* 1/sqrt(a) for a normal and positive. */
static inline float q_rsqrt(float a) {
    float y = q_float(0x5F375A86u - (q_bits(a) >> 1));
    for (int i = 0; i < 4; i++) {
        float t = a * y;
        float e = Q_FMA(-t, y, 1.0f);
        float h = y * 0.5f;
        y = Q_FMA(h, e, y);
    }
    return y;
}

/* ln a for a >= 1. */
static inline float q_log(float a) {
    uint32_t u = q_bits(a);
    int32_t e = (int32_t)(u >> 23) - 127;
    float m = q_float((u & 0x007FFFFFu) | 0x3F800000u);
    if (m > 1.41421354f) {
        m = m * 0.5f;
        e = e + 1;
    }
    float s = (m - 1.0f) * q_rcp(m + 1.0f);
    float z = s * s;
    float p = 0.111111111f;
    p = Q_FMA(p, z, 0.142857143f);
    p = Q_FMA(p, z, 0.2f);
    p = Q_FMA(p, z, 0.333333333f);
    p = Q_FMA(p, z, 1.0f);
    float two_s = s * 2.0f;
    float ef = q_float(0x4B400000u + (uint32_t)e) - 12582912.0f;
    return Q_FMA(ef, 0.693147182f, two_s * p);
}

/* Below -87 the true value is under every normal float, and q_rcp takes normal inputs only: exactly 0, as the compiled layers rely on. */
static inline float q_sigmoid(float x) { return x < -87.0f ? 0.0f : q_rcp(1.0f + q_exp(-x)); }
static inline float q_silu(float x) { return x * q_sigmoid(x); }
static inline float q_softplus(float x) { return x > 20.0f ? x : q_log(1.0f + q_exp(x)); }

/* The reduction every sum over a vector uses: sixteen lanes, then pairs i^1, i^2, i^4, i^8. The card
 * does this with two swizzled adds and two vpermf32x4 adds, and the pairing is what makes both agree. */
static inline float q_tree16(const float a[16]) {
    float v[16];
    memcpy(v, a, sizeof v);
    for (int bit = 1; bit < 16; bit <<= 1) {
        float w[16];
        for (int i = 0; i < 16; i++) w[i] = v[i] + v[i ^ bit];
        memcpy(v, w, sizeof v);
    }
    return v[0];
}

/* sum of x[i]^2 */
static inline float q_sum_sq(const float *x, int n) {
    float acc[16] = {0};
    for (int i = 0; i < n; i++) acc[i % 16] = Q_FMA(x[i], x[i], acc[i % 16]);
    return q_tree16(acc);
}

static inline float q_sum(const float *x, int n) {
    float acc[16] = {0};
    for (int i = 0; i < n; i++) acc[i % 16] = acc[i % 16] + x[i];
    return q_tree16(acc);
}

/* sum of a[i] * b[i] */
static inline float q_dot(const float *a, const float *b, int n) {
    float acc[16] = {0};
    for (int i = 0; i < n; i++) acc[i % 16] = Q_FMA(a[i], b[i], acc[i % 16]);
    return q_tree16(acc);
}

/* The scale of an RMS norm: 1 / sqrt(mean(x^2) + eps), n a power of two. */
static inline float q_rms_scale(const float *x, int n, float one_over_n, float eps) {
    return q_rsqrt(q_sum_sq(x, n) * one_over_n + eps);
}

/* softmax in place; every value relative to the largest */
static inline void q_softmax(float *x, int n) {
    float top = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > top) top = x[i];
    for (int i = 0; i < n; i++) x[i] = q_exp(x[i] - top);
    float scale = q_rcp(q_sum(x, n));
    for (int i = 0; i < n; i++) x[i] = x[i] * scale;
}

#endif
