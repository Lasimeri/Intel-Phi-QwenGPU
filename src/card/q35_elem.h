/* The card's vector forms of src/text/elem.h, operation for operation: same constants, same order, same
 * rounding, so every lane equals the scalar definition bit for bit (gate: tests/card/elem_gate_*). Every
 * float here is in a zmm register. KNC clang only. */
#ifndef CARD_Q35_ELEM_H
#define CARD_Q35_ELEM_H

#include <immintrin.h>
#include <stdint.h>

#define V_SET1(f) _mm512_set1_ps(f)
#define V_FMA(a, b, c) _mm512_fmadd_ps((a), (b), (c))

static inline __m512i v_i(__m512 v) { return _mm512_castps_si512(v); }
static inline __m512 v_f(__m512i v) { return _mm512_castsi512_ps(v); }
static inline __m512 v_neg(__m512 v) { return v_f(_mm512_xor_si512(v_i(v), _mm512_set1_epi32((int)0x80000000u))); }
/* lanes on in k from a, the others from b */
static inline __m512 v_blend(__mmask16 k, __m512 a, __m512 b) { return (__m512)__builtin_ia32_knc_vmovaps_zz((__v16sf)b, k, (__v16sf)a, 0); }

static inline __m512 v_exp(__m512 x) {
    const __mmask16 low = _mm512_cmplt_ps_mask(x, V_SET1(-87.0f));
    x = v_blend(_mm512_cmplt_ps_mask(V_SET1(88.0f), x), V_SET1(88.0f), x);
    __m512 t = _mm512_add_ps(_mm512_mul_ps(x, V_SET1(1.44269504088896341f)), V_SET1(12582912.0f));
    __m512i ni = _mm512_sub_epi32(v_i(t), _mm512_set1_epi32(0x4B400000));
    __m512 n = _mm512_sub_ps(t, V_SET1(12582912.0f));
    __m512 mn = v_neg(n);
    __m512 r = V_FMA(mn, V_SET1(0.693359375f), x);
    r = V_FMA(mn, V_SET1(-2.12194440054690583e-4f), r);
    __m512 p = V_SET1(1.98412698e-4f);
    p = V_FMA(p, r, V_SET1(1.38888889e-3f));
    p = V_FMA(p, r, V_SET1(8.33333333e-3f));
    p = V_FMA(p, r, V_SET1(4.16666667e-2f));
    p = V_FMA(p, r, V_SET1(1.66666667e-1f));
    p = V_FMA(p, r, V_SET1(0.5f));
    p = V_FMA(p, r, V_SET1(1.0f));
    p = V_FMA(p, r, V_SET1(1.0f));
    __m512 out = v_f(_mm512_add_epi32(v_i(p), _mm512_slli_epi32(ni, 23)));
    return v_blend(low, _mm512_setzero_ps(), out);
}

static inline __m512 v_rcp(__m512 a) {
    __m512 y = v_f(_mm512_sub_epi32(_mm512_set1_epi32(0x7EF311C7), v_i(a)));
    const __m512 ma = v_neg(a), one = V_SET1(1.0f);
    for (int i = 0; i < 4; i++) {
        __m512 e = V_FMA(ma, y, one);
        y = V_FMA(y, e, y);
    }
    return y;
}

static inline __m512 v_rsqrt(__m512 a) {
    __m512 y = v_f(_mm512_sub_epi32(_mm512_set1_epi32(0x5F375A86), _mm512_srli_epi32(v_i(a), 1)));
    const __m512 one = V_SET1(1.0f), half = V_SET1(0.5f);
    for (int i = 0; i < 4; i++) {
        __m512 t = _mm512_mul_ps(a, y);
        __m512 e = V_FMA(v_neg(t), y, one);
        __m512 h = _mm512_mul_ps(y, half);
        y = V_FMA(h, e, y);
    }
    return y;
}

static inline __m512 v_log(__m512 a) {
    __m512i u = v_i(a);
    __m512i e = _mm512_sub_epi32(_mm512_srli_epi32(u, 23), _mm512_set1_epi32(127));
    __m512 m = v_f(_mm512_or_si512(_mm512_and_si512(u, _mm512_set1_epi32(0x007FFFFF)), _mm512_set1_epi32(0x3F800000)));
    const __mmask16 big = _mm512_cmplt_ps_mask(V_SET1(1.41421354f), m);
    m = v_blend(big, _mm512_mul_ps(m, V_SET1(0.5f)), m);
    e = (__m512i)__builtin_ia32_knc_vmovdqa32_zz((__v16si)e, big, (__v16si)_mm512_add_epi32(e, _mm512_set1_epi32(1)), 0);
    const __m512 one = V_SET1(1.0f);
    __m512 s = _mm512_mul_ps(_mm512_sub_ps(m, one), v_rcp(_mm512_add_ps(m, one)));
    __m512 z = _mm512_mul_ps(s, s);
    __m512 p = V_SET1(0.111111111f);
    p = V_FMA(p, z, V_SET1(0.142857143f));
    p = V_FMA(p, z, V_SET1(0.2f));
    p = V_FMA(p, z, V_SET1(0.333333333f));
    p = V_FMA(p, z, one);
    __m512 two_s = _mm512_mul_ps(s, V_SET1(2.0f));
    __m512 ef = _mm512_sub_ps(v_f(_mm512_add_epi32(_mm512_set1_epi32(0x4B400000), e)), V_SET1(12582912.0f));
    return V_FMA(ef, V_SET1(0.693147182f), _mm512_mul_ps(two_s, p));
}

static inline __m512 v_sigmoid(__m512 x) { return v_rcp(_mm512_add_ps(V_SET1(1.0f), v_exp(v_neg(x)))); }
static inline __m512 v_silu(__m512 x) { return _mm512_mul_ps(x, v_sigmoid(x)); }
static inline __m512 v_softplus(__m512 x) {
    __m512 small = v_log(_mm512_add_ps(V_SET1(1.0f), v_exp(x)));
    return v_blend(_mm512_cmplt_ps_mask(V_SET1(20.0f), x), x, small);
}

/* Every lane becomes the sum: pairs i^1, i^2, i^4, i^8, as q_tree16. */
static inline __m512 v_tree16(__m512 v) {
    v = _mm512_add_ps(v, _mm512_swizzle_ps(v, _MM_SWIZ_REG_CDAB));
    v = _mm512_add_ps(v, _mm512_swizzle_ps(v, _MM_SWIZ_REG_BADC));
    v = _mm512_add_ps(v, (__m512)__builtin_ia32_knc_vpermf32x4_zzi((__v16sf)v, 0xFFFF, (__v16sf)v, 0xB1));
    v = _mm512_add_ps(v, (__m512)__builtin_ia32_knc_vpermf32x4_zzi((__v16sf)v, 0xFFFF, (__v16sf)v, 0x4E));
    return v;
}

#endif
