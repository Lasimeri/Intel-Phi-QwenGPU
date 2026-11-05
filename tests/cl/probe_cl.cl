// The driver's probe kernels in OpenCL C (tests/gpu/probe.cu in CUDA C, the same answers: tests/cl/probe_cl.sh): the
// pieces the 2B's kernels are built from, through clang's OpenCL front-end and the device header src/cl/qgpucl.h.
#include "qgpucl.h"

// y[i] = 3 x[i] + 7 with the row held in u14's args: n at args[0]
__kernel void affine(__global const int *x, __global int *y, int n) {
    int i = get_global_id(0);
    if (i < n) y[i] = 3 * x[i] + 7;
}

// local memory and barriers: two warps per work-group; warp 1 reads what warp 0 wrote
__kernel void barsum(__global const int *x, __global int *y, int n) {
    __local int s[64];
    uint t = get_local_id(0);
    uint i = get_global_id(0);
    s[t] = x[i];  /* no per-lane condition around a load: the driver has no divergence; the launch is exactly n items */
    barrier(CLK_LOCAL_MEM_FENCE);
    y[i] = s[63 - t] + s[t] + n;
}

// the warp reductions and a cross-warp sum through M: out[0] = sum of x, out[1] = max of x
__kernel void reduce(__global const int *x, __global int *out, int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    int v = x[i];
    int s = __wsum(v), m = __wmax(v);
    out[0] += s;
    out[1] = out[1] > m ? out[1] : m;
}

// a scalar kernel: one warp, uniform code: out[0] = (hi:lo) / d, out[1] = isqrt(hi:lo), out[2] = mulsh(a, b, 16)
__kernel void scalars(__global const int *in, __global int *out) {
    out[0] = (int)__udiv((u32)in[0], (u32)in[1], (u32)in[2]);
    out[1] = (int)__usqrt((u32)in[0], (u32)in[1]);
    out[2] = __mulsh(in[3], in[4], 16);
}

// a uniform loop with a runtime bound: y[i] = sum over k < m of x[k * n + i]
__kernel void loopsum(__global const int *x, __global int *y, int n, int m) {
    int i = get_global_id(0);
    if (i >= n) return;
    int acc = 0;
    __attribute__((opencl_unroll_hint(1))) for (int k = 0; k < m; k++) acc += x[k * n + i];
    y[i] = acc;
}

// the register shift and clz, uniform: out[0] = mulshr(100, 1, 5) = 3, out[1] = mulshr(-100, 1, 5) = -3,
// out[2] = mulshr(131072, 1, 16) = 2, out[3] = mulshr(5, 1, -2) = 20, out[4] = shr_round(-100, 4) = -6, out[5] = 32 - clz(x[2]) - 22 = -4
__kernel void shifts(__global const int *in, __global int *out) {
    int s = in[2] >> 15;  /* 4 */
    out[0] = __mulshr(in[0], 1, s + 1);
    out[1] = __mulshr(in[1], 1, s + 1);
    out[2] = __mulshr(in[2], 1, 4 * s);
    out[3] = __mulshr(s + 1, 1, -2);
    out[4] = __mulshr(in[1], 1, s);
    out[5] = 32 - __clz32((u32)in[2]) - 22;
}

// the banks: A block 1 gets lane values and a scale; S, C and W reads; q8dot of W block (lane) against A block 0
__kernel void banks(__global int *out, int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    __stva8(64 + i, i - 3);
    __stva32(64 + 32, 1 << 24);  /* every lane writes the same scale: no per-lane branch */
    __stvs(i, 1000 * i);
    __stvc(i, 100 - i);
    out[i] = __lds(i) + __ldc(i) + __ldw8(i * 64 + 5) + __q8dot(0, i, 0) + __lda8(64 + i);
    out[32 + i] = __ldw32(i * 64);
}

// shifts by a UNIFORM amount (a runtime arg): the amount is a uniform register read through rc. Every lane writes the
// same four words (no select chain: clang makes a jump table of one).
__kernel void ushifts(__global const int *x, __global int *out, int k) {
    out[0] = x[0] << k;
    out[1] = (int)((uint)x[1] >> k);
    out[2] = x[2] >> k;
    out[3] = __mulshr(x[3], 3, k);
}

// the GDN state instructions on a zero S bank: scfg (e = 1.0 in Q.15, sh1 0, sh2 8), two sups on element i with
// delta 2^20 and kreal 2^4 (val 256 -> S 1 -> val 33024 -> S 129), the largest |S| over the lanes 129, sdot with q = 2: 258
__kernel void gdnunit(__global int *out) {
    u32 i = get_local_id(0);
    __scfg(32768 | (32 << 16) | (40 << 23));
    i32 v = __sup(0, i, 1 << 20, 1 << 4);
    barrier(CLK_GLOBAL_MEM_FENCE);  /* a sup's write lands after its lane's pass: the second update needs a new block */
    v = __sup(v, i, 1 << 20, 1 << 4);
    i32 sm = __wmax(v);
    i32 d = __sdot(0, i, 2);
    out[i] = sm * 1000 + d;
}

// the GDN head's exponent from its 64-bit bound, as the 2B's gdn_head computes it: in = smax, e, se, kscale, maxdelta;
// out = nse, bhi, blo, t1
__kernel void gdnbound(__global const int *in, __global int *out) {
    i32 smax = in[0], e = in[1], se = in[2], kscale = in[3];
    u32 maxdelta = (u32)in[4];
    u32 t1 = (u32)__mulshr(smax, e, se - 17);
    u32 a127 = (u32)(127 * kscale);
    u32 plo = a127 * maxdelta, phi = __umulhi(a127, maxdelta);
    u32 plo2 = plo + 32768u;
    u32 phi2 = phi + (plo2 < plo ? 1u : 0u);
    u32 t2lo = (phi2 << 16) | (plo2 >> 16), t2hi = phi2 >> 16;
    u32 blo = t1 + t2lo;
    u32 bhi = t2hi + (blo < t1 ? 1u : 0u);
    i32 lhi = 64 - __clz32(bhi), llo = 32 - __clz32(blo);
    i32 bitlen = bhi ? lhi : (blo ? llo : 0);
    i32 nse = (bhi | blo) ? 47 - bitlen : se;
    out[0] = nse;
    out[1] = (i32)bhi;
    out[2] = (i32)blo;
    out[3] = (i32)t1;
}

// a work-group of 4 warps folds a maximum through an M word in warp order: the first warp (local id < 32) starts it;
// two work-groups, their words differ. x = -(i + 1): group 0 -> -1, group 1 -> -129
__kernel void blockmax(__global const int *x, __global int *out) {
    int i = get_global_id(0);
    int w = 41056 + get_group_id(0);
    i32 m = __wmax(x[i]);
    i32 cur = __ldm((u32)w);
    __stvm((u32)w, (get_local_id(0) >> 5) == 0 ? m : (cur > m ? cur : m));
    barrier(CLK_GLOBAL_MEM_FENCE);
    out[i] = __ldm((u32)w);
}

// the largest |S| through the same fold: warp w of one work-group of 4 writes S[lane] = w + 1 by a sup, reduces the new
// values (w + 1), folds the maximum through an M word: every lane reads 4 after the barrier
__kernel void smaxblock(__global int *out) {
    u32 i = get_local_id(0);
    int w = 41056 + get_group_id(0);
    __scfg(32768 | (32 << 16) | (40 << 23));
    i32 v = __sup(0, i, (i32)((i >> 5) + 1) << 20, 1 << 4);
    i32 sm = __wmax(v);
    i32 cur = __ldm((u32)w);
    __stvm((u32)w, (i >> 5) == 0 ? sm : (cur > sm ? cur : sm));
    barrier(CLK_GLOBAL_MEM_FENCE);
    out[i] = __ldm((u32)w);
}
