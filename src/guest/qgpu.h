// The guest's CUDA: what a kernel of the 2B may use on the machine of docs/machine.md. Compiled by clang's NVPTX
// backend without the CUDA SDK (scripts/guest_build.sh); the driver (src/gpu/lower.cc) lowers the PTX. The machine's
// own instructions come through as inline PTX asm with plain 32-bit register operands.
//
// Memory: `int *` pointers are words of M (the only pointer-addressed bank: plain loads and stores). The other banks
// are reached by the intrinsics below with element addresses: A (quantized blocks: block * 64 + j, j < 32 an int8,
// j = 32 the Q.24 scale), S (int16), C (int8), W (the weights: block * 64 + j; read-only).
#pragma once
#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __shared__ __attribute__((shared))
#define __forceinline__ __inline__ __attribute__((always_inline))
#include <__clang_cuda_builtin_vars.h>

typedef int i32;
typedef unsigned u32;

// one Q8 block: acc + ((dot(W[wblk], A[ablk]) * mw + 2^15 >> 16) * xs + 2^15 >> 16)
static __device__ __forceinline__ i32 __q8dot(i32 acc, u32 wblk, u32 ablk) {
    i32 d;
    asm volatile("q8dot %0, %1, %2, %3;" : "=r"(d) : "r"(acc), "r"(wblk), "r"(ablk) : "memory");
    return d;
}
// the same with the lane-side block in A (its Q.24 scale): the attention's K and V^T caches
static __device__ __forceinline__ i32 __q8dota(i32 acc, u32 ablk_lane, u32 ablk) {
    i32 d;
    asm volatile("q8dota %0, %1, %2, %3;" : "=r"(d) : "r"(acc), "r"(ablk_lane), "r"(ablk) : "memory");
    return d;
}
// acc + S[addr] * (k8 as int8)
static __device__ __forceinline__ i32 __sdot(i32 acc, u32 addr, i32 k8) {
    i32 d;
    asm volatile("sdot %0, %1, %2, %3;" : "=r"(d) : "r"(acc), "r"(addr), "r"(k8) : "memory");
    return d;
}
// S[addr] = clamp(shr_round(shr_round(S[addr] * e, sh1) + shr_round(kreal * delta, 16), sh2)); returns max(acc, |new S[addr]|)
static __device__ __forceinline__ i32 __sup(i32 acc, u32 addr, i32 delta, i32 kreal) {
    asm volatile("sup %0, %1, %2, %3;" : "+r"(acc) : "r"(addr), "r"(delta), "r"(kreal) : "memory");
    return acc;
}
// e (Q.15) in bits 0..15, sh1 + 32 in 16..22, sh2 + 32 in 23..29
static __device__ __forceinline__ void __scfg(i32 cfg) { asm volatile("scfg %0;" : : "r"(cfg) : "memory"); }
// reductions over the 32 lanes of the warp (lanes past the kernel's early exit do not count): uniform results
static __device__ __forceinline__ i32 __wsum(i32 x) {
    i32 d;
    asm volatile("wsum %0, %1;" : "=r"(d) : "r"(x));
    return d;
}
static __device__ __forceinline__ i32 __wmax(i32 x) {
    i32 d;
    asm volatile("wmax %0, %1;" : "=r"(d) : "r"(x));
    return d;
}
// uniform only: (hi:lo) / d, and isqrt(hi:lo)
static __device__ __forceinline__ u32 __udiv(u32 lo, u32 hi, u32 d) {
    u32 q;
    asm volatile("udiv %0, %1, %2, %3;" : "=r"(q) : "r"(lo), "r"(d), "r"(hi));
    return q;
}
static __device__ __forceinline__ u32 __usqrt(u32 lo, u32 hi) {
    u32 q;
    asm volatile("usqrt %0, %1, %2;" : "=r"(q) : "r"(lo), "r"(hi));
    return q;
}
// (a * b + 2^(s-1)) >> s, the signed 64-bit product; s a constant 0..62, or (__mulshr) a uniform value, signed:
// a negative s is a left shift
#define __mulsh(a, b, s) __mulsh_fn((a), (b), (s))
static __device__ __forceinline__ i32 __mulsh_fn(i32 a, i32 b, const i32 s) {
    i32 d;
    asm volatile("mulsh %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "n"(s));
    return d;
}
static __device__ __forceinline__ i32 __mulshr(i32 a, i32 b, i32 s) {
    i32 d;
    asm volatile("mulsh %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
    return d;
}
// the number of leading zero bits (32 for 0)
static __device__ __forceinline__ i32 __clz32(u32 x) { return __builtin_clz(x); }
// the high word of the unsigned 32 x 32 product
static __device__ __forceinline__ u32 __umulhi(u32 a, u32 b) { return __nvvm_mulhi_ui(a, b); }
// the banks. __ldm / __stvm reach M by word address: a load the compiler must not move under a branch (the machine
// has no divergence), a store that the queue's scalars take
static __device__ __forceinline__ i32 __ldm(u32 a) { i32 d; asm volatile("ldm %0, %1;" : "=r"(d) : "r"(a) : "memory"); return d; }
static __device__ __forceinline__ void __stvm(u32 a, i32 v) { asm volatile("stvm %0, %1;" : : "r"(a), "r"(v) : "memory"); }
static __device__ __forceinline__ i32 __lda8(u32 a) { i32 d; asm volatile("lda8 %0, %1;" : "=r"(d) : "r"(a) : "memory"); return d; }
static __device__ __forceinline__ i32 __lda32(u32 a) { i32 d; asm volatile("lda32 %0, %1;" : "=r"(d) : "r"(a) : "memory"); return d; }
static __device__ __forceinline__ i32 __lds(u32 a) { i32 d; asm volatile("lds %0, %1;" : "=r"(d) : "r"(a) : "memory"); return d; }
static __device__ __forceinline__ i32 __ldc(u32 a) { i32 d; asm volatile("ldc %0, %1;" : "=r"(d) : "r"(a) : "memory"); return d; }
static __device__ __forceinline__ i32 __ldw8(u32 a) { i32 d; asm volatile("ldw8 %0, %1;" : "=r"(d) : "r"(a) : "memory"); return d; }
static __device__ __forceinline__ i32 __ldw32(u32 a) { i32 d; asm volatile("ldw32 %0, %1;" : "=r"(d) : "r"(a) : "memory"); return d; }
static __device__ __forceinline__ void __stva8(u32 a, i32 v) { asm volatile("stva8 %0, %1;" : : "r"(a), "r"(v) : "memory"); }
static __device__ __forceinline__ void __stva32(u32 a, i32 v) { asm volatile("stva32 %0, %1;" : : "r"(a), "r"(v) : "memory"); }
static __device__ __forceinline__ void __stvs(u32 a, i32 v) { asm volatile("stvs %0, %1;" : : "r"(a), "r"(v) : "memory"); }
static __device__ __forceinline__ void __stvc(u32 a, i32 v) { asm volatile("stvc %0, %1;" : : "r"(a), "r"(v) : "memory"); }

// q2b_fix's rounding, on 32-bit values through mulsh
#define SHR_ROUND(v, s) __mulsh((v), 1, (s))
