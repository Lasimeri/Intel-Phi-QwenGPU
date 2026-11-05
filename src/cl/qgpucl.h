/* The embedded GPU's OpenCL C device header (docs/machine.md). Included by every .cl instead of clang's default OpenCL
 * header: the work-item functions read the machine's special registers, the barrier is the thread block's `bar`, and
 * the machine's own instructions are functions (inline PTX asm, lowered by src/gpu/lower.cc), under the same names
 * as the CUDA shim src/guest/qgpu.h so kernel source can be shared between the two front-ends.
 *
 * Build: scripts/cl_build.sh (clang -x cl -cl-no-stdinc -target nvptx64-nvidia-cuda, the kernel stubs inlined; see it)
 * The driver lowers no calls, so every function must inline. */
#ifndef QGPUCL_H
#define QGPUCL_H

typedef unsigned int uint;  /* the OpenCL scalar types (compiled with -cl-no-stdinc: this header is the device header) */
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned long ulong;
typedef signed char int8_t;  /* <stdint.h>, which OpenCL C has not: the map and sm.h use these */
typedef unsigned char uint8_t;
typedef short int16_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long int64_t;
typedef unsigned long uint64_t;
typedef int i32;
typedef unsigned int u32;

/* work items: one dimension; a work-group = a thread block of up to 8 warps, a warp = 32 work-items */
#define CLK_LOCAL_MEM_FENCE 1u
#define CLK_GLOBAL_MEM_FENCE 2u
static inline uint get_local_id(uint d) { return d == 0u ? (uint)__nvvm_read_ptx_sreg_tid_x() : 0u; }
static inline uint get_group_id(uint d) { return d == 0u ? (uint)__nvvm_read_ptx_sreg_ctaid_x() : 0u; }
static inline uint get_local_size(uint d) { return d == 0u ? (uint)__nvvm_read_ptx_sreg_ntid_x() : 1u; }
static inline uint get_num_groups(uint d) { return d == 0u ? (uint)__nvvm_read_ptx_sreg_nctaid_x() : 1u; }
static inline uint get_global_id(uint d) { return get_group_id(d) * get_local_size(d) + get_local_id(d); }
static inline uint get_global_size(uint d) { return get_num_groups(d) * get_local_size(d); }
static inline void barrier(uint flags) {
    (void)flags;
    __nvvm_bar_sync(0);
}

/* the matrix unit: rd = acc + u over one Q8 block; the W form (GGUF block wblk against A block ablk) and the A form
 * (both operands A blocks: the KV caches) */
static inline i32 __q8dot(i32 acc, u32 wblk, u32 ablk) { i32 d; __asm__ volatile("q8dot %0, %1, %2, %3;" : "=r"(d) : "r"(acc), "r"(wblk), "r"(ablk) : "memory"); return d; }
static inline i32 __q8dota(i32 acc, u32 xblk, u32 ablk) { i32 d; __asm__ volatile("q8dota %0, %1, %2, %3;" : "=r"(d) : "r"(acc), "r"(xblk), "r"(ablk) : "memory"); return d; }
/* the GDN state unit */
static inline i32 __sdot(i32 acc, u32 addr, i32 k8) { i32 d; __asm__ volatile("sdot %0, %1, %2, %3;" : "=r"(d) : "r"(acc), "r"(addr), "r"(k8) : "memory"); return d; }
/* S[addr] updated; returns max(acc, |the new S[addr]|): the largest magnitude a lane wrote, carried through acc */
static inline i32 __sup(i32 acc, u32 addr, i32 delta, i32 kreal) { __asm__ volatile("sup %0, %1, %2, %3;" : "+r"(acc) : "r"(addr), "r"(delta), "r"(kreal) : "memory"); return acc; }
static inline void __scfg(i32 cfg) { __asm__ volatile("scfg %0;" : : "r"(cfg) : "memory"); }
/* warp reductions, uniform results */
static inline i32 __wsum(i32 v) { i32 d; __asm__ volatile("wsum %0, %1;" : "=r"(d) : "r"(v)); return d; }
static inline i32 __wmax(i32 v) { i32 d; __asm__ volatile("wmax %0, %1;" : "=r"(d) : "r"(v)); return d; }
/* the scalar unit: (hi:lo) / d and the integer root of (hi:lo); uniform operands */
static inline u32 __udiv(u32 lo, u32 hi, u32 d) { u32 q; __asm__ volatile("udiv %0, %1, %2, %3;" : "=r"(q) : "r"(lo), "r"(d), "r"(hi)); return q; }
static inline u32 __usqrt(u32 lo, u32 hi) { u32 q; __asm__ volatile("usqrt %0, %1, %2;" : "=r"(q) : "r"(lo), "r"(hi)); return q; }
/* (a * b + 2^(s-1)) >> s on the signed 64-bit product: s a constant 0..62, or (__mulshr) a register, signed, negative
 * = left */
#define __mulsh(a, b, s) __mulsh_fn((a), (b), (s))
static inline i32 __mulsh_fn(i32 a, i32 b, const i32 s) { i32 d; __asm__ volatile("mulsh %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "n"(s)); return d; }
static inline i32 __mulshr(i32 a, i32 b, i32 s) { i32 d; __asm__ volatile("mulsh %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }
static inline i32 __clz32(u32 x) { return (i32)__builtin_clz(x); }
static inline u32 __umulhi(u32 a, u32 b) { return __nvvm_mulhi_ui(a, b); }
/* the banks by element address: M (words), A8/A32 (Q8 block elements: block * 64 + j, j = 32 the scale), S (int16),
 * C (int8), W8/W32 (the GGUF's own Q8_0 blocks: block * 64 + j, j = 0 the f16 scale as Q.24 through W32) */
static inline i32 __ldm(u32 a) { i32 d; __asm__ volatile("ldm %0, %1;" : "=r"(d) : "r"(a) : "memory"); return d; }
static inline void __stvm(u32 a, i32 v) { __asm__ volatile("stvm %0, %1;" : : "r"(a), "r"(v) : "memory"); }
static inline i32 __lda8(u32 a) { i32 d; __asm__ volatile("lda8 %0, %1;" : "=r"(d) : "r"(a) : "memory"); return d; }
static inline i32 __lda32(u32 a) { i32 d; __asm__ volatile("lda32 %0, %1;" : "=r"(d) : "r"(a) : "memory"); return d; }
static inline i32 __lds(u32 a) { i32 d; __asm__ volatile("lds %0, %1;" : "=r"(d) : "r"(a) : "memory"); return d; }
static inline i32 __ldc(u32 a) { i32 d; __asm__ volatile("ldc %0, %1;" : "=r"(d) : "r"(a) : "memory"); return d; }
static inline i32 __ldw8(u32 a) { i32 d; __asm__ volatile("ldw8 %0, %1;" : "=r"(d) : "r"(a) : "memory"); return d; }
static inline i32 __ldw32(u32 a) { i32 d; __asm__ volatile("ldw32 %0, %1;" : "=r"(d) : "r"(a) : "memory"); return d; }
static inline void __stva8(u32 a, i32 v) { __asm__ volatile("stva8 %0, %1;" : : "r"(a), "r"(v) : "memory"); }
static inline void __stva32(u32 a, i32 v) { __asm__ volatile("stva32 %0, %1;" : : "r"(a), "r"(v) : "memory"); }
static inline void __stvs(u32 a, i32 v) { __asm__ volatile("stvs %0, %1;" : : "r"(a), "r"(v) : "memory"); }
static inline void __stvc(u32 a, i32 v) { __asm__ volatile("stvc %0, %1;" : : "r"(a), "r"(v) : "memory"); }

#endif
