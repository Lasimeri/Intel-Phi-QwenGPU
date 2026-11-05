/* Seen only by clangd, through compile_commands.json, never by a build. The system clangd has no -march=knc,
 * so the card's VPU files are checked against AVX-512F, and these lines stand in for the few KNC-only names.
 * Their meaning does not matter to a type check; the real build is scripts/q35_build.sh with the KNC clang. */
#ifndef CARD_CLANGD_KNC_H
#define CARD_CLANGD_KNC_H

#include <immintrin.h>

#define _MM_SWIZ_REG_CDAB 1
#define _MM_SWIZ_REG_BADC 2
#define _MM_UPCONV_PS_NONE 0
#define _MM_BROADCAST_1X16 0
#define _mm512_swizzle_ps(v, s) (v)
#define _mm512_extload_ps(p, c, b, h) _mm512_set1_ps(*(const float *)(p))
#define __builtin_ia32_knc_vmovaps_zz(b, k, a, s) ((void)(b), (void)(k), (__v16sf)(a))
#define __builtin_ia32_knc_vmovdqa32_zz(b, k, a, s) ((void)(b), (void)(k), (__v16si)(a))
#define __builtin_ia32_knc_vpermf32x4_zzi(b, k, a, i) ((void)(b), (void)(k), (__v16sf)(a))
#define __builtin_ia32_knc_vprefetch0_m(p) ((void)(p))

#endif
