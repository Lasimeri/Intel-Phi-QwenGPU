/* qgpu's additions to OpenCL for the embedded GPU (libqgpucl):
 *   qgpuCreateBufferAt  a buffer over an existing word range of the device's M, so a host program can address the
 *                       2B's map regions (src/guest/q2b_map.h) when the device is the 2B's image (QGPU_IMAGE)
 *   qgpuStats           the machine's counters since the device came up */
#ifndef QGPUCL_EXT_H
#define QGPUCL_EXT_H

#include <CL/cl.h>

#ifdef __cplusplus
extern "C" {
#endif

cl_mem qgpuCreateBufferAt(cl_context context, cl_uint word, size_t bytes, cl_int *errcode_ret);
void qgpuStats(unsigned long long *records, unsigned long long *trace_tokens, unsigned long long *launches);

#ifdef __cplusplus
}
#endif

#endif
