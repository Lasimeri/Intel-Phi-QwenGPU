# OpenCL on the embedded GPU

Two layers: the OpenCL C front-end (kernels written in OpenCL C, compiled to this machine's blocks) and libqgpucl, an
OpenCL 1.2 subset whose device is the machine on the host's reference (`src/gpu/sm_ref.c`), later the card.

## OpenCL C -> the machine (scripts/cl_build.sh)
`clang -x cl -cl-std=CL2.0 -cl-no-stdinc -target nvptx64-nvidia-cuda -O2 -Werror -Wall -Wextra -I src/cl -I src/guest
-S -o out/cl/NAME.ptx NAME.cl`, then `src/gpu/lower.cc` as for CUDA C. The default OpenCL headers are not used: the
device header `src/cl/qgpucl.h` is the implementation's own (as every OpenCL implementation ships one):
- scalar types `uint uchar ushort ulong` and the stdint names (OpenCL C has no `<stdint.h>`; `sm.h` guards its include);
- work items over the machine's special registers: `get_local_id get_group_id get_local_size get_num_groups
  get_global_id get_global_size` (one dimension; a work-group = a thread block of up to 8 warps, a warp = 32 items);
  `barrier()` = `__nvvm_bar_sync(0)` (the only barrier builtin clang 23 accepts in OpenCL mode);
- the machine's instructions as functions, under the same names as the CUDA shim `src/guest/qgpu.h` so kernel source is
  shared: `__q8dot __q8dota __sdot __sup __scfg __wsum __wmax __udiv __usqrt __mulsh __mulshr __clz32 __umulhi
  __ldm __stvm __lda8 __lda32 __lds __ldc __ldw8 __ldw32 __stva8 __stva32 __stvs __stvc`. The asm operand order is the
  lowerer's: q8dot/q8dota/sdot take the accumulator FIRST; udiv is (lo, d, hi).
clang 23 emits every `__kernel` as a stub `.entry NAME` calling a noinline `.func __clang_ocl_kern_imp_NAME`; no flag
removes it; `src/gpu/ptx.cc` parses the function definitions, skips the forward declarations and the call sequence's
`{ } .param` lines, and gives the kernel its function's body with the parameter names rewritten.

Proof: `tests/cl/probe_cl.cl` + `tests/cl/probe_cl.sh` (the 12 probe kernels in OpenCL C, 15 cases at two ROM bases, 0
failures) and `src/guest/q2b_kernels.cl` (the 2B's 29 kernels; the image built from it gives the same probe results as
the CUDA-built one: 532 ROM blocks against 526, 2929 instructions).

## libqgpucl (src/cl/qgpucl.cc, standard <CL/cl.h>, CL_TARGET_OPENCL_VERSION 120)
Direct-linked (`out/host/libqgpucl.a`, link with `-lstdc++`); no ICD yet. One platform "Intel-Phi-QwenGPU", one GPU
device "SIMT GPU in Qwen3.8-35B (reference machine)". The device is a bare machine (M of QGPU_M_WORDS words, default
4,194,304 = 16 MB; 4096 A blocks; 65536 S elements; no W), or, with `QGPU_IMAGE=DIR`, the 2B's image (its banks and
its program present; the runtime's descriptors go after the image's own).

Implemented: clGetPlatformIDs/Info, clGetDeviceIDs/Info, clCreateContext(FromType), clCreateCommandQueue,
clCreateBuffer (M words from a bump allocator; COPY/USE_HOST_PTR copy in), clEnqueueRead/WriteBuffer (run what is
pending, then copy), clCreateProgramWithSource (built by the system clang with `-include qgpucl.h`; the build log is
clang's stderr), clCreateProgramWithBinary (a binary is PTX text), clBuildProgram, clGetProgramBuildInfo/Info,
clCreateKernel, clSetKernelArg, clGetKernelWorkGroupInfo, clEnqueueNDRangeKernel, clFinish/Flush, clWaitForEvents,
clGetEventInfo, clEnqueueMarker/Barrier, retain/release for every object. Extensions (`src/cl/qgpucl_ext.h`):
`qgpuCreateBufferAt(ctx, word, bytes)` wraps an existing M range (the 2B's map regions); `qgpuStats(records, trace
tokens, launches)`.

How a launch becomes machine work: scalar arguments are runtime args in the launch's 16-word args block (u14);
pointer arguments are baked into the lowered instance until a later launch of the same kernel brings a different
pointer for a parameter, which turns that parameter into a runtime arg and lowers the kernel once more (so the 2B's
per-layer pointers do not multiply instances: 39 instances for 982 launches a position). Each distinct (kernel, baked
pointers, grid) is one ROM instance; a launch is one descriptor in the pending list; clFinish, a blocking transfer or
clWaitForEvents write the pending descriptors into M's queue, append HALT and run the machine.

Limits of this cut: one work dimension, no offsets; local size 32..256 in multiples of 32; at most 8192 work-items a
launch (256 warps); scalars of 4 bytes; a device-wide 4096-word `__local` scratch (shared_base), no per-kernel local
allocation; buffers are never freed; a PTX the parser rejects exits the process (ptx.cc die()); no ICD.

## The 2B through the API (tests/cl/q2b_cl.c)
`q2b_build` writes `program.txt`: the whole program, one descriptor a line: `launch NAME GX BX c:V p:WORD a:K q:K ... |
ARGS` (c baked scalar, p baked pointer, a scalar from the args block, q pointer from the args block), `branch WORD
TARGET`, `emit WORD`, `halt`. The replayer interprets it literally with OpenCL calls: a launch is clSetKernelArg
(buffers wrapped at the map's words) + clEnqueueNDRangeKernel; a branch reads its word (clEnqueueReadBuffer runs the
pending launches) and jumps; an emit reads the token; the job words and the prompt go in through clEnqueueWriteBuffer.
Measured (QGPU_IMAGE=out/gpucl, 19 prompt tokens, 8 generated): 8 of 8 tokens = the reference's (57044 16563 54166 11
198 18978 10583 2404), 24,818 launches, 17,612,035 records, 1,119,514,453 trace tokens, 22.4 s wall on the host.
