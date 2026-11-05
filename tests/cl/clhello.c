/* The embedded GPU through OpenCL, as any OpenCL host program does it: platform, device, context, queue, a program
 * built from OpenCL C source, buffers, kernel arguments, an NDRange launch, a read back. Links libqgpucl. Host.
 * Expected: "y: 10 13 16 19" (3 x + 7 over 1 2 3 4). Exit 1 on any difference. */
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <stdio.h>
#include <string.h>

static const char *SRC =
    "__kernel void affine(__global const int *x, __global int *y, int a, int b) {\n"
    "    int i = get_global_id(0);\n"
    "    y[i] = a * x[i] + b;\n"
    "}\n";

#define CHECK(err, what)                                                      \
    do {                                                                      \
        if ((err) != CL_SUCCESS) {                                            \
            fprintf(stderr, "clhello: %s failed: %d\n", what, (int)(err));   \
            return 1;                                                         \
        }                                                                     \
    } while (0)

int main(void) {
    cl_int err;
    cl_platform_id platform;
    cl_uint n;
    CHECK(clGetPlatformIDs(1, &platform, &n), "clGetPlatformIDs");
    char name[256], vendor[256], version[256];
    CHECK(clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof name, name, NULL), "clGetPlatformInfo");
    cl_device_id device;
    CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &n), "clGetDeviceIDs");
    CHECK(clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof vendor, vendor, NULL), "clGetDeviceInfo name");
    CHECK(clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof version, version, NULL), "clGetDeviceInfo version");
    cl_ulong mem = 0;
    cl_uint cus = 0;
    size_t wg = 0;
    CHECK(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof mem, &mem, NULL), "clGetDeviceInfo mem");
    CHECK(clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof cus, &cus, NULL), "clGetDeviceInfo cus");
    CHECK(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof wg, &wg, NULL), "clGetDeviceInfo wg");
    printf("platform: %s\ndevice: %s, %s, %llu bytes of global memory, %u compute units, work-group up to %zu\n", name, vendor, version, (unsigned long long)mem, cus, wg);

    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK(err, "clCreateContext");
    cl_command_queue q = clCreateCommandQueue(ctx, device, 0, &err);
    CHECK(err, "clCreateCommandQueue");
    cl_program prog = clCreateProgramWithSource(ctx, 1, &SRC, NULL, &err);
    CHECK(err, "clCreateProgramWithSource");
    err = clBuildProgram(prog, 1, &device, "", NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[8192];
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, sizeof log, log, NULL);
        fprintf(stderr, "clhello: build failed (%d):\n%s\n", (int)err, log);
        return 1;
    }
    cl_kernel k = clCreateKernel(prog, "affine", &err);
    CHECK(err, "clCreateKernel");

    int x[32] = {1, 2, 3, 4}, y[32];
    memset(y, 0, sizeof y);
    cl_mem bx = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof x, x, &err);
    CHECK(err, "clCreateBuffer x");
    cl_mem by = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof y, NULL, &err);
    CHECK(err, "clCreateBuffer y");
    int a = 3, b = 7;
    CHECK(clSetKernelArg(k, 0, sizeof(cl_mem), &bx), "clSetKernelArg 0");
    CHECK(clSetKernelArg(k, 1, sizeof(cl_mem), &by), "clSetKernelArg 1");
    CHECK(clSetKernelArg(k, 2, sizeof a, &a), "clSetKernelArg 2");
    CHECK(clSetKernelArg(k, 3, sizeof b, &b), "clSetKernelArg 3");
    size_t global = 32, local = 32;
    CHECK(clEnqueueNDRangeKernel(q, k, 1, NULL, &global, &local, 0, NULL, NULL), "clEnqueueNDRangeKernel");
    CHECK(clEnqueueReadBuffer(q, by, CL_TRUE, 0, sizeof y, y, 0, NULL, NULL), "clEnqueueReadBuffer");
    CHECK(clFinish(q), "clFinish");
    printf("y: %d %d %d %d\n", y[0], y[1], y[2], y[3]);
    const int ok = y[0] == 10 && y[1] == 13 && y[2] == 16 && y[3] == 19;

    clReleaseMemObject(bx);
    clReleaseMemObject(by);
    clReleaseKernel(k);
    clReleaseProgram(prog);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);
    puts(ok ? "host: clhello: the embedded GPU answered through OpenCL" : "host: clhello: WRONG ANSWER");
    return ok ? 0 : 1;
}
