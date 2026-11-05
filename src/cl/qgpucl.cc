// libqgpucl: an OpenCL 1.2 subset whose device is the embedded SIMT GPU (docs/machine.md) running on the reference
// machine (src/gpu/sm_ref.c) on the host. The standard entry points of <CL/cl.h>, direct-linked (no ICD yet):
//   platform / device       one of each; the device is a bare machine (ROM, M, A, S, no W), or the 2B's image when
//                           QGPU_IMAGE names its directory (then the 2B's banks and program are present)
//   buffers                 M word ranges from a bump allocator (byte sizes rounded up to words; no free yet)
//   programs                OpenCL C built by the system clang with the device header pre-included (the same build
//                           scripts/cl_build.sh does) and parsed by src/gpu/ptx.cc; a binary is a PTX text
//   kernels + NDRange       clSetKernelArg: a cl_mem becomes the kernel's pointer parameter (its M word), a 4-byte
//                           scalar a baked constant; clEnqueueNDRangeKernel lowers the kernel for that grid once
//                           (src/gpu/lower.cc), appends the instance to the ROM and a launch descriptor to the queue
//   command queue           in-order; descriptors are pending until clFinish, a blocking read or write, or
//                           clWaitForEvents, which write them into M's queue, append HALT and run the machine
// Limits of this cut: one work dimension; local sizes 32..256 in multiples of 32; at most 8192 work-items a launch
// (256 warps); scalars of 4 bytes; no images, no events with status other than complete, no __local allocation
// beyond the device's 4096-word scratch (shared_base).
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "../gpu/image.h"
#include "../gpu/lower.h"
#include "../gpu/ptx.h"
#include "../gpu/sm_ref.h"

#ifndef QGPU_ROOT
#define QGPU_ROOT "."
#endif

namespace {

struct Instance {
    uint32_t base = 0, warps = 0;
    int lwpb = 0;
};

struct Device {
    Image im;
    SmMachine *mc = nullptr;
    uint32_t free_word = 0, shared_base = 0, queue_base = 0, args_base = 0;  // args: 8 words per pending launch (u14)
    std::vector<std::pair<uint32_t, uint32_t>> pending;  // launch descriptors not yet run
    std::map<std::string, Instance> instances;
    struct Spec {  // per kernel: which pointer parameters have varied between launches (then runtime args)
        std::vector<bool> varying;
        std::vector<uint32_t> last_word;
        bool seen = false;
    };
    std::map<const void *, Spec> specs;
    uint64_t launches = 0, flushes = 0;
    std::string name;

    static Device &get() {
        static Device *d = nullptr;
        if (!d) {
            d = new Device;
            const char *dir = std::getenv("QGPU_IMAGE");
            if (dir && *dir) {
                image_load(d->im, dir);
                d->queue_base = d->im.descriptors;
                d->name = "SIMT GPU in Qwen3.8-35B (reference machine, the 2B's image)";
            } else {
                const char *mw = std::getenv("QGPU_M_WORDS");
                uint32_t m_words = mw ? (uint32_t)std::strtoul(mw, nullptr, 0) : (1u << 22);
                image_bare(d->im, m_words, 4096, 65536);
                d->name = "SIMT GPU in Qwen3.8-35B (reference machine)";
            }
            d->free_word = (uint32_t)d->im.m.size() > SM_M_USER ? (uint32_t)d->im.m.size() : SM_M_USER;
            if (!dir) d->free_word = SM_M_USER;
            d->shared_base = d->alloc(4096);
            d->args_base = d->alloc((1u << SM_QBITS) * 16);
            d->mc = (SmMachine *)std::calloc(1, sizeof *d->mc);
            sm_init(d->mc, &d->im.img);
        }
        return *d;
    }
    // M words from the top of what is used; M grows as needed (the machine is rebound to the new buffer)
    uint32_t alloc(uint32_t words) {
        uint32_t w = free_word;
        free_word += words;
        if (free_word > im.m.size()) {
            im.m.resize(free_word, 0);
            im.bind();
        }
        return w;
    }
    uint8_t *bytes(uint32_t word) { return (uint8_t *)im.m.data() + (size_t)word * 4; }
    // the pending launches run: the descriptors go into M's queue after the image's own, HALT ends them
    cl_int flush() {
        if (pending.empty()) return CL_SUCCESS;
        if (queue_base + pending.size() + 1 > (1u << SM_QBITS)) return CL_OUT_OF_RESOURCES;
        for (size_t i = 0; i < pending.size(); i++) {
            im.m[SM_M_QUEUE + 2 * (queue_base + i)] = (int32_t)pending[i].first;
            im.m[SM_M_QUEUE + 2 * (queue_base + i) + 1] = (int32_t)pending[i].second;
        }
        im.m[SM_M_QUEUE + 2 * (queue_base + pending.size())] = (int32_t)sm_qword(SM_Q_HALT, 0, 0, 0, 0);
        im.m[SM_M_QUEUE + 2 * (queue_base + pending.size()) + 1] = 0;
        mc->qi = queue_base;
        mc->halted = 0;
        mc->running = 0;
        mc->fault = 0;
        int rc = sm_run(mc, 1ull << 62);
        flushes++;
        pending.clear();
        if (rc != 0) {
            std::fprintf(stderr, "qgpucl: the machine stopped with %d (%s)\n", rc, mc->fault_what);
            return CL_OUT_OF_RESOURCES;
        }
        return CL_SUCCESS;
    }
};

struct Arg {
    enum Kind { NONE, BUFFER, SCALAR } kind = NONE;
    cl_mem mem = nullptr;
    int32_t value = 0;
};

cl_int set_info(size_t size, void *value, size_t *size_ret, const void *src, size_t n) {
    if (size_ret) *size_ret = n;
    if (value) {
        if (size < n) return CL_INVALID_VALUE;
        std::memcpy(value, src, n);
    }
    return CL_SUCCESS;
}
cl_int set_str(size_t size, void *value, size_t *size_ret, const char *s) { return set_info(size, value, size_ret, s, std::strlen(s) + 1); }
template <typename T>
cl_int set_val(size_t size, void *value, size_t *size_ret, T v) {
    return set_info(size, value, size_ret, &v, sizeof(T));
}

std::string read_file(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

}  // namespace

// the objects: a dispatch pointer first, as the ICD loader will want
struct _cl_platform_id {
    void *dispatch = nullptr;
};
struct _cl_device_id {
    void *dispatch = nullptr;
};
struct _cl_context {
    void *dispatch = nullptr;
    int refs = 1;
    std::vector<cl_mem> mems;
};
struct _cl_command_queue {
    void *dispatch = nullptr;
    cl_context ctx = nullptr;
    int refs = 1;
};
struct _cl_mem {
    void *dispatch = nullptr;
    cl_context ctx = nullptr;
    uint32_t word = 0, words = 0;
    size_t bytes = 0;
    int refs = 1;
};
struct _cl_program {
    void *dispatch = nullptr;
    cl_context ctx = nullptr;
    std::string source, log, ptx;
    std::map<std::string, ptx::Kernel> kernels;
    bool built = false;
    int refs = 1;
};
struct _cl_kernel {
    void *dispatch = nullptr;
    cl_program prog = nullptr;
    std::string name;
    const ptx::Kernel *k = nullptr;
    std::vector<Arg> args;
    int refs = 1;
};
struct _cl_event {
    void *dispatch = nullptr;
    int refs = 1;
};

namespace {
_cl_platform_id the_platform;
_cl_device_id the_device;
bool is_mem(cl_context ctx, const void *p) {
    for (cl_mem m : ctx->mems)
        if (m == p) return true;
    return false;
}
}  // namespace

extern "C" {

cl_int clGetPlatformIDs(cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms) {
    if ((num_entries == 0 && platforms) || (!platforms && !num_platforms)) return CL_INVALID_VALUE;
    Device::get();
    if (platforms && num_entries >= 1) platforms[0] = &the_platform;
    if (num_platforms) *num_platforms = 1;
    return CL_SUCCESS;
}

cl_int clGetPlatformInfo(cl_platform_id platform, cl_platform_info name, size_t size, void *value, size_t *size_ret) {
    if (platform != &the_platform) return CL_INVALID_PLATFORM;
    switch (name) {
    case CL_PLATFORM_PROFILE: return set_str(size, value, size_ret, "EMBEDDED_PROFILE");
    case CL_PLATFORM_VERSION: return set_str(size, value, size_ret, "OpenCL 1.2 qgpu");
    case CL_PLATFORM_NAME: return set_str(size, value, size_ret, "Intel-Phi-QwenGPU");
    case CL_PLATFORM_VENDOR: return set_str(size, value, size_ret, "Lasimeri");
    case CL_PLATFORM_EXTENSIONS: return set_str(size, value, size_ret, "");
    default: return CL_INVALID_VALUE;
    }
}

cl_int clGetDeviceIDs(cl_platform_id platform, cl_device_type type, cl_uint num_entries, cl_device_id *devices, cl_uint *num_devices) {
    if (platform && platform != &the_platform) return CL_INVALID_PLATFORM;
    if (!(type & (CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_DEFAULT | CL_DEVICE_TYPE_ALL))) {
        if (num_devices) *num_devices = 0;
        return CL_DEVICE_NOT_FOUND;
    }
    Device::get();
    if (devices && num_entries >= 1) devices[0] = &the_device;
    if (num_devices) *num_devices = 1;
    return CL_SUCCESS;
}

cl_int clGetDeviceInfo(cl_device_id device, cl_device_info name, size_t size, void *value, size_t *size_ret) {
    if (device != &the_device) return CL_INVALID_DEVICE;
    Device &d = Device::get();
    switch (name) {
    case CL_DEVICE_TYPE: return set_val<cl_device_type>(size, value, size_ret, CL_DEVICE_TYPE_GPU);
    case CL_DEVICE_VENDOR_ID: return set_val<cl_uint>(size, value, size_ret, 0x5147);
    case CL_DEVICE_MAX_COMPUTE_UNITS: return set_val<cl_uint>(size, value, size_ret, 1);
    case CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: return set_val<cl_uint>(size, value, size_ret, 1);
    case CL_DEVICE_MAX_WORK_ITEM_SIZES: {
        size_t s[3] = {256, 1, 1};
        return set_info(size, value, size_ret, s, sizeof s);
    }
    case CL_DEVICE_MAX_WORK_GROUP_SIZE: return set_val<size_t>(size, value, size_ret, 256);
    case CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT: return set_val<cl_uint>(size, value, size_ret, 1);
    case CL_DEVICE_ADDRESS_BITS: return set_val<cl_uint>(size, value, size_ret, 32);
    case CL_DEVICE_MAX_MEM_ALLOC_SIZE: return set_val<cl_ulong>(size, value, size_ret, (cl_ulong)d.im.m.size() * 4);
    case CL_DEVICE_GLOBAL_MEM_SIZE: return set_val<cl_ulong>(size, value, size_ret, (cl_ulong)d.im.m.size() * 4);
    case CL_DEVICE_LOCAL_MEM_TYPE: return set_val<cl_device_local_mem_type>(size, value, size_ret, CL_GLOBAL);
    case CL_DEVICE_LOCAL_MEM_SIZE: return set_val<cl_ulong>(size, value, size_ret, 4096 * 4);
    case CL_DEVICE_MAX_CLOCK_FREQUENCY: return set_val<cl_uint>(size, value, size_ret, 0);
    case CL_DEVICE_AVAILABLE:
    case CL_DEVICE_COMPILER_AVAILABLE:
    case CL_DEVICE_LINKER_AVAILABLE:
    case CL_DEVICE_ENDIAN_LITTLE: return set_val<cl_bool>(size, value, size_ret, CL_TRUE);
    case CL_DEVICE_IMAGE_SUPPORT: return set_val<cl_bool>(size, value, size_ret, CL_FALSE);
    case CL_DEVICE_NAME: return set_str(size, value, size_ret, d.name.c_str());
    case CL_DEVICE_VENDOR: return set_str(size, value, size_ret, "Lasimeri");
    case CL_DRIVER_VERSION: return set_str(size, value, size_ret, "0.1");
    case CL_DEVICE_PROFILE: return set_str(size, value, size_ret, "EMBEDDED_PROFILE");
    case CL_DEVICE_VERSION: return set_str(size, value, size_ret, "OpenCL 1.2 qgpu");
    case CL_DEVICE_OPENCL_C_VERSION: return set_str(size, value, size_ret, "OpenCL C 2.0 qgpu");
    case CL_DEVICE_EXTENSIONS: return set_str(size, value, size_ret, "");
    case CL_DEVICE_PLATFORM: return set_val<cl_platform_id>(size, value, size_ret, &the_platform);
    case CL_DEVICE_MAX_CONSTANT_ARGS: return set_val<cl_uint>(size, value, size_ret, 8);
    case CL_DEVICE_MAX_PARAMETER_SIZE: return set_val<size_t>(size, value, size_ret, 256);
    case CL_DEVICE_MEM_BASE_ADDR_ALIGN: return set_val<cl_uint>(size, value, size_ret, 32);
    case CL_DEVICE_EXECUTION_CAPABILITIES: return set_val<cl_device_exec_capabilities>(size, value, size_ret, CL_EXEC_KERNEL);
    case CL_DEVICE_QUEUE_PROPERTIES: return set_val<cl_command_queue_properties>(size, value, size_ret, 0);
    default: return CL_INVALID_VALUE;
    }
}

cl_context clCreateContext(const cl_context_properties *, cl_uint num_devices, const cl_device_id *devices, void (*)(const char *, const void *, size_t, void *), void *, cl_int *err) {
    if (num_devices != 1 || !devices || devices[0] != &the_device) {
        if (err) *err = CL_INVALID_DEVICE;
        return nullptr;
    }
    Device::get();
    if (err) *err = CL_SUCCESS;
    return new _cl_context;
}
cl_context clCreateContextFromType(const cl_context_properties *, cl_device_type type, void (*)(const char *, const void *, size_t, void *), void *, cl_int *err) {
    if (!(type & (CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_DEFAULT | CL_DEVICE_TYPE_ALL))) {
        if (err) *err = CL_DEVICE_NOT_FOUND;
        return nullptr;
    }
    Device::get();
    if (err) *err = CL_SUCCESS;
    return new _cl_context;
}
cl_int clRetainContext(cl_context c) { return c ? (c->refs++, CL_SUCCESS) : CL_INVALID_CONTEXT; }
cl_int clReleaseContext(cl_context c) {
    if (!c) return CL_INVALID_CONTEXT;
    if (--c->refs == 0) delete c;
    return CL_SUCCESS;
}

cl_command_queue clCreateCommandQueue(cl_context ctx, cl_device_id device, cl_command_queue_properties, cl_int *err) {
    if (!ctx || device != &the_device) {
        if (err) *err = CL_INVALID_DEVICE;
        return nullptr;
    }
    cl_command_queue q = new _cl_command_queue;
    q->ctx = ctx;
    if (err) *err = CL_SUCCESS;
    return q;
}
cl_int clRetainCommandQueue(cl_command_queue q) { return q ? (q->refs++, CL_SUCCESS) : CL_INVALID_COMMAND_QUEUE; }
cl_int clReleaseCommandQueue(cl_command_queue q) {
    if (!q) return CL_INVALID_COMMAND_QUEUE;
    Device::get().flush();
    if (--q->refs == 0) delete q;
    return CL_SUCCESS;
}
cl_int clFlush(cl_command_queue q) { return q ? CL_SUCCESS : CL_INVALID_COMMAND_QUEUE; }
cl_int clFinish(cl_command_queue q) { return q ? Device::get().flush() : CL_INVALID_COMMAND_QUEUE; }

cl_mem clCreateBuffer(cl_context ctx, cl_mem_flags flags, size_t size, void *host_ptr, cl_int *err) {
    if (!ctx) {
        if (err) *err = CL_INVALID_CONTEXT;
        return nullptr;
    }
    if (size == 0 || ((flags & (CL_MEM_COPY_HOST_PTR | CL_MEM_USE_HOST_PTR)) && !host_ptr)) {
        if (err) *err = CL_INVALID_HOST_PTR;
        return nullptr;
    }
    Device &d = Device::get();
    cl_mem m = new _cl_mem;
    m->ctx = ctx;
    m->bytes = size;
    m->words = (uint32_t)((size + 3) / 4);
    m->word = d.alloc(m->words);
    if (flags & (CL_MEM_COPY_HOST_PTR | CL_MEM_USE_HOST_PTR)) std::memcpy(d.bytes(m->word), host_ptr, size);
    ctx->mems.push_back(m);
    if (err) *err = CL_SUCCESS;
    return m;
}
cl_int clRetainMemObject(cl_mem m) { return m ? (m->refs++, CL_SUCCESS) : CL_INVALID_MEM_OBJECT; }
cl_int clReleaseMemObject(cl_mem m) {
    if (!m) return CL_INVALID_MEM_OBJECT;
    if (--m->refs == 0) {
        std::vector<cl_mem> &v = m->ctx->mems;
        for (size_t i = 0; i < v.size(); i++)
            if (v[i] == m) {
                v.erase(v.begin() + (long)i);
                break;
            }
        delete m;  // the words stay allocated: a bump allocator
    }
    return CL_SUCCESS;
}

cl_int clEnqueueReadBuffer(cl_command_queue q, cl_mem m, cl_bool, size_t offset, size_t size, void *ptr, cl_uint, const cl_event *, cl_event *event) {
    if (!q) return CL_INVALID_COMMAND_QUEUE;
    if (!m || m->ctx != q->ctx) return CL_INVALID_MEM_OBJECT;
    if (offset + size > m->bytes || !ptr) return CL_INVALID_VALUE;
    Device &d = Device::get();
    cl_int rc = d.flush();  // in order: every pending launch before the copy
    if (rc != CL_SUCCESS) return rc;
    std::memcpy(ptr, d.bytes(m->word) + offset, size);
    if (event) *event = new _cl_event;
    return CL_SUCCESS;
}
cl_int clEnqueueWriteBuffer(cl_command_queue q, cl_mem m, cl_bool, size_t offset, size_t size, const void *ptr, cl_uint, const cl_event *, cl_event *event) {
    if (!q) return CL_INVALID_COMMAND_QUEUE;
    if (!m || m->ctx != q->ctx) return CL_INVALID_MEM_OBJECT;
    if (offset + size > m->bytes || !ptr) return CL_INVALID_VALUE;
    Device &d = Device::get();
    cl_int rc = d.flush();
    if (rc != CL_SUCCESS) return rc;
    std::memcpy(d.bytes(m->word) + offset, ptr, size);
    if (event) *event = new _cl_event;
    return CL_SUCCESS;
}

cl_program clCreateProgramWithSource(cl_context ctx, cl_uint count, const char **strings, const size_t *lengths, cl_int *err) {
    if (!ctx) {
        if (err) *err = CL_INVALID_CONTEXT;
        return nullptr;
    }
    if (count == 0 || !strings) {
        if (err) *err = CL_INVALID_VALUE;
        return nullptr;
    }
    cl_program p = new _cl_program;
    p->ctx = ctx;
    for (cl_uint i = 0; i < count; i++) {
        if (!strings[i]) {
            delete p;
            if (err) *err = CL_INVALID_VALUE;
            return nullptr;
        }
        if (lengths && lengths[i]) p->source.append(strings[i], lengths[i]);
        else p->source.append(strings[i]);
    }
    if (err) *err = CL_SUCCESS;
    return p;
}
cl_program clCreateProgramWithBinary(cl_context ctx, cl_uint num_devices, const cl_device_id *devices, const size_t *lengths, const unsigned char **binaries, cl_int *binary_status, cl_int *err) {
    if (!ctx) {
        if (err) *err = CL_INVALID_CONTEXT;
        return nullptr;
    }
    if (num_devices != 1 || !devices || devices[0] != &the_device || !lengths || !binaries || !binaries[0]) {
        if (err) *err = CL_INVALID_VALUE;
        return nullptr;
    }
    cl_program p = new _cl_program;
    p->ctx = ctx;
    p->ptx.assign((const char *)binaries[0], lengths[0]);  // a binary is PTX text
    if (binary_status) binary_status[0] = CL_SUCCESS;
    if (err) *err = CL_SUCCESS;
    return p;
}
cl_int clBuildProgram(cl_program p, cl_uint, const cl_device_id *, const char *options, void (*notify)(cl_program, void *), void *user) {
    if (!p) return CL_INVALID_PROGRAM;
    p->log.clear();
    if (p->ptx.empty()) {
        // the source through the system clang, the device header pre-included: the build scripts/cl_build.sh does
        char dir[] = "/tmp/qgpucl-XXXXXX";
        if (!mkdtemp(dir)) return CL_BUILD_PROGRAM_FAILURE;
        const std::string src = std::string(dir) + "/k.cl", out = std::string(dir) + "/k.ptx", errf = std::string(dir) + "/k.err";
        {
            std::ofstream f(src, std::ios::binary);
            f << p->source;
        }
        std::string cmd = "clang -x cl -cl-std=CL2.0 -cl-no-stdinc -target nvptx64-nvidia-cuda -O2 -I '" QGPU_ROOT "/src/cl' -I '" QGPU_ROOT "/src/guest' -include qgpucl.h ";
        if (options) cmd += std::string(options) + " ";
        cmd += "-S -o '" + out + "' '" + src + "' 2> '" + errf + "'";
        int rc = std::system(cmd.c_str());
        p->log = read_file(errf);
        if (rc == 0) p->ptx = read_file(out);
        unlink(src.c_str());
        unlink(out.c_str());
        unlink(errf.c_str());
        rmdir(dir);
        if (rc != 0 || p->ptx.empty()) {
            if (notify) notify(p, user);
            return CL_BUILD_PROGRAM_FAILURE;
        }
    }
    p->kernels = ptx::parse(p->ptx);
    p->built = true;
    if (notify) notify(p, user);
    return CL_SUCCESS;
}
cl_int clGetProgramBuildInfo(cl_program p, cl_device_id device, cl_program_build_info name, size_t size, void *value, size_t *size_ret) {
    if (!p) return CL_INVALID_PROGRAM;
    if (device != &the_device) return CL_INVALID_DEVICE;
    switch (name) {
    case CL_PROGRAM_BUILD_STATUS: return set_val<cl_build_status>(size, value, size_ret, p->built ? CL_BUILD_SUCCESS : CL_BUILD_ERROR);
    case CL_PROGRAM_BUILD_OPTIONS: return set_str(size, value, size_ret, "");
    case CL_PROGRAM_BUILD_LOG: return set_str(size, value, size_ret, p->log.c_str());
    default: return CL_INVALID_VALUE;
    }
}
cl_int clGetProgramInfo(cl_program p, cl_program_info name, size_t size, void *value, size_t *size_ret) {
    if (!p) return CL_INVALID_PROGRAM;
    switch (name) {
    case CL_PROGRAM_NUM_DEVICES: return set_val<cl_uint>(size, value, size_ret, 1);
    case CL_PROGRAM_SOURCE: return set_str(size, value, size_ret, p->source.c_str());
    case CL_PROGRAM_BINARY_SIZES: return set_val<size_t>(size, value, size_ret, p->ptx.size());
    case CL_PROGRAM_BINARIES: {
        if (size_ret) *size_ret = sizeof(uintptr_t);
        if (value) std::memcpy(*(unsigned char **)value, p->ptx.data(), p->ptx.size());
        return CL_SUCCESS;
    }
    case CL_PROGRAM_NUM_KERNELS: return set_val<size_t>(size, value, size_ret, p->kernels.size());
    default: return CL_INVALID_VALUE;
    }
}
cl_int clRetainProgram(cl_program p) { return p ? (p->refs++, CL_SUCCESS) : CL_INVALID_PROGRAM; }
cl_int clReleaseProgram(cl_program p) {
    if (!p) return CL_INVALID_PROGRAM;
    if (--p->refs == 0) delete p;
    return CL_SUCCESS;
}

cl_kernel clCreateKernel(cl_program p, const char *name, cl_int *err) {
    if (!p || !p->built) {
        if (err) *err = CL_INVALID_PROGRAM_EXECUTABLE;
        return nullptr;
    }
    auto it = p->kernels.find(name ? name : "");
    if (it == p->kernels.end()) {
        if (err) *err = CL_INVALID_KERNEL_NAME;
        return nullptr;
    }
    cl_kernel k = new _cl_kernel;
    k->prog = p;
    p->refs++;
    k->name = name;
    k->k = &it->second;
    k->args.resize(it->second.params.size());
    if (err) *err = CL_SUCCESS;
    return k;
}
cl_int clSetKernelArg(cl_kernel k, cl_uint index, size_t size, const void *value) {
    if (!k) return CL_INVALID_KERNEL;
    if (index >= k->args.size()) return CL_INVALID_ARG_INDEX;
    if (!value) return CL_INVALID_ARG_VALUE;
    const ptx::Param &pm = k->k->params[index];
    Arg &a = k->args[index];
    if (pm.size == 8) {  // a pointer parameter: a cl_mem
        if (size != sizeof(cl_mem)) return CL_INVALID_ARG_SIZE;
        cl_mem m;
        std::memcpy(&m, value, sizeof(cl_mem));
        if (!is_mem(k->prog->ctx, m)) return CL_INVALID_MEM_OBJECT;
        a.kind = Arg::BUFFER;
        a.mem = m;
        return CL_SUCCESS;
    }
    if (pm.size != 4 || size != 4) return CL_INVALID_ARG_SIZE;  // the machine's scalars are 32 bits
    a.kind = Arg::SCALAR;
    std::memcpy(&a.value, value, 4);
    return CL_SUCCESS;
}
cl_int clGetKernelWorkGroupInfo(cl_kernel k, cl_device_id, cl_kernel_work_group_info name, size_t size, void *value, size_t *size_ret) {
    if (!k) return CL_INVALID_KERNEL;
    switch (name) {
    case CL_KERNEL_WORK_GROUP_SIZE: return set_val<size_t>(size, value, size_ret, 256);
    case CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: return set_val<size_t>(size, value, size_ret, 32);
    case CL_KERNEL_LOCAL_MEM_SIZE: return set_val<cl_ulong>(size, value, size_ret, 0);
    case CL_KERNEL_PRIVATE_MEM_SIZE: return set_val<cl_ulong>(size, value, size_ret, 64);
    default: return CL_INVALID_VALUE;
    }
}
cl_int clRetainKernel(cl_kernel k) { return k ? (k->refs++, CL_SUCCESS) : CL_INVALID_KERNEL; }
cl_int clReleaseKernel(cl_kernel k) {
    if (!k) return CL_INVALID_KERNEL;
    if (--k->refs == 0) {
        clReleaseProgram(k->prog);
        delete k;
    }
    return CL_SUCCESS;
}

cl_int clEnqueueNDRangeKernel(cl_command_queue q, cl_kernel k, cl_uint work_dim, const size_t *offset, const size_t *global, const size_t *local, cl_uint, const cl_event *, cl_event *event) {
    if (!q) return CL_INVALID_COMMAND_QUEUE;
    if (!k) return CL_INVALID_KERNEL;
    if (work_dim != 1) return CL_INVALID_WORK_DIMENSION;
    if (!global || global[0] == 0) return CL_INVALID_GLOBAL_WORK_SIZE;
    if (offset && offset[0] != 0) return CL_INVALID_GLOBAL_OFFSET;
    const size_t lsz = local ? local[0] : 32;
    if (lsz == 0 || lsz % 32 || lsz > 256 || global[0] % lsz) return CL_INVALID_WORK_GROUP_SIZE;
    if (global[0] > 8192) return CL_INVALID_GLOBAL_WORK_SIZE;  // 256 warps a launch
    // Scalars are runtime args in the launch's args block (u14). Pointers are baked into the instance (a buffer's
    // word) until a launch of the same kernel brings a different pointer for a parameter: then that parameter becomes
    // a runtime arg too and the kernel is lowered once more; from there one instance serves every launch over any
    // buffers. (The 2B: 29 kernels, a few dozen instances, whatever the layer.)
    Device &d = Device::get();
    for (size_t i = 0; i < k->args.size(); i++)
        if (k->args[i].kind == Arg::NONE) return CL_INVALID_KERNEL_ARGS;
    Device::Spec &spec = d.specs[k->k];
    if (spec.varying.size() != k->args.size()) {
        spec.varying.assign(k->args.size(), false);
        spec.last_word.assign(k->args.size(), 0);
        spec.seen = false;
    }
    if (spec.seen)
        for (size_t i = 0; i < k->args.size(); i++)
            if (k->args[i].kind == Arg::BUFFER && !spec.varying[i] && spec.last_word[i] != k->args[i].mem->word) spec.varying[i] = true;
    std::vector<lower::ParamSpec> ps;
    std::string key = k->name;
    int32_t argw[16];
    int nargs = 0;
    for (size_t i = 0; i < k->args.size(); i++) {
        const Arg &a = k->args[i];
        lower::ParamSpec p;
        if (a.kind == Arg::BUFFER && !spec.varying[i]) {
            p.kind = lower::ParamSpec::PTR;
            p.value = (int64_t)a.mem->word;
            key += ":p" + std::to_string(p.value);
        } else {
            if (nargs == 16) return CL_OUT_OF_RESOURCES;
            p.kind = lower::ParamSpec::ARG;
            p.value = nargs;
            argw[nargs++] = a.kind == Arg::BUFFER ? (int32_t)a.mem->word : a.value;
            key += ":a";
        }
        if (a.kind == Arg::BUFFER) spec.last_word[i] = a.mem->word;
        ps.push_back(p);
    }
    spec.seen = true;
    key += ":g" + std::to_string(global[0] / lsz) + ":b" + std::to_string(lsz);
    auto it = d.instances.find(key);
    if (it == d.instances.end()) {
        lower::Options opt;
        opt.shared_base = d.shared_base;
        lower::Launch L;
        try {
            L = lower::lower(*k->k, (int)(global[0] / lsz), 1, (int)lsz, 1, ps, opt);
        } catch (const lower::Unsupported &e) {
            k->prog->log += std::string("qgpucl: ") + k->name + ": " + e.what() + "\n";
            std::fprintf(stderr, "qgpucl: %s: %s\n", k->name.c_str(), e.what());
            return CL_INVALID_KERNEL;
        }
        Instance in;
        in.base = (uint32_t)d.im.rom.size();
        if (in.base + L.blocks.size() > (1u << SM_BBITS)) return CL_OUT_OF_RESOURCES;
        lower::encode(L, d.im.rom, in.base);
        d.im.bind();
        in.warps = L.warps;
        in.lwpb = L.lwpb;
        it = d.instances.emplace(key, in).first;
    }
    const uint32_t ab = d.args_base + 16 * (uint32_t)d.pending.size();
    for (int i = 0; i < nargs; i++) d.im.m[ab + (uint32_t)i] = argw[i];
    d.pending.emplace_back(sm_qword(SM_Q_LAUNCH, it->second.base, it->second.warps, (uint32_t)it->second.lwpb, 0), ab);
    d.launches++;
    if (event) *event = new _cl_event;
    if (d.queue_base + d.pending.size() + 1 >= (1u << SM_QBITS)) return d.flush();
    return CL_SUCCESS;
}

cl_int clWaitForEvents(cl_uint n, const cl_event *events) {
    if (n == 0 || !events) return CL_INVALID_VALUE;
    return Device::get().flush();
}
cl_int clGetEventInfo(cl_event e, cl_event_info name, size_t size, void *value, size_t *size_ret) {
    if (!e) return CL_INVALID_EVENT;
    if (name == CL_EVENT_COMMAND_EXECUTION_STATUS) return set_val<cl_int>(size, value, size_ret, CL_COMPLETE);
    return CL_INVALID_VALUE;
}
cl_int clRetainEvent(cl_event e) { return e ? (e->refs++, CL_SUCCESS) : CL_INVALID_EVENT; }
cl_int clReleaseEvent(cl_event e) {
    if (!e) return CL_INVALID_EVENT;
    if (--e->refs == 0) delete e;
    return CL_SUCCESS;
}
cl_int clEnqueueMarker(cl_command_queue q, cl_event *event) {
    if (!q) return CL_INVALID_COMMAND_QUEUE;
    if (event) *event = new _cl_event;
    return CL_SUCCESS;
}
cl_int clEnqueueBarrier(cl_command_queue q) { return q ? CL_SUCCESS : CL_INVALID_COMMAND_QUEUE; }

// qgpu's own (src/cl/qgpucl_ext.h): a buffer over an existing word range of M, and the machine's counters
cl_mem qgpuCreateBufferAt(cl_context ctx, cl_uint word, size_t bytes, cl_int *err) {
    if (!ctx) {
        if (err) *err = CL_INVALID_CONTEXT;
        return nullptr;
    }
    Device &d = Device::get();
    cl_mem m = new _cl_mem;
    m->ctx = ctx;
    m->bytes = bytes ? bytes : 4;
    m->words = (uint32_t)((m->bytes + 3) / 4);
    m->word = word;
    if ((size_t)word + m->words > d.im.m.size()) {
        d.im.m.resize((size_t)word + m->words, 0);
        d.im.bind();
        if (d.free_word < word + m->words) d.free_word = word + m->words;
    }
    ctx->mems.push_back(m);
    if (err) *err = CL_SUCCESS;
    return m;
}
void qgpuStats(unsigned long long *records, unsigned long long *trace_tokens, unsigned long long *launches) {
    Device &d = Device::get();
    if (records) *records = d.mc->records;
    if (trace_tokens) *trace_tokens = d.mc->sink.n;
    if (launches) *launches = d.launches;
}

}  // extern "C"
