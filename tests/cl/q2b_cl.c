/* The 2B through OpenCL: a host program that replays the 2B's program (out/gpucl/program.txt, one descriptor a line) on
 * the embedded GPU through the standard API: every launch is clSetKernelArg + clEnqueueNDRangeKernel, a branch reads
 * its word (clEnqueueReadBuffer, which runs what is pending) and jumps, an emit reads the token, halt ends the turn. The
 * device must be the 2B's image: QGPU_IMAGE=out/gpucl. The kernels come from the PTX as a program binary. Host.
 *
 *   q2b_cl PROGRAM.txt KERNELS.ptx GEN TOKEN... [-- EXPECTED...]
 * prints every emitted token id; with EXPECTED compares them and exits 1 on any difference. */
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../src/cl/qgpucl_ext.h"
#include "../../src/guest/q2b_map.h"

#define CHECK(err, what)                                                      \
    do {                                                                      \
        if ((err) != CL_SUCCESS) {                                            \
            fprintf(stderr, "q2b_cl: %s failed: %d\n", what, (int)(err));    \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

enum { MAXP = 24, MAXA = 8, MAXK = 64, MAXL = 4096 };
typedef struct {
    int kind;  /* 0 launch, 1 branch, 2 emit, 3 halt */
    char name[48];
    int gx, bx, np, na;
    char pkind[MAXP];
    long pval[MAXP];
    int args[MAXA];
    unsigned word, target;
} Line;

static Line prog[MAXL];
static int nlines;
static cl_context ctx;
static cl_command_queue q;
static cl_kernel kernels[MAXK];
static char knames[MAXK][48];
static int nk;
static cl_mem bufs[1024];
static unsigned bufword[1024];
static int nbufs;

static cl_mem buffer_at(unsigned word) {
    for (int i = 0; i < nbufs; i++)
        if (bufword[i] == word) return bufs[i];
    if (nbufs == 1024) {
        fputs("q2b_cl: too many buffers\n", stderr);
        exit(1);
    }
    cl_int err;
    cl_mem m = qgpuCreateBufferAt(ctx, word, 4, &err);
    CHECK(err, "qgpuCreateBufferAt");
    bufword[nbufs] = word;
    bufs[nbufs] = m;
    return bufs[nbufs++];
}
static cl_kernel kernel_named(const char *name, cl_program p) {
    for (int i = 0; i < nk; i++)
        if (!strcmp(knames[i], name)) return kernels[i];
    if (nk == MAXK) {
        fputs("q2b_cl: too many kernels\n", stderr);
        exit(1);
    }
    cl_int err;
    kernels[nk] = clCreateKernel(p, name, &err);
    CHECK(err, name);
    strcpy(knames[nk], name);
    return kernels[nk++];
}
static int read_word(unsigned word) {
    int v = 0;
    CHECK(clEnqueueReadBuffer(q, buffer_at(word), CL_TRUE, 0, 4, &v, 0, NULL, NULL), "clEnqueueReadBuffer");
    return v;
}
static void write_word(unsigned word, int v) { CHECK(clEnqueueWriteBuffer(q, buffer_at(word), CL_TRUE, 0, 4, &v, 0, NULL, NULL), "clEnqueueWriteBuffer"); }

static void load_program(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) {
        perror(path);
        exit(1);
    }
    char line[4096];
    while (fgets(line, sizeof line, f)) {
        if (nlines == MAXL) {
            fputs("q2b_cl: too many lines\n", stderr);
            exit(1);
        }
        Line *L = &prog[nlines];
        memset(L, 0, sizeof *L);
        char *tok = strtok(line, " \n");
        if (!tok) continue;
        if (!strcmp(tok, "launch")) {
            L->kind = 0;
            strncpy(L->name, strtok(NULL, " \n"), sizeof L->name - 1);
            L->gx = atoi(strtok(NULL, " \n"));
            L->bx = atoi(strtok(NULL, " \n"));
            int in_args = 0;
            while ((tok = strtok(NULL, " \n"))) {
                if (!strcmp(tok, "|")) {
                    in_args = 1;
                    continue;
                }
                if (in_args) {
                    if (L->na < MAXA) L->args[L->na++] = atoi(tok);
                } else if (L->np < MAXP) {
                    L->pkind[L->np] = tok[0];
                    L->pval[L->np++] = atol(tok + 2);
                }
            }
        } else if (!strcmp(tok, "branch")) {
            L->kind = 1;
            L->word = (unsigned)strtoul(strtok(NULL, " \n"), NULL, 10);
            L->target = (unsigned)strtoul(strtok(NULL, " \n"), NULL, 10);
        } else if (!strcmp(tok, "emit")) {
            L->kind = 2;
            L->word = (unsigned)strtoul(strtok(NULL, " \n"), NULL, 10);
        } else {
            L->kind = 3;
        }
        nlines++;
    }
    fclose(f);
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fputs("usage: q2b_cl PROGRAM.txt KERNELS.ptx GEN TOKEN... [-- EXPECTED...]\n", stderr);
        return 2;
    }
    load_program(argv[1]);
    const int gen = atoi(argv[3]);
    int toks[1024], ntoks = 0, expected[1024], nexp = 0, in_exp = 0;
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--")) {
            in_exp = 1;
            continue;
        }
        if (in_exp) expected[nexp++] = atoi(argv[i]);
        else toks[ntoks++] = atoi(argv[i]);
    }
    /* the PTX as the program binary */
    FILE *pf = fopen(argv[2], "rb");
    if (!pf) {
        perror(argv[2]);
        return 1;
    }
    fseek(pf, 0, SEEK_END);
    size_t plen = (size_t)ftell(pf);
    fseek(pf, 0, SEEK_SET);
    unsigned char *ptx = malloc(plen);
    if (fread(ptx, 1, plen, pf) != plen) return 1;
    fclose(pf);

    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    CHECK(clGetPlatformIDs(1, &platform, NULL), "clGetPlatformIDs");
    CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL), "clGetDeviceIDs");
    char dname[256];
    CHECK(clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof dname, dname, NULL), "clGetDeviceInfo");
    printf("host: device: %s\n", dname);
    ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK(err, "clCreateContext");
    q = clCreateCommandQueue(ctx, device, 0, &err);
    CHECK(err, "clCreateCommandQueue");
    const unsigned char *bins[1] = {ptx};
    cl_program p = clCreateProgramWithBinary(ctx, 1, &device, &plen, bins, NULL, &err);
    CHECK(err, "clCreateProgramWithBinary");
    CHECK(clBuildProgram(p, 1, &device, "", NULL, NULL), "clBuildProgram");

    /* the job: the prompt's tokens and the job words, as q2b_gpu_check writes them */
    for (int i = 0; i < ntoks; i++) write_word((unsigned)M_TOKS + (unsigned)i, toks[i]);
    write_word(JOB_SEQ, 1);
    write_word(JOB_N, ntoks);
    write_word(JOB_GEN, gen);
    write_word(JOB_POS, 0);
    write_word(JOB_PROMPT_N, ntoks);
    write_word(JOB_MADE, 0);
    write_word(JOB_NSTOPS, 1);
    write_word(JOB_STOPS, 248046); /* <|im_end|> */

    /* the program, literally */
    int emitted[1024], nem = 0;
    unsigned pc = 0, steps = 0;
    while (pc < (unsigned)nlines && steps++ < 2000000u) {
        Line *L = &prog[pc];
        if (L->kind == 0) {
            cl_kernel k = kernel_named(L->name, p);
            for (int i = 0; i < L->np; i++) {
                if (L->pkind[i] == 'p' || L->pkind[i] == 'q') {  // a pointer: baked, or from this launch's args
                    cl_mem m = buffer_at(L->pkind[i] == 'p' ? (unsigned)L->pval[i] : (unsigned)L->args[L->pval[i]]);
                    CHECK(clSetKernelArg(k, (cl_uint)i, sizeof(cl_mem), &m), L->name);
                } else {
                    int v = L->pkind[i] == 'c' ? (int)L->pval[i] : L->args[L->pval[i]];
                    CHECK(clSetKernelArg(k, (cl_uint)i, 4, &v), L->name);
                }
            }
            size_t global = (size_t)L->gx * (size_t)L->bx, local = (size_t)L->bx;
            CHECK(clEnqueueNDRangeKernel(q, k, 1, NULL, &global, &local, 0, NULL, NULL), L->name);
            pc++;
        } else if (L->kind == 1) {
            pc = read_word(L->word) ? L->target : pc + 1;
        } else if (L->kind == 2) {
            int t = read_word(L->word);
            emitted[nem++ & 1023] = t;
            printf("token %d\n", t);
            fflush(stdout);
            pc++;
        } else {
            break;
        }
    }
    CHECK(clFinish(q), "clFinish");
    unsigned long long records, trace, launches;
    qgpuStats(&records, &trace, &launches);
    printf("host: %d tokens emitted, %llu launches, %llu records, %llu trace tokens, %u program steps\n", nem, launches, records, trace, steps);
    int ok = 1;
    if (nexp) {
        ok = nem == nexp;
        for (int i = 0; ok && i < nem; i++) ok = emitted[i] == expected[i];
        printf("host: q2b_cl: %s\n", ok ? "every token as expected" : "TOKENS DIFFER FROM THE EXPECTED LIST");
    }
    return ok ? 0 : 1;
}
