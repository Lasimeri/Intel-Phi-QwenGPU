/* The card side of the merged model: the computer's layers run on the 7120A, free running, one token after another. The host only gives it the image and
 * prompts, and reads what comes out: this program holds no interpreter, no NPU, no reference.
 *
 *   chip_card CORES TPC PAGES WIDTH PAR_MIN MEMO MAX_TOKENS
 *
 * Runs as root; logical CPU 1 + 4c + t is thread t of core c; the run script reserves the 2 MiB pages first. */
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "../text/q35_link.h"
#include "chip_engine.h"
#include "q35_dma.h"

#define HUGE_2M (2ULL << 20)

static uint8_t *arena;
static uint64_t *arena_phys, arena_pages;
static uint64_t region_addr;
static ChipCtl *ctl;

static void die(const char *what) {
    fprintf(stderr, "chip_card: %s\n", what);
    exit(2);
}

static void arena_alloc(uint64_t pages) {
    arena_pages = pages;
    arena = mmap(NULL, arena_pages * HUGE_2M, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE, -1, 0);
    if (arena == MAP_FAILED) die("not enough 2 MiB pages: reserve more in vm.nr_hugepages");
    if (mlock(arena, arena_pages * HUGE_2M)) perror("mlock");
    arena_phys = malloc(arena_pages * 8);
    int pm = open("/proc/self/pagemap", O_RDONLY);
    if (pm < 0) die("no pagemap");
    for (uint64_t i = 0; i < arena_pages; i++) {
        uint64_t e = 0, e2 = 0;
        const uintptr_t v = (uintptr_t)(arena + i * HUGE_2M);
        if (pread(pm, &e, 8, (off_t)(v / 4096) * 8) != 8 || pread(pm, &e2, 8, (off_t)((v + HUGE_2M - 4096) / 4096) * 8) != 8) die("pagemap read");
        const uint64_t mask = (1ULL << 55) - 1;
        arena_phys[i] = (e & mask) * 4096;
        if (!(e >> 63) || arena_phys[i] % HUGE_2M || (e2 & mask) * 4096 != arena_phys[i] + HUGE_2M - 4096) die("a huge page is not contiguous");
    }
    close(pm);
}

static uint64_t to_arena(uint64_t src, uint64_t to, uint64_t bytes) {
    uint64_t ticket = 0;
    while (bytes) {
        const uint64_t in = to % HUGE_2M;
        uint64_t run = HUGE_2M - in;
        if (run > bytes) run = bytes;
        if (run > Q35_DMA_MAX_RUN) run = Q35_DMA_MAX_RUN;
        ticket = q35_dma_copy(src, arena_phys[to / HUGE_2M] + in, run);
        src += run;
        to += run;
        bytes -= run;
    }
    return ticket;
}

static void pin_to(int cpu) {
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET((size_t)cpu, &s);
    if (sched_setaffinity(0, sizeof s, &s)) perror("sched_setaffinity");
}

typedef struct {
    ChipEngine *e;
    int tid, cpu;
    int vpu_ok;
    uint32_t seen;
} Arg;

static void *member_main(void *p) {
    Arg *a = p;
    pin_to(a->cpu);
    a->vpu_ok = chip_vpu_setup(&a->seen);
    chip_engine_worker(a->e, a->tid);
    return NULL;
}

static uint64_t ev_head;

static void post(uint64_t kind, uint64_t a, uint64_t b, uint64_t c) {
    volatile ChipEvent *v = &ctl->ev[ev_head % CHIP_EVENTS];
    v->kind = kind;
    v->a = a;
    v->b = b;
    v->c = c;
    ctl->ev_head = ++ev_head;
}

/* ------------------------------------------------------------------------------------ a checkpoint
 *
 * The machine's state and nothing the engine only remembers: the conv and memory windows, oldest first, the attention states, VM, VS and the KV cache. Everything a line
 * keeps to save work starts again from zeros on a resume, which is what it would be had those inputs been zero last token, so any engine can go on from any checkpoint. */

static uint64_t state_walk(uint8_t *pack, uint32_t next_t, int restore) {
    const ChipHeader *h = (const ChipHeader *)arena;
    ChipLayer *layers = (ChipLayer *)(arena + h->o_layers);
    const uint32_t r[3] = {next_t % 3, (next_t + 1) % 3, (next_t + 2) % 3};
    uint8_t *p = pack;
    for (uint32_t il = 0; il < h->layers; il++) {
        ChipLayer *L = &layers[il];
        ChipConv *conv = (ChipConv *)(arena + L->o_conv);
        ChipSell *work = (ChipSell *)(arena + L->a.o_work);
        for (uint32_t i = 0; i < L->n_conv; i++) {
            for (int k = 0; k < 3; k++) {
                if (restore) memcpy(conv[i].win[r[k]], p, 64);
                else memcpy(p, conv[i].win[r[k]], 64);
                p += 64;
            }
            /* a window may hold what the projection does not: this line computes again until the ring has gone round */
            if (restore) work[conv[i].work].changed_at = next_t;
        }
        ChipHead *heads = (ChipHead *)(arena + L->o_heads);
        for (uint32_t i = 0; i < L->n_heads; i++) {
            ChipHead *H = &heads[i];
            if (restore) {
                memcpy(H->S, p, sizeof H->S);
                H->rows_nz[0] = H->rows_nz[1] = 0;
                const uint32_t *u = (const uint32_t *)H->S;
                for (int rw = 0; rw < CHIP_ST; rw++)
                    for (int c = 0; c < CHIP_ST; c++)
                        if (u[rw * CHIP_ST + c] << 1) H->rows_nz[rw / 64] |= 1ULL << (rw % 64);
            } else {
                memcpy(p, H->S, sizeof H->S);
            }
            p += sizeof H->S;
        }
        ChipBank *banks = (ChipBank *)(arena + L->o_banks);
        for (uint32_t i = 0; i < L->n_banks; i++) {
            if (banks[i].kind == 0) continue;
            for (int k = 0; k < 3; k++) {
                if (restore) memcpy(banks[i].win[r[k]], p, sizeof banks[i].win[0]);
                else memcpy(p, banks[i].win[r[k]], sizeof banks[i].win[0]);
                p += sizeof banks[i].win[0];
            }
        }
    }
    return (uint64_t)(p - pack);
}

static uint64_t ck_seq;

/* Card memory to the host, a window at a time, each piece acknowledged. */
static void ck_send(uint64_t at, uint64_t bytes) {
    const uint64_t window = region_addr + Q35_PART_OFFSET + Q35_WINDOW_OFFSET;
    while (bytes) {
        const uint64_t n = bytes < Q35_SLOT_BYTES ? bytes : Q35_SLOT_BYTES, lines = (n + 63) / 64 * 64;
        uint64_t ticket = 0, done = 0;
        while (done < lines) {
            const uint64_t from = at + done, in = from % HUGE_2M;
            uint64_t run = HUGE_2M - in;
            if (run > lines - done) run = lines - done;
            if (run > Q35_DMA_MAX_RUN) run = Q35_DMA_MAX_RUN;
            ticket = q35_dma_copy(arena_phys[from / HUGE_2M] + in, window + done, run);
            done += run;
        }
        while (!q35_dma_finished(ticket)) q35_pause();
        ctl->ck_len = n;
        ctl->ck_seq = ++ck_seq;
        while (ctl->ck_ack != ck_seq) q35_pause();
        at += n;
        bytes -= n;
    }
}

static void checkpoint(ChipEngine *e, uint64_t total, uint64_t n, uint32_t token, uint64_t hash) {
    const ChipHeader *h = (const ChipHeader *)arena;
    ChipLayer *layers = (ChipLayer *)(arena + h->o_layers);
    const uint64_t pack_at = CHIP_PACK_AT(total);
    const uint64_t pack_bytes = state_walk(arena + pack_at, chip_engine_token_no(e) + 1, 0);
    uint64_t all = pack_bytes;
    for (uint32_t il = 0; il < h->layers; il++) {
        const ChipBank *banks = (const ChipBank *)(arena + layers[il].o_banks);
        for (uint32_t i = 0; i < layers[il].n_banks; i++)
            if (banks[i].kind != 0) all += banks[i].mem_bytes;
    }
    ctl->ck_pack = pack_bytes;
    ctl->ck_total = all;
    post(CHIP_EV_CKPT, n, token, hash);
    ck_send(pack_at, pack_bytes);
    for (uint32_t il = 0; il < h->layers; il++) {
        const ChipBank *banks = (const ChipBank *)(arena + layers[il].o_banks);
        for (uint32_t i = 0; i < layers[il].n_banks; i++)
            if (banks[i].kind != 0) ck_send(banks[i].o_mem, banks[i].mem_bytes);
    }
}

static inline uint64_t tsc(void) {
    uint32_t lo, hi;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return (uint64_t)hi << 32 | lo;
}

#define CHECKPOINT_TICKS (1200ULL * 1238094000ULL)  /* 20 minutes */

int main(int argc, char **argv) {
    if (argc != 8) {
        fprintf(stderr, "usage: chip_card CORES TPC PAGES WIDTH PAR_MIN MEMO MAX_TOKENS\n");
        return 2;
    }
    const int cores = atoi(argv[1]), tpc = atoi(argv[2]), width = atoi(argv[4]), par_min = atoi(argv[5]), memo = atoi(argv[6]);
    const uint64_t pages = strtoull(argv[3], NULL, 10), max_tokens = strtoull(argv[7], NULL, 10);
    const int nthreads = cores * tpc;
    if (nthreads > 256 || width > nthreads || width > 255) die("too many threads");
    pin_to(1);
    setvbuf(stdout, NULL, _IOLBF, 0);
    int mem = open("/dev/phihost", O_RDWR);
    if (mem < 0) die("cannot open /dev/phihost");
    uint8_t *part = mmap(NULL, 1u << 20, PROT_READ | PROT_WRITE, MAP_SHARED, mem, (off_t)Q35_PART_OFFSET);
    close(mem);
    if (part == MAP_FAILED) die("cannot map the shared part");
    ctl = (ChipCtl *)part;
    while (ctl->magic != CHIP_LINK_MAGIC) q35_pause();
    arena_alloc(pages);
    region_addr = q35_dma_open(Q35_DMA_CHANNEL);
    printf("card: %llu 2 MiB pages, team %d x %d = %d threads\n", (unsigned long long)pages, cores, tpc, nthreads);
    uint32_t seen = 0;
    const int vpu_ok = chip_vpu_setup(&seen);
    printf("card: denormals on the VPU: %s (2^-99 x 2^-35 x 2^35 has bits %08x)\n", vpu_ok ? "kept, as the host keeps them" : "NOT KEPT", seen);
    ctl->card_ready = 1;

    /* the image, a piece at a time through the first window slot */
    uint64_t done = 0;
    uint64_t total = 0, resume = 0;
    for (;;) {
        while (ctl->cmd_seq == done) q35_pause();
        done++;
        const uint64_t op = ctl->cmd_op;
        if (op == CHIP_OP_BLOB) {
            const uint64_t at = ctl->cmd_a, n = ctl->cmd_b;
            if (at + n > arena_pages * HUGE_2M) die("the image does not fit the pages");
            const uint64_t last = to_arena(region_addr + Q35_PART_OFFSET + Q35_WINDOW_OFFSET, at, n);
            while (!q35_dma_finished(last)) q35_pause();
            ctl->res_seq = done;
        } else if (op == CHIP_OP_START) {
            total = ctl->cmd_a;
            resume = ctl->cmd_b;
            break;
        } else {
            die("unexpected command during the load");
        }
    }
    const uint64_t work_at = (total + 4095) / 4096 * 4096, work_bytes = chip_engine_bytes(nthreads);
    if (work_at + work_bytes > arena_pages * HUGE_2M) die("the engine's arrays do not fit the pages");
    memset(arena + work_at, 0, work_bytes);
    ChipEngine *e = chip_engine_new(arena, arena + work_at, nthreads, width, par_min, memo);
    if (!e) die("this is no image");
    static Arg args[256];
    static pthread_t th[256];
    int tid = 1;
    for (int c = 0; c < cores; c++)
        for (int t = 0; t < tpc; t++) {
            if (c == 0 && t == 0) continue;
            args[tid].e = e;
            args[tid].tid = tid;
            args[tid].cpu = 1 + 4 * c + t;
            if (pthread_create(&th[tid], NULL, member_main, &args[tid])) die("pthread_create");
            tid++;
        }
    printf("card: empty phase, ticks at 1.238 GHz: width %d: %llu, width 21: %llu, width 5: %llu\n", nthreads, (unsigned long long)chip_engine_phase_ticks(e, 1000, nthreads),
           (unsigned long long)chip_engine_phase_ticks(e, 1000, nthreads < 21 ? nthreads : 21), (unsigned long long)chip_engine_phase_ticks(e, 1000, nthreads < 5 ? nthreads : 5));
    int all_ok = 1;
    for (int t = 1; t < nthreads; t++) all_ok = all_ok && args[t].vpu_ok;
    printf("card: denormals kept on every thread: %s\n", all_ok ? "yes" : "NO");
    ctl->res_seq = done;

    const ChipHeader *h = (const ChipHeader *)arena;
    const uint32_t byte0 = h->byte_token;
    uint32_t token = h->go_token;
    uint64_t hash = 0xCBF29CE484222325ULL;
    hash = (hash ^ token) * 0x100000001B3ULL;
    uint64_t n = 1;
    if (resume) {
        n = ctl->res_n;
        token = (uint32_t)ctl->res_token;
        hash = ctl->res_hash;
        state_walk(arena + CHIP_PACK_AT(total), 1, 1);
        chip_engine_wake(e);
        printf("card: resumed at %llu tokens\n", (unsigned long long)n);
    }
    uint64_t stats[CHIP_STATS];
    uint64_t last_ck = tsc();
    int stopping = 0, want_ck = 0;
    while (n < max_tokens) {
        const uint32_t next = chip_engine_step(e, token);
        hash = (hash ^ next) * 0x100000001B3ULL;
        n++;
        if ((n & 4095) == 0) {
            post(CHIP_EV_HASH, n, hash, 0);
            uint32_t dims[8], bits[8];
            const int k = chip_engine_strays(e, dims, bits, 8);
            for (int i = 0; i < k; i++) post(CHIP_EV_LOG, dims[i], bits[i], 0);
            if ((n & 65535) == 0) {
                chip_engine_stats(e, stats);
                for (int i = 0; i < CHIP_STATS; i++) ctl->stats[i] = stats[i];
            }
            if (ctl->stop) stopping = 1;
            if (tsc() - last_ck > CHECKPOINT_TICKS) want_ck = 1;
        }
        if (next >= byte0 && next < byte0 + 512) {
            const uint32_t b = next - byte0;
            post(b == 511 ? CHIP_EV_READY : CHIP_EV_BYTE, n, b, hash);
            if (b == 511) {
                /* the host answers with what goes into VM, or with stop */
                while (ctl->cmd_seq == done && !ctl->stop) q35_pause();
                if (ctl->stop) {
                    stopping = 1;
                } else {
                    done++;
                    if (ctl->cmd_op == CHIP_OP_POKE) {
                        const uint64_t pokes = ctl->cmd_a;
                        if (pokes > CHIP_POKES) die("too many words for VM");
                        const volatile ChipPoke *list = (const volatile ChipPoke *)(part + CHIP_POKE_OFFSET);
                        int32_t *vm = chip_engine_vm(e);
                        for (uint64_t i = 0; i < pokes; i++) {
                            const uint32_t w = list[i].word;
                            const int32_t v = list[i].value;
                            if (list[i].add) vm[w] += v;
                            else vm[w] = v;
                        }
                    }
                    ctl->res_seq = done;
                }
            }
        }
        token = next;
        /* between tokens, after every event of this one: what the host has seen and what is saved agree */
        if (want_ck || stopping) {
            checkpoint(e, total, n, token, hash);
            want_ck = 0;
            last_ck = tsc();
            if (stopping) break;
        }
    }
    chip_engine_stats(e, stats);
    for (int i = 0; i < CHIP_STATS; i++) ctl->stats[i] = stats[i];
    post(CHIP_EV_END, n, hash, 0);
    chip_engine_stop(e);
    q35_dma_close();
    printf("card: stopped at %llu tokens, hash %016llx\n", (unsigned long long)n, (unsigned long long)hash);
    return 0;
}
