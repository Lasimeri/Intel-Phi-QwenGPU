#define _GNU_SOURCE
#include "q35_dma.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <unistd.h>

#include "q35_engine.h"

enum {
    REG_CONTROL = 0x00 / 4,        /* per channel, as 32-bit words */
    REG_HEAD = 0x04 / 4,
    REG_TAIL = 0x08 / 4,
    REG_RING_LO = 0x14 / 4,
    REG_RING_HI = 0x18 / 4,
    REG_ERROR_MASK = 0x30 / 4,
    REG_DCR = 0x280 / 4,           /* one word for all channels: 2 bits each, owner and enable */
    CHANNEL_WORDS = 0x40 / 4,
    RING_ENTRIES = 256,
    ENTRIES_PER_COPY = 4
};
#define DMA_REGISTERS 0x08007DA000ULL /* SBOX + 0xA000 */
#define HOST_REGION_INFO 0x10000000ULL /* the region header: "PHIR", then at +32 the host window's address */
#define INT_MASKS 0x03000000u          /* no handler on the card side */
#define ADDR_BITS ((1ULL << 40) - 1)
#define OP_MEMCOPY_DST (1ULL << 60)
#define OP_STATUS (2ULL << 60)
#define PAGE_2M (2ULL << 20)

typedef struct {
    uint64_t a, b;
} Entry;

static volatile uint32_t *chan_regs, *all_regs;
static uint32_t dcr_bits;
static Entry *ring;
static uint64_t ring_phys;
static volatile uint64_t *completed;  /* the engine writes the last finished ticket here */
static uint64_t issued;
static uint32_t head;

static void fail(const char *what) {
    fprintf(stderr, "q35_dma: %s\n", what);
    exit(2);
}

/* KNC has no mfence; a locked operation orders everything */
static inline void full_fence(void) { __asm__ volatile("lock addq $0, (%%rsp)" ::: "memory"); }

static uint64_t physical(const void *virt) {
    int fd = open("/proc/self/pagemap", O_RDONLY);
    uint64_t e = 0;
    if (fd < 0 || pread(fd, &e, 8, (off_t)((uintptr_t)virt / 4096) * 8) != 8 || !(e >> 63)) fail("no physical address for the ring");
    close(fd);
    return (e & ((1ULL << 55) - 1)) * 4096 + ((uintptr_t)virt & 4095);
}

uint64_t q35_dma_open(int channel) {
    char name[64];
    snprintf(name, sizeof name, "/tmp/phi-cdma%d.lock", channel); /* the stack's convention, so programs do not collide */
    int lock = open(name, O_CREAT | O_RDWR, 0600);
    if (lock < 0 || flock(lock, LOCK_EX | LOCK_NB)) fail("another program holds this DMA channel");
    int mem = open("/dev/mem", O_RDWR | O_SYNC);
    if (mem < 0) fail("cannot open /dev/mem");
    all_regs = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, mem, (off_t)DMA_REGISTERS);
    const volatile uint8_t *info = mmap(NULL, 4096, PROT_READ, MAP_SHARED, mem, (off_t)HOST_REGION_INFO);
    close(mem);
    if (all_regs == MAP_FAILED || info == MAP_FAILED) fail("cannot map the registers");
    if (memcmp((const void *)info, "PHIR", 4)) fail("no region header");
    uint64_t region = 0;
    memcpy(&region, (const void *)(info + 32), 8);
    if (!region) fail("the host gave no shared region");
    uint8_t *page = mmap(NULL, PAGE_2M, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE, -1, 0);
    if (page == MAP_FAILED) fail("no 2 MiB page for the ring");
    memset(page, 0, PAGE_2M);
    ring = (Entry *)page;
    ring_phys = physical(page);
    completed = (volatile uint64_t *)(page + 4096);
    chan_regs = all_regs + channel * CHANNEL_WORDS;
    dcr_bits = 3u << (2 * channel);
    /* owner card and disabled while it is set up; the other channels' bits stay as they are */
    const uint32_t before = all_regs[REG_DCR];
    all_regs[REG_DCR] = before & ~dcr_bits;
    chan_regs[REG_CONTROL] = INT_MASKS;
    chan_regs[REG_ERROR_MASK] = 0;
    chan_regs[REG_RING_LO] = (uint32_t)ring_phys;
    chan_regs[REG_RING_HI] = (RING_ENTRIES << 4) | (uint32_t)((ring_phys >> 32) & 0xF);
    /* the engine resumes from where its tail stands, so start on the next whole copy */
    head = (chan_regs[REG_TAIL] % RING_ENTRIES + ENTRIES_PER_COPY - 1) / ENTRIES_PER_COPY * ENTRIES_PER_COPY % RING_ENTRIES;
    full_fence();
    chan_regs[REG_HEAD] = head;
    all_regs[REG_DCR] = (all_regs[REG_DCR] & ~dcr_bits) | (2u << (2 * channel));
    for (long spin = 0; spin < 100000000L && chan_regs[REG_TAIL] % RING_ENTRIES != head; spin++) {}
    return region;
}

uint64_t q35_dma_copy(uint64_t src, uint64_t dst, uint64_t bytes) {
    while (issued - *completed >= RING_ENTRIES / ENTRIES_PER_COPY - 1) q35_pause();
    const uint64_t ticket = ++issued;
    volatile Entry *e = (volatile Entry *)&ring[head];
    e[0].a = (src & ADDR_BITS) | ((bytes / 64) << 46);
    e[0].b = (dst & ADDR_BITS) | OP_MEMCOPY_DST;
    e[1].a = ticket;
    e[1].b = ((ring_phys + 4096) & ADDR_BITS) | OP_STATUS;
    e[2].a = e[2].b = e[3].a = e[3].b = 0;
    head = (head + ENTRIES_PER_COPY) % RING_ENTRIES;
    full_fence();
    chan_regs[REG_HEAD] = head;
    return ticket;
}

int q35_dma_finished(uint64_t ticket) { return *completed >= ticket; }

void q35_dma_close(void) { all_regs[REG_DCR] &= ~(dcr_bits & 0xAAAAAAAAu); }
