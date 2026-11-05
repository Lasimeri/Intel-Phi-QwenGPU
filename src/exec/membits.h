/* The element forms of the memories a bank addresses: how an address and a width select bytes of a memory held in one
 * piece, and how a written value lands. Shared by the exact evaluator (src/circuit/eval.cc) and the file's forward
 * (src/exec/tforward.cc): both must select the same bytes for the same address, and this is the one place that says
 * which. A kind is a property of the bank in the file (chip.bank.K.info[0]); nothing here knows what the bytes mean. */
#ifndef EXEC_MEMBITS_H
#define EXEC_MEMBITS_H

#include <stdint.h>
#include <string.h>

/* the kinds, in the order the circuit's MemKind has them (checked by a static assert in eval.cc) */
enum MemBitsKind {
    MB_WORDS = 0,  /* 32-bit words, addressed by word */
    MB_ABLK = 1,   /* blocks of 36 bytes (32 bytes then a 32-bit word): address block << 6 | j, j < 32 a byte, j = 32 the word;
                      a whole block is 288 bits */
    MB_HALF = 2,   /* 16-bit elements */
    MB_BYTE = 3,   /* bytes */
    MB_WBLK = 4,   /* read-only blocks of 34 bytes (a 16-bit word then 32 bytes): block << 6 | j, j < 32 a byte, else the
                      word; a whole block is 272 bits */
    MB_ROM = 5,    /* read-only blocks of 68 bytes, 544 bits, addressed by block */
    MB_REG = 6,    /* words */
    MB_UREG = 7,   /* words */
    MB_F16 = 8,    /* read-only words, addressed by the low 16 bits */
    MB_SCR = 9,    /* words */
    MB_COUNT = 10
};
enum MemBitsWidth { MBW_BYTE = 8, MBW_HALF = 16, MBW_WORD = 32, MBW_ABLK = 288, MBW_WBLK = 272, MBW_ROMBLK = 544 };

static inline void membits_unpack(const uint8_t *src, int nbytes, uint8_t *bits) {
    for (int i = 0; i < nbytes; i++)
        for (int b = 0; b < 8; b++) bits[8 * i + b] = (uint8_t)(src[i] >> b & 1);
}

/* Fills `width` bits (0/1 bytes) from the element at `addr`; returns how many bits the element has (the rest are 0). An
 * empty memory reads as zeros. Addresses wrap at the memory's size, as the evaluator's always have. */
static inline uint32_t membits_read(int kind, int width, uint32_t addr, const uint8_t *mem, uint64_t nbytes, uint8_t *bits) {
    memset(bits, 0, (size_t)width);
    switch (kind) {
    case MB_WORDS: case MB_REG: case MB_UREG: case MB_SCR: {
        const uint64_t words = nbytes / 4;
        if (words) membits_unpack(mem + (addr % words) * 4, 4, bits);
        return 32;
    }
    case MB_F16: {
        const uint64_t words = nbytes / 4;
        if (words) membits_unpack(mem + ((addr & 0xFFFFu) % words) * 4, 4, bits);
        return 32;
    }
    case MB_HALF: {
        const uint64_t n = nbytes / 2;
        if (n) membits_unpack(mem + (addr % n) * 2, 2, bits);
        return 16;
    }
    case MB_BYTE:
        if (nbytes) membits_unpack(mem + addr % nbytes, 1, bits);
        return 8;
    case MB_ABLK: {
        const uint64_t blocks = nbytes / 36;
        if (!blocks) return (uint32_t)width;
        if (width == MBW_ABLK) {
            membits_unpack(mem + (addr % blocks) * 36, 36, bits);
            return 288;
        }
        const uint64_t blk = (addr >> 6) % blocks, j = addr & 63;
        if (width == MBW_BYTE) {
            if (j < 32) membits_unpack(mem + blk * 36 + j, 1, bits);
            return 8;
        }
        membits_unpack(mem + blk * 36 + 32, 4, bits);
        return 32;
    }
    case MB_WBLK: {
        const uint64_t blocks = nbytes / 34;
        if (!blocks) return (uint32_t)width;
        if (width == MBW_WBLK) {
            membits_unpack(mem + (addr % blocks) * 34, 34, bits);
            return 272;
        }
        const uint64_t blk = (addr >> 6) % blocks, j = addr & 63;
        if (width == MBW_BYTE) {
            if (j < 32) membits_unpack(mem + blk * 34 + 2 + j, 1, bits);
            return 8;
        }
        membits_unpack(mem + blk * 34, 2, bits);
        return 16;
    }
    case MB_ROM: {
        const uint64_t blocks = nbytes / 68;
        if (blocks) membits_unpack(mem + (addr % blocks) * 68, 68, bits);
        return 544;
    }
    default: return (uint32_t)width;
    }
}

/* Lands `value` (its low `width` bits) at `addr`; returns 0, or -1 for a read-only kind. */
static inline int membits_write(int kind, int width, uint32_t addr, uint32_t value, uint8_t *mem, uint64_t nbytes) {
    switch (kind) {
    case MB_WORDS: case MB_REG: case MB_UREG: case MB_SCR: {
        const uint64_t words = nbytes / 4;
        if (words) memcpy(mem + (addr % words) * 4, &value, 4);
        return 0;
    }
    case MB_HALF: {
        const uint64_t n = nbytes / 2;
        if (n) {
            const uint16_t h = (uint16_t)value;
            memcpy(mem + (addr % n) * 2, &h, 2);
        }
        return 0;
    }
    case MB_BYTE:
        if (nbytes) mem[addr % nbytes] = (uint8_t)value;
        return 0;
    case MB_ABLK: {
        const uint64_t blocks = nbytes / 36;
        if (!blocks) return 0;
        const uint64_t blk = (addr >> 6) % blocks, j = addr & 63;
        if (width == MBW_BYTE) {
            if (j < 32) mem[blk * 36 + j] = (uint8_t)value;
        } else {
            memcpy(mem + blk * 36 + 32, &value, 4);
        }
        return 0;
    }
    default: return -1;
    }
}

#endif
