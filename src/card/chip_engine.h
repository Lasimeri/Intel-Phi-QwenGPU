/* The boundary between the VPU code (KNC clang) and the card's main (knc-cc, x87 ABI): pointers and integers only, never a float by value. */
#ifndef CARD_CHIP_ENGINE_H
#define CARD_CHIP_ENGINE_H

#include <stdint.h>

#include "../exec/chip_image.h"

typedef struct ChipEngine ChipEngine;

/* What this card's vector unit does with a denormal: the host keeps them, so the card must too. Returns 1 if it does, and sets the VPU's control
 * word for the calling thread so that it does; every thread calls it. Writes what it saw to *seen. */
int chip_vpu_setup(uint32_t *seen);

/* image: the image, in card memory. work: chip_engine_bytes() of zeroed card memory. width: how many threads a full phase uses; par_min: how many lines
 * make a phase worth its cost; memo: 0 computes every line of every sublayer, 1 only those whose inputs changed. */
uint64_t chip_engine_bytes(int nthreads);
ChipEngine *chip_engine_new(void *image, void *work, int nthreads, int width, int par_min, int memo);
void chip_engine_worker(ChipEngine *e, int tid);
void chip_engine_stop(ChipEngine *e);
/* One token through every layer. Returns the next token. */
uint32_t chip_engine_step(ChipEngine *e, uint32_t token);
int32_t *chip_engine_vm(ChipEngine *e);
uint32_t chip_engine_token_no(const ChipEngine *e);
void chip_engine_wake(ChipEngine *e);
void chip_engine_stats(const ChipEngine *e, uint64_t out[CHIP_STATS]);
uint64_t chip_engine_phase_ticks(ChipEngine *e, int rounds, int width);
/* Output dims no row reads that have reached the output, each reported once: returns how many, fills up to `room`. */
int chip_engine_strays(ChipEngine *e, uint32_t *dims, uint32_t *bits, int room);
void q35_pause(void);

#endif
