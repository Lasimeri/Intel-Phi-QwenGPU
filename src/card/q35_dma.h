/* A card-owned channel of the coprocessor's DMA engine, driven from user space: copies between the shared host
 * region and card memory. Registers and descriptors are the hardware's, per the Xeon Phi System Software
 * Developer's Guide (328207), section 2.1.8. Runs on the 7120A, as root. */
#ifndef CARD_Q35_DMA_H
#define CARD_Q35_DMA_H

#include <stdint.h>

/* Takes the channel, or exits with a message. Returns the shared host region's address as the card sees it. */
uint64_t q35_dma_open(int channel);
/* Queues a copy of a physically contiguous run, both ends 64 B aligned, whole lines, at most Q35_DMA_MAX_RUN.
 * Waits only if the ring is full. Returns a ticket for q35_dma_finished. */
uint64_t q35_dma_copy(uint64_t src, uint64_t dst, uint64_t bytes);
int q35_dma_finished(uint64_t ticket);
void q35_dma_close(void);

#define Q35_DMA_MAX_RUN (16383ULL * 64)

#endif
