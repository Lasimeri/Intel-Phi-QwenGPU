// The SIMT machine (docs/machine.md) as a circuit of the 35B's layer form: one forward pass over one token of the
// machine's stream computes one lane's stages of the current block and predicts the next token. The reference it must
// equal, token for token, is src/gpu/sm_ref.c.
#ifndef GPU_CIRCUIT_SM_CIRC_H
#define GPU_CIRCUIT_SM_CIRC_H

#include "gate.h"

struct SmCircStats {
    uint64_t units = 0, nodes = 0, heads = 0, banks = 0, dims = 0;
    int layers = 0;
};

// The stage classes decide which units each stage holds (docs/machine.md): the circuit builds exactly those.
Circ *build_sm_circuit(const uint8_t classes[SM_STAGES], Target target, SmCircStats *stats);

// the scratch memory's words (MEM_SCR): the machine state that is read low in a pass and written high in it
enum ScrWord {
    SCR_LN,      // the lane the datapath computes (0..32; 32 = the uniform final pass)
    SCR_LI,      // the item within the current part
    SCR_ANY,     // the branch predicate over the lanes so far (lanes 1..31)
    SCR_P0,      // lane 0's branch predicate
    SCR_RED0A, SCR_RED0B,          // reduction 0: its value before the lane being computed, and after it (a lane's own
    SCR_RED1A, SCR_RED1B,          // passes must see the "before": the computing pass writes B, the next lane copies B to A)
    SCR_DIV_R0, SCR_DIV_R1,        // udiv: the partial remainder (33 bits)
    SCR_DIV_N0, SCR_DIV_N1,        // udiv: the numerator shifting out, the quotient shifting in (64 bits)
    SCR_SQ_V0, SCR_SQ_V1,          // usqrt: the radicand's remainder (64 bits)
    SCR_SQ_R0, SCR_SQ_R1,          // usqrt: the root so far (33 bits)
    SCR_SCFG,                      // the state unit's constants as the last scfg left them
    SCR_STV0,                      // pending stores: 8 slots x (address, value, flags) = 24 words
    SCR_SUP0 = SCR_STV0 + 24,      // pending state writes: 6 stages x (address, value, valid) = 18 words
    SCR_PREG0 = SCR_SUP0 + 18,     // pending lane register commits: 4 items x (value, rd | valid << 4) = 8 words (the items that
                                   // end with VHI; they land when the lane ends, so the lane's later passes read the block's input)
    SCR_PUREG0 = SCR_PREG0 + 8,    // the same for the uniform items: land at the uniform part's end
    SCR_WORDS = SCR_PUREG0 + 8
};
enum { STV_SLOTS = 8 };  // 4 lane live-outs, 4 uniform live-outs

#endif
