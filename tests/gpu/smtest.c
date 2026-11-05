/* The reference machine's smoke test: hand-assembled blocks, known answers. Host, test only. Checks the encodings round
 * trip, lane and uniform live-outs, stv into every bank, q8dot against a block computed here, wsum/wmax, udiv/usqrt,
 * shifts, the uniform loop and the queue (launch, branch, emit, halt, GO). Prints the token count and hash of the run. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../src/gpu/sm_ref.h"

static int fails;
#define CHECK(cond, ...) do { if (!(cond)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static SmInstr I(int op) {
    SmInstr i;
    memset(&i, 0, sizeof i);
    i.op = (uint8_t)op;
    i.valid = 1;
    i.rd = i.ra = i.rb = i.rc = SM_ZERO;
    return i;
}
static SmInstr add_imm(int rd, int ra, uint32_t imm) { SmInstr i = I(SM_ADD); i.rd = rd; i.ra = ra; i.ub = SM_SRC_IMM; i.imm = imm; return i; }
static SmInstr add_rr(int rd, int ra, int rb) { SmInstr i = I(SM_ADD); i.rd = rd; i.ra = ra; i.rb = rb; return i; }
static SmInstr uadd_imm(int rd, int ra, uint32_t imm) { SmInstr i = add_imm(rd, ra, imm); i.ua = 1; i.ud = 1; return i; }
static SmInstr ld(int rd, int ra, uint32_t imm, int bank) { SmInstr i = I(SM_LD); i.rd = rd; i.ra = ra; i.imm = imm; i.bank = bank; return i; }
static SmInstr stv(int ra, int rb, uint32_t imm, int bank) { SmInstr i = I(SM_STV); i.ra = ra; i.rb = rb; i.imm = imm; i.bank = bank; return i; }
static SmInstr ustv(int rb, uint32_t imm, int bank) { SmInstr i = stv(SM_ZERO, rb, imm, bank); i.ua = 1; i.ub = SM_SRC_UNIFORM; i.ud = 1; return i; }
static SmInstr tid(int rd) { SmInstr i = I(SM_TID); i.rd = rd; return i; }
static SmInstr shift(int op, int rd, int ra, int sh) { SmInstr i = I(op); i.rd = rd; i.ra = ra; i.sh = sh; return i; }
static SmInstr and_imm(int rd, int ra, uint32_t imm) { SmInstr i = I(SM_AND); i.rd = rd; i.ra = ra; i.ub = SM_SRC_IMM; i.imm = imm; return i; }

static void put_block(SmBlock *b, const SmInstr *ins, int n, SmLive lv) {
    memset(b, 0, sizeof *b);
    for (int s = 0; s < n; s++) sm_encode(&ins[s], b->rows[s]);
    b->live = sm_live_encode(&lv);
}
static SmLive live(int nlive, int s0, int s1, int s2, int s3, int branch, int target) {
    SmLive l;
    memset(&l, 0, sizeof l);
    l.nlive = nlive;
    l.src[0] = s0; l.src[1] = s1; l.src[2] = s2; l.src[3] = s3;
    l.branch = branch;
    l.target = target;
    return l;
}

int main(void) {
    /* encodings round trip over every field */
    {
        SmInstr a = I(SM_MULSH), b;
        memset(&b, 0, sizeof b);
        a.rd = 3; a.ra = 14; a.rb = 7; a.rc = 9; a.pd = 5; a.cmp = SM_CMP_GE; a.sgn = 1; a.ub = SM_SRC_UNIFORM; a.ua = 1; a.uc = 1; a.ud = 1;
        a.gvalid = 1; a.g = 6; a.gneg = 1; a.bank = SM_BANK_C; a.sh = 45; a.imm = 0xDEADBEEF;
        uint16_t rows[5];
        sm_encode(&a, rows);
        sm_decode(rows, &b);
        CHECK(memcmp(&a, &b, sizeof a) == 0, "instruction encoding does not round trip");
        SmLive l = live(3, 5, 2, 4, 0, SM_BR_COND, 4095), l2;
        memset(&l2, 0, sizeof l2);
        l.stage = 3; l.neg = 1; l.any = 1; l.lwpb = 3;
        sm_live_decode(sm_live_encode(&l), &l2);
        CHECK(memcmp(&l, &l2, sizeof l) == 0, "live encoding does not round trip");
    }

    SmImage img;
    memset(&img, 0, sizeof img);
    uint8_t classes[SM_STAGES] = {127, 127, 127, 127, 127, 127};  /* the blocks below are hand-placed: every unit in every stage */
    memcpy(img.classes, classes, sizeof classes);
    img.m_words = 1 << 16;
    img.m = calloc(img.m_words, 4);
    img.a_blocks = 64;
    img.a = calloc(img.a_blocks, 36);
    img.s_elems = 4096;
    img.s = calloc(img.s_elems, 2);
    img.w_blocks = 64;
    uint8_t *w = calloc(img.w_blocks, 34);
    img.w = w;
    img.c_bytes = 4096;
    img.c = calloc(img.c_bytes, 1);
    int64_t *tab = calloc(65536, 8);
    img.f16_q24 = tab;
    tab[0x3C00] = 1 << 24;  /* an f16 of 1.0 in Q.24 */
    tab[0x3800] = 1 << 23;

    /* data: M[1000 + i] = i, M[2000 + i] = 3 i; A block 0 = 1..32, scale Q.24 of 1.0; W blocks 0..31: block L all +1 (even L) or -1 (odd), scale 1.0 */
    for (int i = 0; i < 64; i++) {
        img.m[1000 + i] = i;
        img.m[2000 + i] = 3 * i;
    }
    for (int j = 0; j < 32; j++) img.a[j] = (uint8_t)(int8_t)(j + 1);
    int32_t xs = 1 << 24;
    memcpy(img.a + 32, &xs, 4);
    for (int L = 0; L < 32; L++) {
        uint16_t h = 0x3C00;
        memcpy(w + L * 34, &h, 2);
        for (int j = 0; j < 32; j++) w[L * 34 + 2 + j] = (uint8_t)(int8_t)(L & 1 ? -1 : 1);
    }

    SmBlock rom[9];
    memset(rom, 0, sizeof rom);
    /* kernel 0, block 0: r1 = tid; r2 = M[1000 + r1]; r3 = M[2000 + r1]; r4 = r2 + r3; stv M[3000 + r1] = r4; live-outs r1 r2 r3 and the stv */
    {
        SmInstr ins[5] = {tid(1), ld(2, 1, 1000, SM_BANK_M), ld(3, 1, 2000, SM_BANK_M), add_rr(4, 2, 3), stv(1, 4, 3000, SM_BANK_M)};
        put_block(&rom[0], ins, 5, live(4, 0, 1, 2, 4, SM_BR_NEXT, 0));
    }
    /* block 1: r5 = 0; q8dot r5, [r1 + 0], u0 (A block 0); q8dot r5, [r1 + 32], u0 (blocks 32..63 are zero); stv M[4000 + r1] = r5 */
    {
        SmInstr ins[4];
        ins[0] = add_imm(5, SM_ZERO, 0);
        ins[1] = I(SM_Q8DOT); ins[1].rd = 5; ins[1].ra = 1; ins[1].rb = 0; ins[1].ub = SM_SRC_UNIFORM; ins[1].imm = 0; ins[1].rc = SM_ZERO;
        ins[2] = I(SM_Q8DOT); ins[2].rd = 5; ins[2].ra = 1; ins[2].rb = 0; ins[2].ub = SM_SRC_UNIFORM; ins[2].imm = 32; ins[2].rc = 5;
        ins[3] = stv(1, 5, 4000, SM_BANK_M);
        put_block(&rom[1], ins, 4, live(1, 3, 0, 0, 0, SM_BR_NEXT, 0));
    }
    /* block 2: u1 = wsum r2; u2 = wmax r3; stv M[5002] = u2; live-outs: u1 (read by the next block: a reduction is late, a
       scalar may not read it in the same block) and the stv */
    {
        SmInstr ins[3];
        ins[0] = I(SM_WSUM); ins[0].rd = 1; ins[0].ra = 2; ins[0].ud = 1;
        ins[1] = I(SM_WMAX); ins[1].rd = 2; ins[1].ra = 3; ins[1].ud = 1;
        ins[2] = ustv(2, 5002, SM_BANK_M);
        put_block(&rom[2], ins, 3, live(2, 0, 2, 0, 0, SM_BR_NEXT, 0));
    }
    /* block 3: u3 = udiv (0 : u1) / 7; stv M[5000] = u3 */
    {
        SmInstr ins[2];
        ins[0] = I(SM_UDIV); ins[0].rd = 3; ins[0].ra = 1; ins[0].ua = 1; ins[0].ub = SM_SRC_IMM; ins[0].imm = 7; ins[0].rc = SM_ZERO; ins[0].uc = 1; ins[0].ud = 1;
        ins[1] = ustv(3, 5000, SM_BANK_M);
        put_block(&rom[3], ins, 2, live(1, 1, 0, 0, 0, SM_BR_NEXT, 0));
    }
    /* block 4: u5 = 1000000; u4 = usqrt (0 : u5); stv M[5001] = u4; stv S[r1] = r3; stv C[r1] = r2 */
    {
        SmInstr ins[5];
        ins[0] = uadd_imm(5, SM_ZERO, 1000000);
        ins[1] = I(SM_USQRT); ins[1].rd = 4; ins[1].ra = 5; ins[1].ua = 1; ins[1].rc = SM_ZERO; ins[1].uc = 1; ins[1].ud = 1;
        ins[2] = ustv(4, 5001, SM_BANK_M);
        ins[3] = stv(1, 3, 0, SM_BANK_S);
        ins[4] = stv(1, 2, 0, SM_BANK_C);
        put_block(&rom[4], ins, 5, live(3, 2, 3, 4, 0, SM_BR_NEXT, 0));
    }
    /* block 5: A8[(1 + (r1 >> 5)) * 64 + (r1 & 31)] = r1 + 100: r6 = r1 >> 5; r7 = r6 << 6; r8 = r1 & 31; r9 = r7 + r8; r10 = r1 + 100; stv A8[r9 + 64] = r10; end */
    {
        SmInstr ins[6] = {shift(SM_SHR, 6, 1, 5), shift(SM_SHL, 7, 6, 6), and_imm(8, 1, 31), add_rr(9, 7, 8), add_imm(10, 1, 100), stv(9, 10, 64, SM_BANK_A8)};
        put_block(&rom[5], ins, 6, live(1, 5, 0, 0, 0, SM_BR_END, 0));
    }
    /* kernel 1, block 6: u6 = u6 + 1; setp p0 = u6 < 3; cond -> block 6 while p0 (lane 0's predicate) */
    {
        SmInstr ins[2];
        ins[0] = uadd_imm(6, 6, 1);
        ins[1] = I(SM_SETP); ins[1].pd = 0; ins[1].ra = 6; ins[1].ua = 1; ins[1].ub = SM_SRC_IMM; ins[1].imm = 3; ins[1].cmp = SM_CMP_LT; ins[1].sgn = 0;
        SmLive lv = live(1, 0, 0, 0, 0, SM_BR_COND, 6);
        lv.stage = 1;
        put_block(&rom[6], ins, 2, lv);
    }
    /* block 7: u7 = M[6000]; u7 = u7 - 1; stv M[6000] = u7; end */
    {
        SmInstr ins[3];
        ins[0] = ld(7, SM_ZERO, 6000, SM_BANK_M); ins[0].ua = 1; ins[0].ud = 1;
        ins[1] = I(SM_SUB); ins[1].rd = 7; ins[1].ra = 7; ins[1].ua = 1; ins[1].ub = SM_SRC_IMM; ins[1].imm = 1; ins[1].ud = 1;
        ins[2] = ustv(7, 6000, SM_BANK_M);
        put_block(&rom[7], ins, 3, live(1, 2, 0, 0, 0, SM_BR_END, 0));
    }
    img.rom = rom;
    img.nblk = 9;

    /* the queue: launch kernel 0 (2 warps), launch kernel 1 (1 warp), emit M[5000], branch to descriptor 1 if M[6000] != 0, halt */
    img.m[SM_M_QUEUE + 0] = (int32_t)sm_qword(SM_Q_LAUNCH, 0, 2, 0, 0);
    img.m[SM_M_QUEUE + 2] = (int32_t)sm_qword(SM_Q_LAUNCH, 6, 1, 0, 0);
    img.m[SM_M_QUEUE + 4] = (int32_t)sm_qword(SM_Q_EMIT, 0, 0, 0, 0);
    img.m[SM_M_QUEUE + 5] = 5000;
    img.m[SM_M_QUEUE + 6] = (int32_t)sm_qword(SM_Q_BRANCH, 0, 0, 0, 1);
    img.m[SM_M_QUEUE + 7] = 6000;
    img.m[SM_M_QUEUE + 8] = (int32_t)sm_qword(SM_Q_HALT, 0, 0, 0, 0);
    /* descriptor 5, where GO resumes: an unconditional jump to 1 (a branch on a word that holds 1) */
    img.m[SM_M_QUEUE + 10] = (int32_t)sm_qword(SM_Q_BRANCH, 0, 0, 0, 1);
    img.m[SM_M_QUEUE + 11] = 6001;
    img.m[6001] = 1;
    img.m[6000] = 1;  /* kernel 1 takes one off it: the branch after the emit is not taken in the first turn */

    SmMachine *mc = calloc(1, sizeof *mc);
    sm_init(mc, &img);
    int rc = sm_run(mc, 1000000);
    CHECK(rc == 0, "run ended with %d: %s", rc, mc->fault_what);
    for (int t = 0; t < 64; t++) CHECK(img.m[3000 + t] == 4 * t, "vector add at %d: %d", t, img.m[3000 + t]);
    /* q8dot: lane L of warp 0 reads W block L (+-1, scale 1.0) against A block 0 (1..32, sum 528, scale 1.0): Q.16 of +-528, then
       block 32 + L against A block 32 (both zero, the immediate advances both); warp 1 reads block 32 + L (zero) then 64 + L = L
       (wrapped) against A block 32 (zero): 0 */
    for (int t = 0; t < 32; t++) CHECK(img.m[4000 + t] == (t & 1 ? -(528 << 16) : 528 << 16), "q8dot lane %d: %d", t, img.m[4000 + t]);
    for (int t = 32; t < 64; t++) CHECK(img.m[4000 + t] == 0, "q8dot warp 1 lane %d: %d", t, img.m[4000 + t]);
    /* from warp 1, the last to run: r2 = 32..63 (sum 1520), r3 = 96..189 (max 189); 1520 / 7 = 217 */
    CHECK(img.m[5000] == 217, "udiv: %d", img.m[5000]);
    CHECK(img.m[5002] == 189, "wmax: %d", img.m[5002]);
    CHECK(img.m[5001] == 1000, "usqrt: %d", img.m[5001]);
    for (int t = 0; t < 64; t++) {
        CHECK(img.s[t] == (int16_t)(3 * t), "S[%d] = %d", t, img.s[t]);
        CHECK(img.c[t] == (int8_t)t, "C[%d] = %d", t, img.c[t]);
    }
    for (int t = 0; t < 64; t++) {
        int8_t got = (int8_t)img.a[(1 + t / 32) * 36 + t % 32];
        CHECK(got == (int8_t)(t + 100), "A8 at %d: %d", t, got);
    }
    CHECK(mc->ureg[0][6] == 3, "loop counter: %u", mc->ureg[0][6]);
    CHECK(mc->halted == 1, "not halted");
    /* tokens, lane by lane: NEXT; kernel 0 per warp: block 0 EXEC + 32 x (4 x 2), block 1 EXEC + 32 x 2, block 2 EXEC + 32 x (2 reductions x 2)
       + 2 (the stv; the reductions' results are in the lanes' partials), block 3 EXEC + 32 LSTEP + 2, block 4 EXEC + 32 x (2 x 2) + 2, block 5
       EXEC + 32 x 2; WARP(1) between the two warps; kernel 1: NEXT + 3 x (EXEC + 32 LSTEP + 2) + (EXEC + 32 LSTEP + 2); emit NEXT EMIT VAL VAL;
       branch NEXT; halt NEXT HALT */
    const uint64_t per_warp = (1 + 256) + (1 + 64) + (1 + 128 + 2) + (1 + 32 + 2) + (1 + 128 + 2) + (1 + 64);
    const uint64_t want = 1 + 2 * per_warp + 1 + (1 + 3 * 35 + 35) + 4 + 1 + 2;
    CHECK(mc->sink.n == want, "token count %llu, want %llu", (unsigned long long)mc->sink.n, (unsigned long long)want);
    const uint64_t n1 = mc->sink.n;
    /* the host's turn: M[6000] = 2, GO: kernel 1 runs twice more (each run takes one off M[6000]), then the queue halts */
    img.m[6000] = 2;
    sm_go(mc);
    rc = sm_run(mc, 10000);
    CHECK(rc == 0 && mc->halted, "second turn did not halt: rc %d %s", rc, mc->fault_what);
    CHECK(mc->ureg[0][6] == 5, "loop counter after the second turn (u6 persists, the loop runs once a launch): %u", mc->ureg[0][6]);
    CHECK(mc->launches == 4, "launches %llu", (unsigned long long)mc->launches);
    CHECK(img.m[6000] == 0, "M[6000] = %d", img.m[6000]);
    /* second turn: GO; descriptor 5 NEXT (the jump); kernel 1 (71: NEXT, one loop record of 35, u6 >= 3, block 7's 35); emit (4); branch (1, taken);
       kernel 1 (71); emit (4); branch (1); halt (2) */
    CHECK(mc->sink.n - n1 == 1 + 1 + 71 + 4 + 1 + 71 + 4 + 1 + 2, "second turn tokens: %llu", (unsigned long long)(mc->sink.n - n1));
    printf("host: smtest: %llu tokens, hash %016llx, %llu records, %llu instructions, %llu launches, %d failures\n", (unsigned long long)mc->sink.n,
           (unsigned long long)mc->sink.hash, (unsigned long long)mc->records, (unsigned long long)mc->instrs, (unsigned long long)mc->launches, fails);
    return fails ? 1 : 0;
}
