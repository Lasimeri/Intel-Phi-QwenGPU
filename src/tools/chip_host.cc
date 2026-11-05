// The host's side of the merged model on the 7120A: opens the model file, builds the card's image from its tensors, moves it into card memory, then only
// gives prompts and reads what comes out. No forward runs here: this program links the file reader, the tokenizer, and the image builder, which loads the
// tensors. No interpreter, no NPU, no circuit, no reference.
//
//   chip_host MODEL.gguf "PROMPT" [--gen N] [--chat] [--hashlog PATH] [--ref REF_HASH_FILE] [--reply PATH] [--started PATH] [--ckpt PATH]
//
// With --ckpt the machine's state is saved there every 20 minutes and on SIGTERM or SIGINT, and a run that finds it there goes on from it.
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>

#include "../exec/chip_image.hpp"
#include "../text/q35_link.h"
#include "../text/tok.h"

extern "C" {
#include "../guest64/q2b64_map.h"
}

namespace {

ChipCtl *ctl;
uint8_t *part;
uint64_t cmd_no;

void post_cmd(uint64_t op, uint64_t a, uint64_t b) {
    ctl->cmd_op = op;
    ctl->cmd_a = a;
    ctl->cmd_b = b;
    __sync_synchronize();
    ctl->cmd_seq = ++cmd_no;
}

void wait_res() {
    for (long i = 0; ctl->res_seq != cmd_no; i++) {
        usleep(50);
        if (i > 20 * 60 * 20000) {
            std::fputs("host: the card does not answer\n", stderr);
            std::exit(3);
        }
    }
}

// Bytes to a place in card memory, through the first window slot, whole lines.
void to_card(uint64_t at, const uint8_t *from, uint64_t bytes) {
    uint8_t *window = part + Q35_WINDOW_OFFSET;
    while (bytes) {
        const uint64_t n = std::min<uint64_t>(bytes, Q35_SLOT_BYTES);
        std::memcpy(window, from, n);
        const uint64_t lines = (n + 63) / 64 * 64;
        if (lines != n) std::memset(window + n, 0, lines - n);
        post_cmd(CHIP_OP_BLOB, at, lines);
        wait_res();
        at += n;
        from += n;
        bytes -= n;
    }
}

void on_signal(int) {
    if (ctl) ctl->stop = 1;  // the card saves the machine's state and ends
}

// A checkpoint on disk: this, then the pack of windows and states, then VM, VS and the KV cache as the file orders its memories.
struct CkHeader {
    uint64_t magic, n, token, hash, pos, prompt_end, prompt_start, turns, got, bytes, total_bytes, slices, pack_bytes, state_bytes, last_mark_pos;
};
constexpr uint64_t CK_MAGIC = 0x31544E494F504B43ULL;

// The card sends the state a window at a time; it goes to a new file, which then takes the old one's name. Never half a checkpoint under that name.
void save_checkpoint(const std::string &path, const CkHeader &head) {
    const auto t0 = std::chrono::steady_clock::now();
    const std::string tmp = path + ".tmp", prev = path + ".prev";
    FILE *f = path.empty() ? nullptr : std::fopen(tmp.c_str(), "w");
    if (f) std::fwrite(&head, sizeof head, 1, f);
    uint64_t seen = ctl->ck_ack, got = 0;
    const uint64_t want = head.state_bytes;
    const uint8_t *window = part + Q35_WINDOW_OFFSET;
    while (got < want) {
        const uint64_t s = ctl->ck_seq;
        if (s == seen) {
            usleep(50);
            continue;
        }
        const uint64_t n = ctl->ck_len;
        if (f && std::fwrite(window, 1, n, f) != n) {
            std::fputs("host: cannot write the checkpoint\n", stderr);
            std::fclose(f);
            f = nullptr;
        }
        got += n;
        seen = s;
        __sync_synchronize();
        ctl->ck_ack = s;
    }
    if (f) {
        std::fflush(f);
        fsync(fileno(f));
        std::fclose(f);
        std::rename(path.c_str(), prev.c_str());
        std::rename(tmp.c_str(), path.c_str());
        std::printf("host: the machine's state saved at %llu trace tokens, %.0f MB, %.1f s, %s\n", (unsigned long long)head.n, (double)want / 1e6, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), path.c_str());
        std::fflush(stdout);
    }
}

// A checkpoint's bytes to a place in card memory.
void file_to_card(FILE *f, uint64_t at, uint64_t bytes) {
    std::vector<uint8_t> chunk(Q35_SLOT_BYTES);
    while (bytes) {
        const uint64_t n = std::min<uint64_t>(bytes, chunk.size());
        if (std::fread(chunk.data(), 1, n, f) != n) {
            std::fputs("host: the checkpoint is short\n", stderr);
            std::exit(3);
        }
        to_card(at, chunk.data(), n);
        at += n;
        bytes -= n;
    }
}

const char *const STAT_NAMES[] = {"tokens", "embed", "scale", "hn", "snap", "forms", "heads", "banks", "y", "apply", "units", "out", "phases", "lines_a", "lines_f", "lines_y", "par_lines", "hn_writes", "chg_units", "conv", "e_pass", "keyed", "o", "mix", "full_scans", "table", "token"};

void print_stats(uint64_t since[CHIP_STATS]) {
    uint64_t now[CHIP_STATS];
    for (int i = 0; i < CHIP_STATS; i++) now[i] = ctl->stats[i];
    const double toks = (double)(now[0] - since[0]);
    if (toks <= 0) return;
    std::printf("card, per token over %.0f tokens:", toks);
    for (int i = 1; i <= 11; i++) std::printf(" %s %.1f", STAT_NAMES[i], (double)(now[i] - since[i]) / 1238.094 / toks);
    std::printf(" us, conv %.1f us, all %.1f us;", (double)(now[27] - since[27]) / 1238.094 / toks, (double)(now[26] - since[26]) / 1238.094 / toks);
    for (int i = 12; i < 26; i++) std::printf(" %s %.1f", STAT_NAMES[i], (double)(now[i] - since[i]) / toks);
    std::printf(" marks %.1f snap_diff %.1f fast %.3f\n", (double)(now[28] - since[28]) / toks, (double)(now[29] - since[29]) / toks, (double)(now[30] - since[30]) / toks);
    std::memcpy(since, now, sizeof now);
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fputs("usage: chip_host MODEL.gguf PROMPT [--gen N] [--chat] [--hashlog PATH] [--ref REF_HASH_FILE]\n", stderr);
        return 2;
    }
    long long gen = 64;
    bool chat = false;
    FILE *hlog = nullptr, *ref = nullptr, *reply = nullptr;
    const char *started_path = nullptr, *hlog_path = nullptr, *reply_path = nullptr;
    std::string ckpt_path;
    for (int i = 3; i < argc; i++) {
        if (!std::strcmp(argv[i], "--chat")) chat = true;
        else if (i + 1 < argc && !std::strcmp(argv[i], "--gen")) gen = std::atoll(argv[++i]);
        else if (i + 1 < argc && !std::strcmp(argv[i], "--hashlog")) hlog_path = argv[++i];
        else if (i + 1 < argc && !std::strcmp(argv[i], "--ref")) ref = std::fopen(argv[++i], "r");
        else if (i + 1 < argc && !std::strcmp(argv[i], "--reply")) reply_path = argv[++i];
        else if (i + 1 < argc && !std::strcmp(argv[i], "--started")) started_path = argv[++i];
        else if (i + 1 < argc && !std::strcmp(argv[i], "--ckpt")) ckpt_path = argv[++i];
    }
    // A checkpoint there means this run goes on from it, and its logs go on too.
    FILE *ck_in = ckpt_path.empty() ? nullptr : std::fopen(ckpt_path.c_str(), "r");
    CkHeader ck{};
    if (ck_in && (std::fread(&ck, sizeof ck, 1, ck_in) != 1 || ck.magic != CK_MAGIC)) {
        std::fputs("host: that is no checkpoint\n", stderr);
        return 3;
    }
    if (hlog_path) hlog = std::fopen(hlog_path, ck_in ? "a" : "w");
    if (reply_path) reply = std::fopen(reply_path, ck_in ? "a" : "w");
    std::signal(SIGTERM, on_signal);
    std::signal(SIGINT, on_signal);
    const auto t_load = std::chrono::steady_clock::now();
    int shm = open("/dev/shm/phi-hostmem", O_RDWR);
    if (shm < 0) {
        perror("/dev/shm/phi-hostmem");
        return 2;
    }
    part = static_cast<uint8_t *>(mmap(nullptr, Q35_PART_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, shm, static_cast<off_t>(Q35_PART_OFFSET)));
    close(shm);
    if (part == MAP_FAILED) {
        perror("mmap shared part");
        return 2;
    }
    ctl = reinterpret_cast<ChipCtl *>(part);
    std::memset(static_cast<void *>(part), 0, 1u << 20);

    TForward tf(argv[1]);
    Tokenizer tok(tf.file());
    ChipImage image(tf);
    const ChipHeader &h = image.header();
    std::printf("host: the image from the file's tensors: %.1f MB of layers, %llu slices, %llu weight lines, %llu embedding nonzeros, %.2f GB with the memories, %.1f s\n", (double)h.image_bytes / 1e6, (unsigned long long)h.slices, (unsigned long long)h.wlines, (unsigned long long)h.emb_nnz, (double)h.total_bytes / 1e9, std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count());
    std::fflush(stdout);
    __sync_synchronize();
    ctl->magic = CHIP_LINK_MAGIC;
    for (long i = 0; !ctl->card_ready; i++) {
        usleep(1000);
        if (i > 600000) {
            std::fputs("host: the card did not start\n", stderr);
            return 3;
        }
    }
    const auto t_move = std::chrono::steady_clock::now();
    to_card(0, image.data(), image.size());
    for (const ChipImage::Piece &p : image.pieces()) to_card(p.to, p.from, p.bytes);
    if (ck_in) {
        // the state is the model's, whatever the engine's layout: only the file's own shape has to match
        if (ck.slices != h.slices) {
            std::fputs("host: that checkpoint is of another image\n", stderr);
            return 3;
        }
        file_to_card(ck_in, CHIP_PACK_AT(h.total_bytes), ck.pack_bytes);
        for (const ChipImage::Piece &p : image.pieces())
            if (p.kind != 0) file_to_card(ck_in, p.to, p.bytes);
        std::fclose(ck_in);
        ctl->res_n = ck.n;
        ctl->res_token = ck.token;
        ctl->res_hash = ck.hash;
        std::printf("host: going on from the checkpoint at %llu trace tokens\n", (unsigned long long)ck.n);
    }
    std::printf("host: %.2f GB moved into card memory in %.1f s\n", (double)h.total_bytes / 1e9, std::chrono::duration<double>(std::chrono::steady_clock::now() - t_move).count());
    std::fflush(stdout);
    post_cmd(CHIP_OP_START, h.total_bytes, ck.magic ? 1 : 0);
    wait_res();
    if (started_path && !ck.magic) {
        FILE *sf = std::fopen(started_path, "w");
        if (sf) {
            std::fprintf(sf, "%lld\n", (long long)std::time(nullptr));
            std::fclose(sf);
        }
    }
    std::printf("host: card team is up; the run starts\n");
    std::fflush(stdout);

    uint64_t pos = h.system_tokens, prompt_end = pos, prompt_start = pos, seen = 0, last_n = 0;
    int turns = 0, got = 0;
    uint32_t bytes = 0;
    std::string prompt = argv[2];
    uint64_t since[CHIP_STATS];
    std::memset(since, 0, sizeof since);
    auto t0 = std::chrono::steady_clock::now(), t_mark = t0;
    uint64_t n_mark = 0, n_start = 0, ref_lines = 0, last_mark_pos = 1;
    if (ck.magic) {
        pos = ck.pos;
        prompt_end = ck.prompt_end;
        prompt_start = ck.prompt_start;
        turns = (int)ck.turns;
        got = (int)ck.got;
        bytes = (uint32_t)ck.bytes;
        last_mark_pos = ck.last_mark_pos;
        n_mark = n_start = ck.n;
        // the reference's lines this run has already passed
        if (ref) {
            char skip[128];
            for (uint64_t l = 0; l < ck.n / 4096 && std::fgets(skip, sizeof skip, ref); l++) ref_lines++;
        }
        std::printf("host: turn %d: %llu tokens in, positions %llu to %llu; going on\n", turns, (unsigned long long)(prompt_end - prompt_start), (unsigned long long)prompt_start, (unsigned long long)prompt_end);
        std::fflush(stdout);
    }
    for (;;) {
        const uint64_t head = ctl->ev_head;
        if (seen == head) {
            usleep(200);
            continue;
        }
        if (head - seen >= CHIP_EVENTS) {
            std::fputs("host: lost events\n", stderr);
            return 3;
        }
        const volatile ChipEvent &v = ctl->ev[seen % CHIP_EVENTS];
        const uint64_t kind = v.kind, a = v.a, b = v.b, c = v.c;
        seen++;
        if (kind == CHIP_EV_HASH) {
            last_n = a;
            if (hlog) {
                std::fprintf(hlog, "tokens %llu hash %016llx\n", (unsigned long long)a, (unsigned long long)b);
                std::fflush(hlog);
            }
            if (ref) {
                char want[128], got_line[128];
                std::snprintf(got_line, sizeof got_line, "tokens %llu hash %016llx\n", (unsigned long long)a, (unsigned long long)b);
                if (std::fgets(want, sizeof want, ref)) {
                    ref_lines++;
                    if (std::strcmp(want, got_line)) {
                        std::printf("card: THE HASH DIFFERS from the reference at %llu tokens, within the last 4,096: reference %scard      %s", (unsigned long long)a, want, got_line);
                        std::fflush(stdout);
                        ctl->stop = 1;
                    }
                }
            }
            if ((a & 65535) == 0) {
                const auto now = std::chrono::steady_clock::now();
                const double el = std::chrono::duration<double>(now - t_mark).count(), all = std::chrono::duration<double>(now - t0).count();
                std::printf("card, inside the model: %llu trace tokens, %.0f tok/s now, %.0f tok/s overall, %.1f s, position %llu, hash = reference on %llu lines\n", (unsigned long long)a, (double)(a - n_mark) / el, (double)(a - n_start) / all, all, (unsigned long long)pos, (unsigned long long)ref_lines);
                print_stats(since);
                std::fflush(stdout);
                n_mark = a;
                t_mark = now;
            }
        } else if (kind == CHIP_EV_LOG) {
            std::printf("card: residual dim %llu, bits %08llx, reaches the output and no row reads it\n", (unsigned long long)a, (unsigned long long)b);
            std::fflush(stdout);
        } else if (kind == CHIP_EV_BYTE) {
            bytes |= (uint32_t)b << (8 * got);
            if (++got == 3) {
                const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                std::printf("card, inside the model: position %llu done, %llu trace tokens for it, %.0f tok/s\n", (unsigned long long)pos, (unsigned long long)(a - last_mark_pos), (double)(a - n_start) / el);
                if (pos + 1 >= prompt_end) {
                    const std::string piece = tok.decode((int32_t)bytes);
                    std::printf("%s", piece.c_str());
                    if (reply) {
                        std::fputs(piece.c_str(), reply);
                        std::fflush(reply);
                    }
                }
                std::fflush(stdout);
                last_mark_pos = a;
                pos++;
                bytes = 0;
                got = 0;
            }
        } else if (kind == CHIP_EV_READY) {
            if (turns && !chat) {
                ctl->stop = 1;
                continue;
            }
            if (turns) {
                std::printf("\n> ");
                std::fflush(stdout);
                if (!std::getline(std::cin, prompt)) {
                    ctl->stop = 1;
                    continue;
                }
            }
            const std::string text = (turns ? std::string("<|im_end|>\n") : std::string()) + Tokenizer::chat_prompt(prompt);
            const std::vector<int32_t> toks = tok.encode(text);
            ChipPoke *list = reinterpret_cast<ChipPoke *>(part + CHIP_POKE_OFFSET);
            size_t k = 0;
            for (size_t i = 0; i < toks.size(); i++) list[k++] = ChipPoke{(uint32_t)(VM_TOKS + pos + i), toks[i], 0, 0};
            prompt_end = pos + toks.size();
            prompt_start = pos;
            list[k++] = ChipPoke{JOB_N, (int32_t)prompt_end, 0, 0};
            list[k++] = ChipPoke{JOB_GEN, (int32_t)gen, 0, 0};
            list[k++] = ChipPoke{JOB_SEQ, 1, 1, 0};
            __sync_synchronize();
            post_cmd(CHIP_OP_POKE, k, 0);
            wait_res();
            turns++;
            last_mark_pos = a;
            std::printf("host: turn %d: %zu tokens in, positions %llu to %llu; the first reply token comes after that many positions of trace\n", turns, toks.size(), (unsigned long long)pos, (unsigned long long)prompt_end);
            std::fflush(stdout);
        } else if (kind == CHIP_EV_CKPT) {
            CkHeader head{};
            head.magic = CK_MAGIC;
            head.n = a;
            head.token = b;
            head.hash = c;
            head.pos = pos;
            head.prompt_end = prompt_end;
            head.prompt_start = prompt_start;
            head.turns = (uint64_t)turns;
            head.got = (uint64_t)got;
            head.bytes = bytes;
            head.total_bytes = h.total_bytes;
            head.slices = h.slices;
            head.pack_bytes = ctl->ck_pack;
            head.state_bytes = ctl->ck_total;
            head.last_mark_pos = last_mark_pos;
            save_checkpoint(ckpt_path, head);
        } else if (kind == CHIP_EV_END) {
            const double all = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("\ncard, inside the model: %llu trace tokens, hash %016llx, %.2f s = %.0f tok/s, 60 x 2 team\n", (unsigned long long)a, (unsigned long long)b, all, (double)(a - n_start) / all);
            print_stats(since);
            (void)last_n;
            break;
        }
    }
    return 0;
}
