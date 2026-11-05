/* A live view of the 2B running inside the 35B on the 7120A: elapsed time, trace tokens, the rate, the hash against the reference, and the 2B's reply as it comes
 * out of the model's layers. It only reads the run's logs, so closing it stops nothing, and it shows nothing from any other source: no estimates, no reference text.
 *
 *   chip_watch LOGDIR "PROMPT" [REFERENCE_LOG]
 *
 * REFERENCE_LOG, if given, is q2b64_check's log for the same job: how many trace tokens each position took there, measured, for the progress lines. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

enum { WIDTH = 92, SAMPLES = 64 };

static char *slurp(const char *path, long *size) {
    FILE *f = fopen(path, "r");
    *size = 0;
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = malloc((size_t)n + 1);
    if (!b) {
        fclose(f);
        return NULL;
    }
    const size_t got = fread(b, 1, (size_t)n, f);
    b[got] = 0;
    *size = (long)got;
    fclose(f);
    return b;
}

static void commas(char *out, size_t room, unsigned long long v) {
    char t[32];
    snprintf(t, sizeof t, "%llu", v);
    const size_t n = strlen(t);
    size_t o = 0;
    for (size_t i = 0; i < n && o + 2 < room; i++) {
        out[o++] = t[i];
        if ((n - i - 1) % 3 == 0 && i + 1 < n) out[o++] = ',';
    }
    out[o] = 0;
}

static void line(const char *text) {
    int cols = 0;
    /* counts columns, not bytes: continuation bytes of UTF-8 take no column */
    for (const unsigned char *p = (const unsigned char *)text; *p; p++) cols += (*p & 0xC0) != 0x80;
    printf("\xE2\x95\x91 %s", text);
    for (int i = cols; i < WIDTH - 4; i++) putchar(' ');
    printf(" \xE2\x95\x91\x1b[K\n");
}

static void rule(const char *left, const char *right) {
    fputs(left, stdout);
    for (int i = 0; i < WIDTH - 2; i++) fputs("\xE2\x95\x90", stdout);
    printf("%s\x1b[K\n", right);
}

static void bar(char *out, size_t room, double frac, int cells) {
    size_t o = 0;
    const int on = (int)(frac * cells + 0.5);
    for (int i = 0; i < cells && o + 4 < room; i++) {
        const char *c = i < on ? "\xE2\x96\x88" : "\xE2\x96\x91";
        memcpy(out + o, c, 3);
        o += 3;
    }
    out[o] = 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: chip_watch LOGDIR PROMPT [REFERENCE_LOG]\n");
        return 2;
    }
    const char *dir = argv[1], *prompt = argv[2], *ref_path = argc > 3 ? argv[3] : NULL;
    char p_run[512], p_hash[512], p_reply[512], p_start[512], p_end[512];
    snprintf(p_run, sizeof p_run, "%s/run.txt", dir);
    snprintf(p_hash, sizeof p_hash, "%s/hash.txt", dir);
    snprintf(p_reply, sizeof p_reply, "%s/reply.txt", dir);
    snprintf(p_start, sizeof p_start, "%s/started.txt", dir);
    snprintf(p_end, sizeof p_end, "%s/first-token.txt", dir);
    double t_s[SAMPLES];
    unsigned long long n_s[SAMPLES];
    int ns = 0;
    time_t first_token = 0;
    fputs("\x1b[2J\x1b[?25l", stdout);
    for (;;) {
        const time_t now = time(NULL);
        long size = 0;
        /* when it started, as the host wrote it */
        time_t started = 0;
        char *s = slurp(p_start, &size);
        if (s) {
            started = (time_t)atoll(s);
            free(s);
        }
        /* trace tokens: the last hash line */
        unsigned long long tokens = 0;
        char *h = slurp(p_hash, &size);
        if (h && size > 0) {
            long i = size - 1;
            while (i > 0 && h[i - 1] != '\n') i--;
            if (size - i < 8 && i > 0) {
                i--;
                while (i > 0 && h[i - 1] != '\n') i--;
            }
            sscanf(h + i, "tokens %llu", &tokens);
        }
        free(h);
        if (ns && tokens < n_s[(ns - 1) % SAMPLES]) ns = 0;  /* a new run */
        if (ns == 0 || tokens != n_s[(ns - 1) % SAMPLES]) {
            t_s[ns % SAMPLES] = (double)now;
            n_s[ns % SAMPLES] = tokens;
            ns++;
        }
        /* the run's own lines */
        char *r = slurp(p_run, &size);
        unsigned long long pos = 0, p0 = 0, p1 = 0, done_positions = 0;
        unsigned long long last_pos_tokens = 0;
        int differs = 0, up = 0, moved = 0;
        char last_err[160] = "";
        if (r) {
            for (char *l = r; *l;) {
                char *e = strchr(l, '\n');
                if (e) *e = 0;
                unsigned long long a, b;
                size_t nt;
                if (sscanf(l, "host: turn %*d: %zu tokens in, positions %llu to %llu", &nt, &a, &b) == 3) {
                    p0 = a;
                    p1 = b;
                    pos = a;
                } else if (sscanf(l, "card, inside the model: position %llu done, %llu trace tokens", &a, &b) == 2) {
                    pos = a + 1;
                    done_positions++;
                    last_pos_tokens = b;
                }
                if (strstr(l, "DIFFERS")) {
                    differs = 1;
                    snprintf(last_err, sizeof last_err, "%.86s", l);
                }
                if (strstr(l, "card team is up")) up = 1;
                if (strstr(l, "moved into card memory")) moved = 1;
                if (strstr(l, "Segmentation") || strstr(l, "does not answer") || strstr(l, "did not start") || strstr(l, "refused")) snprintf(last_err, sizeof last_err, "%.86s", l);
                if (!e) break;
                l = e + 1;
            }
            free(r);
        }
        /* the reference's measured count for each position, the same job on the checker */
        double ref_first = 0, ref_to_first = 0;
        char *f = ref_path ? slurp(ref_path, &size) : NULL;
        unsigned long long ref_seen = 0;
        if (f) {
            for (char *l = f; *l;) {
                char *e = strchr(l, '\n');
                if (e) *e = 0;
                unsigned long long a;
                const char *m = strstr(l, " M trace tokens");
                if (m && sscanf(l, "reference: position %llu,", &a) == 1) {
                    const char *q = m;
                    while (q > l && q[-1] != ' ') q--;
                    const double v = atof(q) * 1e6;
                    if (!ref_seen) ref_first = v;
                    if (!p1 || a < p1) ref_to_first += v;
                    ref_seen++;
                }
                if (!e) break;
                l = e + 1;
            }
            free(f);
        }
        char *reply = slurp(p_reply, &size);
        if (reply && size > 0 && !first_token) {
            first_token = now;
            FILE *ft = fopen(p_end, "a");
            if (ft) {
                fprintf(ft, "%lld\n", (long long)now);
                fclose(ft);
            }
        }

        /* the rate over the last samples, and overall */
        double rate = 0, overall = 0;
        if (ns >= 2) {
            const int a = ns > SAMPLES ? ns - SAMPLES : 0, b = ns - 1;
            const double dt = t_s[b % SAMPLES] - t_s[a % SAMPLES];
            if (dt > 0) rate = (double)(n_s[b % SAMPLES] - n_s[a % SAMPLES]) / dt;
        }
        const long el = started ? (long)(now - started) : 0;
        if (el > 0) overall = (double)tokens / (double)el;

        char t[512], a1[64], a2[64], br[256];
        fputs("\x1b[H", stdout);
        rule("\xE2\x95\x94", "\xE2\x95\x97");
        line("Qwen3.8-2B INSIDE Qwen3.8-35B: on the 7120A's VPU, 60 cores x 2 threads");
        line("Each token through the computer layers in the model file. No interpreter or NPU linked.");
        rule("\xE2\x95\xA0", "\xE2\x95\xA3");
        snprintf(t, sizeof t, "prompt: %.80s", prompt);
        line(t);
        if (started) snprintf(t, sizeof t, "running for: %ld d %02ld:%02ld:%02ld", el / 86400, el / 3600 % 24, el / 60 % 60, el % 60);
        else snprintf(t, sizeof t, "%s", moved ? "starting the team on the card" : "loading the model file, moving it into card memory");
        line(t);
        commas(a1, sizeof a1, tokens);
        snprintf(t, sizeof t, "trace tokens: %s", a1);
        line(t);
        snprintf(t, sizeof t, "rate: %.0f tok/s over the last samples, %.0f tok/s overall", rate, overall);
        line(t);
        rule("\xE2\x95\xA0", "\xE2\x95\xA3");
        if (p1) {
            snprintf(t, sizeof t, "2B position: %llu; the prompt is positions %llu to %llu, %llu of %llu done", pos, p0, p1, done_positions < p1 - p0 ? done_positions : p1 - p0, p1 - p0);
            line(t);
        } else {
            line(up ? "2B position: the machine is booting, the prompt goes in at its READY" : "2B position: not started");
        }
        if (last_pos_tokens) {
            commas(a1, sizeof a1, last_pos_tokens);
            snprintf(t, sizeof t, "last position took %s trace tokens in the layers", a1);
            line(t);
        } else {
            line("no position finished in the layers yet");
        }
        if (ref_first > 0 && tokens) {
            commas(a1, sizeof a1, tokens);
            commas(a2, sizeof a2, (unsigned long long)ref_first);
            double fr = (double)tokens / ref_first;
            if (fr > 1) fr = 1;
            bar(br, sizeof br, fr, 24);
            snprintf(t, sizeof t, "first position:    %s %6.3f %% of %s (reference)", br, fr * 100.0, a2);
            line(t);
            if (ref_to_first > 0 && ref_seen && p1 && ref_seen >= p1 - p0) {
                commas(a2, sizeof a2, (unsigned long long)ref_to_first);
                fr = (double)tokens / ref_to_first;
                if (fr > 1) fr = 1;
                bar(br, sizeof br, fr, 24);
                snprintf(t, sizeof t, "first reply token: %s %6.3f %% of %s (reference)", br, fr * 100.0, a2);
                line(t);
            } else {
                line("first reply token: the reference has not reached it yet");
            }
        } else {
            line("first position: no count yet");
            line("first reply token: no count yet");
        }
        const unsigned long long lines = tokens / 4096;
        if (differs) line(last_err);
        else {
            snprintf(t, sizeof t, "hash against the reference: the same on all %llu lines so far", lines);
            line(t);
        }
        if (!differs && last_err[0]) line(last_err);
        rule("\xE2\x95\xA0", "\xE2\x95\xA3");
        if (reply && size > 0) {
            if (first_token && started) {
                const long ft = (long)(first_token - started);
                snprintf(t, sizeof t, "THE 2B'S REPLY, from the layers; first token after %ld d %02ld:%02ld:%02ld", ft / 86400, ft / 3600 % 24, ft / 60 % 60, ft % 60);
                line(t);
            }
            /* the reply, wrapped at the frame */
            const char *p = reply;
            char row[WIDTH * 4];
            while (*p) {
                int cols = 0;
                size_t o = 0;
                while (*p && *p != '\n' && cols < WIDTH - 4 && o + 8 < sizeof row) {
                    const unsigned char c = (unsigned char)*p;
                    row[o++] = *p++;
                    if ((c & 0xC0) != 0x80) cols++;
                }
                row[o] = 0;
                if (*p == '\n') p++;
                line(row);
            }
        } else {
            line("THE 2B'S REPLY: nothing yet. It appears here token by token, as the layers emit it.");
        }
        free(reply);
        rule("\xE2\x95\x9A", "\xE2\x95\x9D");
        fputs("\x1b[J", stdout);
        fflush(stdout);
        sleep(1);
    }
    return 0;
}
