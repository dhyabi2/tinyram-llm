// test_expert_pool_fuzz.c: seeded random scenarios against expert_pool.c (run under ASan/UBSan). Each scenario builds a random extent file and replays random get/unpin
// traffic with random capacities, comparing every returned buffer with the source bytes and checking the pool's invariants after every call.
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "expert_pool.h"
static int failures, checks;
#define CHECK(cond, ...) do { checks++; if (!(cond)) { failures++; printf("FAIL "); printf(__VA_ARGS__); printf("\n"); } } while (0)
static uint64_t s;
static uint64_t rnd(void) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
int main(int argc, char **argv) {
    int nscen = argc > 1 ? atoi(argv[1]) : 150;
    for (int sc = 0; sc < nscen; sc++) {
        s = 0x9E3779B97F4A7C15ull ^ (uint64_t)(sc + 1) * 1000003ull;
        uint32_t NL = 1 + rnd() % 5, NE = 1 + rnd() % 12, N = NL * NE;
        ep_extent *man = calloc(N, sizeof *man); uint8_t **cont = calloc(N, sizeof *cont);
        char path[] = "/tmp/ep_fuzz_XXXXXX"; int fd = mkstemp(path); uint64_t off = 0, total = 0, maxlen = 0;
        for (uint32_t i = 0; i < N; i++) {
            uint64_t n = 1 + rnd() % 70000; if (rnd() % 8 == 0) n = 4096 * (1 + rnd() % 4);              // sometimes exactly aligned
            cont[i] = malloc(n); for (uint64_t k = 0; k < n; k++) cont[i][k] = (uint8_t)rnd();
            man[i].offset = off; man[i].length = n; man[i].crc = ep_crc32(cont[i], n);
            if (pwrite(fd, cont[i], n, (off_t)off) != (ssize_t)n) return 2;
            off += (n + 4095) / 4096 * 4096; total += n; if (n > maxlen) maxlen = n;
        }
        if (ftruncate(fd, (off_t)off)) return 2; close(fd);
        uint64_t cap = maxlen + rnd() % (total + 1); int odirect = (int)(rnd() % 2);
        ep_pool p; CHECK(ep_open(&p, path, man, NL, NE, cap, odirect) == 0, "scenario %d: open", sc);
        const uint8_t *held[8]; int nheld = 0; uint64_t gets = 0, expect_hits = 0; (void)expect_hits;
        for (int step = 0; step < 400; step++) {
            uint64_t r = rnd() % 10; int err = 0; uint64_t len = 0;
            if (r < 6 || nheld == 0) {
                uint32_t l = rnd() % NL, e = rnd() % NE, i = l * NE + e; const uint8_t *b = ep_get(&p, l, e, &len, &err); gets++;
                if (b) { CHECK(len == man[i].length && !memcmp(b, cont[i], len), "scenario %d step %d: bytes differ", sc, step); if (nheld < 8 && rnd() % 3 == 0) held[nheld++] = b; else ep_unpin(&p, b); }
                else CHECK(err == -ENOSPC, "scenario %d step %d: unexpected error %d", sc, step, err);
            } else if (r < 8 && nheld) { ep_unpin(&p, held[--nheld]); }
            else {                                                                                   // invalid requests must fail cleanly
                const uint8_t *b = ep_get(&p, NL + (uint32_t)(rnd() % 3), 0, &len, &err); CHECK(!b && err == -EINVAL, "scenario %d: layer out of range accepted", sc);
                b = ep_get(&p, 0, NE + (uint32_t)(rnd() % 3), &len, &err); CHECK(!b && err == -EINVAL, "scenario %d: expert out of range accepted", sc);
            }
            CHECK(p.used <= p.cap, "scenario %d step %d: used %llu > cap %llu", sc, step, (unsigned long long)p.used, (unsigned long long)p.cap);
        }
        while (nheld) ep_unpin(&p, held[--nheld]);
        CHECK(p.hits + p.misses <= gets, "scenario %d: hits+misses %llu > gets %llu", sc, (unsigned long long)(p.hits + p.misses), (unsigned long long)gets);
        CHECK(p.crc_errors == 0, "scenario %d: spurious crc errors", sc);
        ep_close(&p);
        if (sc % 10 == 0) {                                                                          // truncated file: reads past EOF must fail, not crash
            int tf = open(path, O_RDWR); if (ftruncate(tf, (off_t)(off / 2))) return 2; close(tf);
            ep_pool q; if (ep_open(&q, path, man, NL, NE, total, 0) == 0) { int err = 0; uint64_t len; const uint8_t *b = ep_get(&q, NL - 1, NE - 1, &len, &err); CHECK(!b && err != 0, "scenario %d: truncated file read succeeded", sc); ep_close(&q); }
        }
        if (sc % 15 == 0) { ep_pool q; CHECK(ep_open(&q, "/nonexistent/file", man, NL, NE, total, 0) != 0, "scenario %d: opening a missing file succeeded", sc); }
        unlink(path); for (uint32_t i = 0; i < N; i++) free(cont[i]); free(cont); free(man);
    }
    printf("%d/%d checks passed\n", checks - failures, checks); return failures ? 1 : 0;
}
