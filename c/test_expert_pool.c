// test_expert_pool.c — self-test for expert_pool.c. Builds a synthetic extent file and checks the pool against an independent model.
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "expert_pool.h"

#define NL 4
#define NE 16
static int failures, checks;
#define CHECK(name, cond, ...) do { checks++; if (cond) printf("PASS %s\n", name); else { failures++; printf("FAIL %s  ", name); printf(__VA_ARGS__); printf("\n"); } } while (0)

static uint64_t rng_state = 88172645463325252ull;
static uint64_t rnd(void) { rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17; return rng_state; }

static ep_extent man[NL * NE];
static uint8_t *content[NL * NE];

static void build_file(const char *path) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    uint64_t off = 0;
    static const uint64_t sizes[3] = {61000, 91000, 131000};          // deliberately NOT multiples of 4096
    for (int i = 0; i < NL * NE; i++) {
        uint64_t n = sizes[rnd() % 3];
        content[i] = malloc(n);
        for (uint64_t k = 0; k < n; k++) content[i][k] = (uint8_t)rnd();
        man[i].offset = off; man[i].length = n; man[i].crc = ep_crc32(content[i], n); man[i]._pad = 0;
        if (pwrite(fd, content[i], n, (off_t)off) != (ssize_t)n) { perror("pwrite"); exit(2); }
        off += (n + 4095) / 4096 * 4096;
    }
    if (ftruncate(fd, (off_t)off)) exit(2);
    close(fd);
}

static uint64_t total_bytes(void) { uint64_t t = 0; for (int i = 0; i < NL * NE; i++) t += man[i].length; return t; }

int main(void) {
    char path[] = "/tmp/ep_test_XXXXXX";
    int tfd = mkstemp(path); close(tfd);
    build_file(path);

    int aligned = 1;
    for (int i = 0; i < NL * NE; i++) if (man[i].offset % 4096) aligned = 0;
    CHECK("every extent starts on a 4 KiB boundary", aligned, "n/a");

    // ---- 1. miss then hit returns identical bytes -------------------------------------------------------------------------
    ep_pool p; int err; uint64_t len;
    CHECK("open succeeds", ep_open(&p, path, man, NL, NE, total_bytes(), 1) == 0, "n/a");
    printf("INFO O_DIRECT in use: %s\n", p.odirect ? "yes" : "no (filesystem does not support it; buffered fallback)");
    const uint8_t *b = ep_get(&p, 2, 5, &len, &err);
    CHECK("first access is a miss and returns the exact bytes", b && !err && len == man[2 * NE + 5].length && !memcmp(b, content[2 * NE + 5], len), "err=%d", err);
    ep_unpin(&p, b);
    b = ep_get(&p, 2, 5, &len, &err);
    CHECK("second access is a hit", p.hits == 1 && p.misses == 1 && b && !memcmp(b, content[2 * NE + 5], len), "hits=%llu misses=%llu", (unsigned long long)p.hits, (unsigned long long)p.misses);
    ep_unpin(&p, b);
    ep_close(&p);

    // ---- 2. LRU order under a byte cap ------------------------------------------------------------------------------------
    uint64_t a = man[0].length, bb = man[1].length, c = man[2].length, d = man[3].length;
    uint64_t cap = a + bb + c;                                           // exactly the first three fit
    ep_open(&p, path, man, NL, NE, cap, 0);
    for (int i = 0; i < 3; i++) { b = ep_get(&p, 0, i, &len, &err); ep_unpin(&p, b); }
    b = ep_get(&p, 0, 0, &len, &err); ep_unpin(&p, b);                  // touch 0: order now (oldest) 1, 2, 0
    b = ep_get(&p, 0, 3, &len, &err); ep_unpin(&p, b);                  // must evict 1 (least recently used), and only enough of it
    uint64_t m0 = p.misses;
    b = ep_get(&p, 0, 0, &len, &err); ep_unpin(&p, b);
    CHECK("the recently touched entry survives", p.misses == m0, "n/a");
    b = ep_get(&p, 0, 1, &len, &err); ep_unpin(&p, b);
    CHECK("the least recently used entry was the one evicted", p.misses == m0 + 1, "misses=%llu expected %llu", (unsigned long long)p.misses, (unsigned long long)(m0 + 1));
    CHECK("bytes in use never exceed the capacity", p.used <= p.cap, "used=%llu cap=%llu", (unsigned long long)p.used, (unsigned long long)p.cap);
    (void)a; (void)bb; (void)c; (void)d;
    ep_close(&p);

    // ---- 3. cross-check against an independent byte-LRU model on a random persistence trace -----------------------------------
    uint64_t cap3 = total_bytes() / 4;
    ep_open(&p, path, man, NL, NE, cap3, 0);
    int order[NL * NE], nres = 0;                                        // model: array ordered oldest -> newest
    uint64_t mused = 0, mhits = 0, maccess = 0;
    int prev[NL][4]; for (int l = 0; l < NL; l++) for (int k = 0; k < 4; k++) prev[l][k] = (int)(rnd() % NE);
    int inv_ok = 1;
    for (int t = 0; t < 3000; t++) for (int l = 0; l < NL; l++) for (int k = 0; k < 4; k++) {
        if (rnd() % 10 >= 7) prev[l][k] = (int)(rnd() % NE);            // persistence 0.7
        int idx = l * NE + prev[l][k];
        // model
        int pos = -1; for (int i = 0; i < nres; i++) if (order[i] == idx) pos = i;
        maccess++;
        if (pos >= 0) { mhits++; memmove(&order[pos], &order[pos + 1], (size_t)(nres - pos - 1) * sizeof(int)); order[nres - 1] = idx; }
        else {
            while (mused + man[idx].length > cap3) { mused -= man[order[0]].length; memmove(&order[0], &order[1], (size_t)(nres - 1) * sizeof(int)); nres--; }
            order[nres++] = idx; mused += man[idx].length;
        }
        // pool
        b = ep_get(&p, (uint32_t)l, (uint32_t)prev[l][k], &len, &err);
        if (!b || memcmp(b, content[idx], len)) inv_ok = 0;
        ep_unpin(&p, b);
        if (p.used > p.cap) inv_ok = 0;
    }
    CHECK("every returned buffer matched the source bytes and the cap was never exceeded", inv_ok, "n/a");
    CHECK("pool hits equal the independent LRU model exactly", p.hits == mhits && p.hits + p.misses == maccess, "pool=%llu model=%llu of %llu", (unsigned long long)p.hits, (unsigned long long)mhits, (unsigned long long)maccess);
    printf("INFO trace: %llu accesses, hit rate %.3f, evictions %llu, bytes read %.1f MB\n", (unsigned long long)maccess, (double)p.hits / (double)maccess, (unsigned long long)p.evictions, (double)p.bytes_read / 1e6);
    ep_close(&p);

    // ---- 4. corruption is detected ----------------------------------------------------------------------------------------
    char bad[64]; snprintf(bad, sizeof bad, "%s.bad", path);
    { char cmd[256]; snprintf(cmd, sizeof cmd, "cp %s %s", path, bad); if (system(cmd)) return 2; }
    int bfd = open(bad, O_RDWR); uint8_t junk = (uint8_t)(content[7][100] ^ 0xFF); if (pwrite(bfd, &junk, 1, (off_t)(man[7].offset + 100)) != 1) return 2; close(bfd);
    ep_open(&p, bad, man, NL, NE, total_bytes(), 0);
    b = ep_get(&p, 0, 7, &len, &err);
    CHECK("a corrupted extent is rejected with EBADMSG and counted", !b && err == -EBADMSG && p.crc_errors == 1, "err=%d", err);
    b = ep_get(&p, 0, 8, &len, &err);
    CHECK("other extents in the same file still load", b && !err, "err=%d", err);
    ep_unpin(&p, b); ep_close(&p); unlink(bad);

    // ---- 5. pinning ---------------------------------------------------------------------------------------------------------
    int j = -1; for (int i = 2; i < NL * NE; i++) if (man[i].length <= man[0].length) { j = i; break; }   // an expert guaranteed to fit in expert 0's slot
    ep_open(&p, path, man, NL, NE, man[0].length + man[1].length, 0);
    const uint8_t *p0 = ep_get(&p, 0, 0, &len, &err);                    // stays pinned
    const uint8_t *p1 = ep_get(&p, 0, 1, &len, &err);                    // stays pinned; the pool is now full of pinned entries
    const uint8_t *p2 = j >= 0 ? ep_get(&p, (uint32_t)(j / NE), (uint32_t)(j % NE), &len, &err) : NULL;
    CHECK("when everything resident is pinned, a new load fails with ENOSPC instead of evicting", j >= 0 && !p2 && err == -ENOSPC, "j=%d err=%d", j, err);
    ep_unpin(&p, p0);
    p2 = ep_get(&p, (uint32_t)(j / NE), (uint32_t)(j % NE), &len, &err);
    CHECK("after unpinning, the unpinned entry is evicted and the load succeeds", p2 && !err && !memcmp(p2, content[j], man[j].length), "err=%d", err);
    CHECK("a pinned entry's bytes stay valid while others are evicted", p1 && !memcmp(p1, content[1], man[1].length), "n/a");
    ep_close(&p);

    CHECK("an expert larger than the whole capacity is rejected with E2BIG", (ep_open(&p, path, man, NL, NE, 1000, 0) == 0) && !ep_get(&p, 0, 0, &len, &err) && err == -E2BIG, "err=%d", err);
    ep_close(&p);
    CHECK("an out-of-range expert id is rejected with EINVAL", (ep_open(&p, path, man, NL, NE, total_bytes(), 0) == 0) && !ep_get(&p, NL, 0, &len, &err) && err == -EINVAL, "err=%d", err);
    ep_close(&p);

    unlink(path);
    printf("\n%d/%d checks passed\n", checks - failures, checks);
    return failures ? 1 : 0;
}
