// expert_pool.c — byte-capacity LRU pool for expert extents (the core of patch P-C).
//
//   - experts live in ONE extent file; a manifest gives (offset, length, crc32) per (layer, expert); offsets are 4 KiB aligned
//   - ep_get() returns a pointer to the expert's bytes, reading it with pread (optionally O_DIRECT) on a miss and evicting
//     least-recently-used UNPINNED entries until the byte budget fits; the entry stays pinned until ep_unpin()
//   - every read is verified against its crc32; a mismatch is an error, never silently used
//
// Single-threaded core: the llama.cpp integration calls it from the thread that owns the graph; the prefetch path (io_uring)
// replaces the pread inside load_entry() and is the only part that needs liburing.
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "expert_pool.h"

#define ALIGN 4096u

static uint32_t crc_table[256];
static int crc_ready;
uint32_t ep_crc32(const uint8_t *p, uint64_t n) {
    if (!crc_ready) {
        for (uint32_t i = 0; i < 256; i++) { uint32_t c = i; for (int k = 0; k < 8; k++) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1; crc_table[i] = c; }
        crc_ready = 1;
    }
    uint32_t c = 0xFFFFFFFFu;
    for (uint64_t i = 0; i < n; i++) c = crc_table[(c ^ p[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

static uint64_t key_of(uint32_t layer, uint32_t expert) { return ((uint64_t)layer << 32) | expert; }
static size_t slot_of(const ep_pool *p, uint64_t key) { return (size_t)((key * 0x9E3779B97F4A7C15ull) >> 40) & (p->nbuckets - 1); }

static void list_unlink(ep_pool *p, ep_entry *e) {
    if (e->prev) e->prev->next = e->next; else p->mru = e->next;
    if (e->next) e->next->prev = e->prev; else p->lru = e->prev;
    e->prev = e->next = NULL;
}
static void list_push_mru(ep_pool *p, ep_entry *e) {
    e->prev = NULL; e->next = p->mru;
    if (p->mru) p->mru->prev = e;
    p->mru = e; if (!p->lru) p->lru = e;
}
static ep_entry *find(ep_pool *p, uint64_t key) {
    for (ep_entry *e = p->buckets[slot_of(p, key)]; e; e = e->hnext) if (e->key == key) return e;
    return NULL;
}
static void hash_remove(ep_pool *p, ep_entry *e) {
    ep_entry **pp = &p->buckets[slot_of(p, e->key)];
    while (*pp && *pp != e) pp = &(*pp)->hnext;
    if (*pp) *pp = e->hnext;
}

int ep_open(ep_pool *p, const char *path, const ep_extent *manifest, uint32_t n_layers, uint32_t n_experts, uint64_t capacity_bytes, int want_odirect) {
    memset(p, 0, sizeof *p);
    p->odirect = 0;
    p->fd = -1;
#ifdef O_DIRECT
    if (want_odirect) { p->fd = open(path, O_RDONLY | O_DIRECT); if (p->fd >= 0) p->odirect = 1; }
#endif
    if (p->fd < 0) p->fd = open(path, O_RDONLY);          // filesystem without O_DIRECT (tmpfs): fall back, report via p->odirect
    if (p->fd < 0) return -errno;
    p->manifest = manifest; p->n_layers = n_layers; p->n_experts = n_experts; p->cap = capacity_bytes;
    p->nbuckets = 1u << 12;
    p->buckets = calloc(p->nbuckets, sizeof *p->buckets);
    return p->buckets ? 0 : -ENOMEM;
}

static int evict_one(ep_pool *p) {
    for (ep_entry *e = p->lru; e; e = e->prev) {           // oldest first
        if (e->pins) continue;
        list_unlink(p, e); hash_remove(p, e);
        p->used -= e->len; p->evictions++;
        free(e->buf); free(e);
        return 0;
    }
    return -ENOSPC;                                         // everything resident is pinned
}

static int load_entry(ep_pool *p, const ep_extent *x, ep_entry *e) {
    uint64_t rd = (x->length + ALIGN - 1) / ALIGN * ALIGN;  // O_DIRECT needs aligned length, buffer and offset
    void *buf = NULL;
    if (posix_memalign(&buf, ALIGN, rd)) return -ENOMEM;
    uint64_t got = 0;
    while (got < x->length) {
        ssize_t r = pread(p->fd, (uint8_t *)buf + got, rd - got, (off_t)(x->offset + got));
        if (r < 0) { if (errno == EINTR) continue; free(buf); return -errno; }
        if (r == 0) break;
        got += (uint64_t)r;
    }
    if (got < x->length) { free(buf); return -EIO; }        // short extent
    if (ep_crc32(buf, x->length) != x->crc) { free(buf); p->crc_errors++; return -EBADMSG; }
    p->bytes_read += rd;
    e->buf = buf;
    return 0;
}

const uint8_t *ep_get(ep_pool *p, uint32_t layer, uint32_t expert, uint64_t *len, int *err) {
    int dummy; if (!err) err = &dummy;
    *err = 0;
    if (layer >= p->n_layers || expert >= p->n_experts) { *err = -EINVAL; return NULL; }
    uint64_t key = key_of(layer, expert);
    ep_entry *e = find(p, key);
    if (e) {                                                // hit: refresh recency, pin
        list_unlink(p, e); list_push_mru(p, e); e->pins++; p->hits++;
        if (len) *len = e->len;
        return e->buf;
    }
    const ep_extent *x = &p->manifest[(size_t)layer * p->n_experts + expert];
    if (x->length > p->cap) { *err = -E2BIG; return NULL; }
    while (p->used + x->length > p->cap) if ((*err = evict_one(p))) return NULL;
    e = calloc(1, sizeof *e);
    if (!e) { *err = -ENOMEM; return NULL; }
    e->key = key; e->len = x->length;
    if ((*err = load_entry(p, x, e))) { free(e); return NULL; }
    e->pins = 1; p->used += e->len; p->misses++;
    size_t s = slot_of(p, key); e->hnext = p->buckets[s]; p->buckets[s] = e;
    list_push_mru(p, e);
    if (len) *len = e->len;
    return e->buf;
}

void ep_unpin(ep_pool *p, const uint8_t *buf) {
    for (ep_entry *e = p->mru; e; e = e->next) if (e->buf == buf) { if (e->pins) e->pins--; return; }
}

void ep_close(ep_pool *p) {
    ep_entry *e = p->mru;
    while (e) { ep_entry *n = e->next; free(e->buf); free(e); e = n; }
    free(p->buckets);
    if (p->fd >= 0) close(p->fd);
    memset(p, 0, sizeof *p); p->fd = -1;
}
