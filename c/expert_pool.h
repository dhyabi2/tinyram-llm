#ifndef EXPERT_POOL_H
#define EXPERT_POOL_H
#include <stdint.h>
#include <stddef.h>

typedef struct { uint64_t offset, length; uint32_t crc; uint32_t _pad; } ep_extent;   // one per (layer, expert), index = layer * n_experts + expert

typedef struct ep_entry {
    uint64_t key, len;
    uint8_t *buf;
    int pins;
    struct ep_entry *prev, *next, *hnext;
} ep_entry;

typedef struct {
    int fd, odirect;
    const ep_extent *manifest;
    uint32_t n_layers, n_experts;
    uint64_t cap, used;
    ep_entry **buckets; size_t nbuckets;
    ep_entry *mru, *lru;
    uint64_t hits, misses, evictions, bytes_read, crc_errors;
} ep_pool;

uint32_t ep_crc32(const uint8_t *p, uint64_t n);
// Returns 0, or -errno. want_odirect falls back to buffered reads where the filesystem has no O_DIRECT; see pool->odirect.
int ep_open(ep_pool *p, const char *path, const ep_extent *manifest, uint32_t n_layers, uint32_t n_experts, uint64_t capacity_bytes, int want_odirect);
// Returns a pinned pointer to the expert's bytes (valid until ep_unpin), or NULL with *err = -errno (-EBADMSG: crc mismatch,
// -ENOSPC: every resident entry is pinned, -E2BIG: one expert exceeds the whole capacity).
const uint8_t *ep_get(ep_pool *p, uint32_t layer, uint32_t expert, uint64_t *len, int *err);
void ep_unpin(ep_pool *p, const uint8_t *buf);
void ep_close(ep_pool *p);
#endif
