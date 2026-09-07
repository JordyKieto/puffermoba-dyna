// Trajectory hashing, shared by the headless server and the rendering client.
//
// Both sides must hash the identical byte sequence or replay verification is
// meaningless, so the stepping protocol lives here rather than being written
// out twice.

#ifndef RACE_HASH_H
#define RACE_HASH_H

#include <stdint.h>
#include <stddef.h>
#include "race.h"

#define RACE_FNV_OFFSET 14695981039346656037ULL
#define RACE_FNV_PRIME  1099511628211ULL

static inline uint64_t race_fnv1a(uint64_t h, const void* data, size_t n) {
    const uint8_t* p = (const uint8_t*)data;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= RACE_FNV_PRIME; }
    return h;
}

// Call once, immediately after c_reset().
static inline uint64_t race_hash_begin(const Race* env) {
    uint64_t h = RACE_FNV_OFFSET;
    return race_fnv1a(h, env->observations, sizeof(float) * RACE_OBS_SIZE);
}

// Call once per step, immediately after c_step(), with the action just applied.
static inline uint64_t race_hash_step(uint64_t h, const Race* env, int action) {
    h = race_fnv1a(h, &action, sizeof(action));
    h = race_fnv1a(h, env->observations, sizeof(float) * RACE_OBS_SIZE);
    h = race_fnv1a(h, &env->rewards[0], sizeof(float));
    h = race_fnv1a(h, &env->terminated, sizeof(int));
    h = race_fnv1a(h, &env->truncated, sizeof(int));
    return h;
}

#endif
