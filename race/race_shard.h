// Transition shard format, shared by the headless server (race_eval) and the
// RGB capture path (race_play --rgb-out).
//
// Both write the identical format, so racewm/shards.py reads either one and an
// RGB capture is index-aligned with its symbolic state by construction:
// record t in states.bin is frame t on disk is step t in the env.

#ifndef RACE_SHARD_H
#define RACE_SHARD_H

#include <stdint.h>
#include "race.h"

#define RACE_SHARD_MAGIC 0x52434531u /* "RCE1" */

// Self-describing: everything needed to invert the observation normalization
// lives in the shard, so a dataset can never be silently misread against a
// different config.
typedef struct {
    uint32_t magic;
    uint32_t state_version;
    uint32_t obs_dim;
    uint32_t record_bytes;
    uint32_t seed;
    uint32_t n_records;
    uint32_t num_radial_sectors;
    uint32_t max_episode_steps;
    float    width;
    float    height;
    float    maxv;
    uint32_t frameskip;
    uint32_t total_points;
    uint32_t reserved[3];
} ShardHeader;

typedef struct {
    int32_t env_idx;
    int32_t episode_id;
    int32_t t;
    int32_t action;
    float   reward;
    uint8_t terminated;
    uint8_t truncated;
    uint8_t bot_kind;
    uint8_t pad;
    float   obs[RACE_OBS_SIZE];   // s_t, captured BEFORE the step
} Record;

static inline void shard_header_init(ShardHeader* h, const Race* env,
                                     uint32_t seed, int total_points) {
    h->magic = RACE_SHARD_MAGIC;
    h->state_version = RACE_STATE_VERSION;
    h->obs_dim = RACE_OBS_SIZE;
    h->record_bytes = sizeof(Record);
    h->seed = seed;
    h->n_records = 0;
    h->num_radial_sectors = (uint32_t)env->num_radial_sectors;
    h->max_episode_steps = (uint32_t)env->max_episode_steps;
    h->width = (float)env->width;
    h->height = (float)env->height;
    h->maxv = env->maxv;
    h->frameskip = (uint32_t)env->frameskip;
    h->total_points = (uint32_t)total_points;
    h->reserved[0] = h->reserved[1] = h->reserved[2] = 0;
}

#endif
