// One replay format, used by both the headless server and the rendering
// client. Nothing is duplicated: race_eval records, race_play renders, and
// both drive the env through the identical code path.
//
// A replay stores only (config, seed, env_idx, action sequence). That is
// sufficient because Race is fully deterministic given those -- which is
// exactly what tests/test_determinism.sh proves. Upstream whisker_racer could
// not have done this: episode starts came from the process-global rand()
// stream, so the same actions replayed to a different state.
//
// TEMPORAL ALIGNMENT. Step t in the file is step t in the env on both sides.
// c_step applies frameskip internally, so one recorded action is exactly one
// c_step is exactly one rendered frame. The upstream demo applied frameskip
// OUTSIDE the env (stepping 4x per action for smoother video), which would
// desynchronize client and server by a factor of frameskip; race_play does not
// do that. traj_hash is recomputed on replay and compared, so any divergence
// is caught rather than silently rendered.

#ifndef RACE_REPLAY_H
#define RACE_REPLAY_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "race.h"

#define RACE_REPLAY_MAGIC 0x52525031u /* "RRP1" */

typedef struct {
    uint32_t magic;
    uint32_t state_version;
    uint32_t seed;
    int32_t  env_idx;
    uint32_t n_steps;
    uint32_t flags;
    uint64_t traj_hash;        // FNV-1a over (action, obs, reward, term, trunc)

    // Full env config, so a replay is self-contained and cannot be silently
    // rendered against a different physics setup.
    int32_t width, height, track_width, num_radial_sectors;
    int32_t num_points, bezier_resolution, frameskip, method;
    int32_t max_episode_steps;
    float turn_pi_frac, maxv, w_ang, max_whisker_length;
    float reward_yellow, corner_thresh;

    uint8_t bot_kind;
    uint8_t pad[7];
} ReplayHeader;

// Apply a replay's config to an env. Callers still must call init().
static inline void replay_apply(Race* env, const ReplayHeader* h) {
    memset(env, 0, sizeof(*env));
    env->width = h->width;
    env->height = h->height;
    env->track_width = h->track_width;
    env->num_radial_sectors = h->num_radial_sectors;
    env->num_points = h->num_points;
    env->bezier_resolution = h->bezier_resolution;
    env->frameskip = h->frameskip;
    env->method = h->method;
    env->max_episode_steps = h->max_episode_steps;
    env->turn_pi_frac = h->turn_pi_frac;
    env->maxv = h->maxv;
    env->w_ang = h->w_ang;
    env->max_whisker_length = h->max_whisker_length;
    env->reward_yellow = h->reward_yellow;
    env->corner_thresh = h->corner_thresh;
    env->seed = h->seed;
    env->env_idx = h->env_idx;
    env->render = 0;
    env->render_many = 0;
    env->mode7 = 0;
}

static inline int replay_write(const char* path, const ReplayHeader* h,
                               const uint8_t* actions) {
    FILE* f = fopen(path, "wb");
    if (!f) return 1;
    fwrite(h, sizeof(*h), 1, f);
    fwrite(actions, 1, h->n_steps, f);
    fclose(f);
    return 0;
}

// Caller owns *actions_out (free()).
static inline int replay_read(const char* path, ReplayHeader* h,
                              uint8_t** actions_out) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "replay: cannot open %s\n", path); return 1; }
    if (fread(h, sizeof(*h), 1, f) != 1) {
        fprintf(stderr, "replay: %s is truncated\n", path); fclose(f); return 1;
    }
    if (h->magic != RACE_REPLAY_MAGIC) {
        fprintf(stderr, "replay: %s bad magic %08x\n", path, h->magic);
        fclose(f); return 1;
    }
    if (h->state_version != RACE_STATE_VERSION) {
        fprintf(stderr, "replay: %s is state v%u, this build is v%d\n",
                path, h->state_version, RACE_STATE_VERSION);
        fclose(f); return 1;
    }
    uint8_t* a = (uint8_t*)malloc(h->n_steps ? h->n_steps : 1);
    if (fread(a, 1, h->n_steps, f) != h->n_steps) {
        fprintf(stderr, "replay: %s action stream is short\n", path);
        free(a); fclose(f); return 1;
    }
    fclose(f);
    *actions_out = a;
    return 0;
}

#endif
