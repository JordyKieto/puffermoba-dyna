// MOBA with capturable randomness, a non-resetting step, and restore.
//
// Upstream draws all of its randomness (respawn and spawn jitter, the cached
// jitter table) from the process-global rand(), whose state cannot be read or
// set. Here rand() is redirected, at compile time, to a SplitMix64 stream
// whose 64-bit state is part of MobaState. Each MobaGame owns its own stream
// and swaps it in around every call into the env, so several games can live
// in one process without sharing randomness, and a game restored from a
// MobaState continues exactly as the original. third_party/ is not modified.
//
// moba_step() is upstream c_step() without the automatic reset: when an
// ancient falls it returns the result and leaves the terminal state in place,
// so it can be captured; moba_game_end() then does what upstream would have
// (add_log + c_reset). `moba_collect markov` checks this path against
// upstream c_step byte-for-byte.
//
// Include this instead of moba.h.

#ifndef MOBA_GAME_H
#define MOBA_GAME_H

#include <stdint.h>
#include <stdlib.h>

static uint64_t moba_rng_current;   // the stream of whichever game is executing

static inline int moba_rand(void) {
    uint64_t z = (moba_rng_current += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return (int)((z ^ (z >> 31)) >> 33);   // 31 bits: [0, RAND_MAX] on macOS and glibc
}

#define rand() moba_rand()
#include "moba.h"
#undef rand

#include "moba_state.h"

// The cached jitter table is filled from rand() when an env is built. It is
// generated from this fixed seed in every game, so it is a constant of the
// simulator rather than state; only the index into it (rng_idx) varies.
#define MOBA_TABLE_SEED 0x7AB1E5EEDULL

typedef struct {
    MOBA env;
    uint64_t rng;
} MobaGame;

static inline MOBA moba_default_config(void) {
    // Upstream demo config (the one the shipped weights were trained on),
    // except that dire acts through the action interface instead of creep_ai.
    return (MOBA){
        .vision_range = 5,
        .agent_speed = 1.0,
        .reward_death = -0.163764,
        .reward_xp = 0.00665677,
        .reward_distance = 0,
        .reward_tower = 0.642119,
        .script_opponents = false,
    };
}

static inline void moba_game_alloc(MobaGame* g) {
    g->env = moba_default_config();
    moba_rng_current = MOBA_TABLE_SEED;
    allocate_moba(&g->env);
    g->rng = moba_rng_current;
}

static inline void moba_game_free(MobaGame* g) {
    free_allocated_moba(&g->env);
}

// Rebuild the env from scratch for a new game: nothing survives from the last.
static inline void moba_game_start(MobaGame* g, uint64_t seed) {
    MOBA* env = &g->env;
    c_close(env);
    memset(env->scanned_targets, 0, sizeof(env->scanned_targets));
    memset(env->player_logs, 0, sizeof(env->player_logs));
    memset(&env->log, 0, sizeof(env->log));
    env->should_reset = false;
    env->ai_path_buffer = calloc(3 * 8 * 128 * 128, sizeof(int));
    moba_rng_current = MOBA_TABLE_SEED;
    init_moba(env, game_map_npy);
    moba_rng_current = seed;
    c_reset(env);
    g->rng = moba_rng_current;
}

// Upstream c_step() minus the reset. Returns 0 while the game runs, else a
// bitmask: 1 the dire ancient fell (radiant win), 2 the radiant ancient fell.
static inline int moba_step_env(MOBA* env) {
    for (int pid = 0; pid < NUM_ENTITIES; pid++) {
        Entity* entity = &env->entities[pid];
        entity->target_pid = -1;
        entity->attack_aoe = 0;
        entity->last_x = entity->x;
        entity->last_y = entity->y;
        entity->is_hit = 0;
    }

    step_neutrals(env);
    step_creeps(env);
    step_towers(env);
    step_players(env);
    env->tick += 1;

    int result = 0;
    Entity* ancient = &env->entities[TOWER_OFFSET + 22];
    if (ancient->health <= 0 || ancient->pid == -1)
        result |= 1;
    ancient = &env->entities[TOWER_OFFSET + 23];
    if (ancient->health <= 0 || ancient->pid == -1)
        result |= 2;

    if (!result && env->should_reset) {
        fprintf(stderr, "moba_step: should_reset set without an ancient falling\n");
        exit(4);
    }
    compute_observations(env);
    return result;
}

static inline int moba_game_step(MobaGame* g) {
    moba_rng_current = g->rng;
    int result = moba_step_env(&g->env);
    g->rng = moba_rng_current;
    return result;
}

// What upstream c_step does after an ancient falls: log, then reset.
static inline void moba_game_end(MobaGame* g, int result) {
    moba_rng_current = g->rng;
    g->env.should_reset = false;
    add_log(&g->env, result & 1, (result & 2) != 0);
    c_reset(&g->env);
    g->rng = moba_rng_current;
}

// Upstream's own step, for checking moba_game_step against it.
static inline void moba_game_c_step(MobaGame* g) {
    moba_rng_current = g->rng;
    c_step(&g->env);
    g->rng = moba_rng_current;
}

static inline void moba_game_capture(const MobaGame* g, MobaState* s) {
    moba_capture(&g->env, g->rng, s);
}

// Build a fresh game (any seed) and overwrite it with s.
static inline void moba_game_restore(MobaGame* g, const MobaState* s, uint64_t omit) {
    moba_game_start(g, 0xF4E5ULL);
    moba_restore_partial(&g->env, &g->rng, s, omit);
}

#endif
