// The complete MOBA game state, restore, and the ego observation as a pure
// function of it.
//
// MobaState is sufficient: restored into a freshly built env, it reproduces
// the original game's future byte-for-byte (`moba_collect markov` tests this,
// together with the necessity of every part and the irrelevance of every
// field left out). The randomness is part of it (rng_state), so the next
// state is an exact function of (state, the 10 heroes' actions).
//
// moba_ego_obs() is a copy of upstream compute_observations() (moba.h) that
// reads a MobaState instead of a live env, so observations can be produced
// from a state no env ever held -- e.g. one predicted inside a learned model.
// It is byte-exact against the env, quirks included. mobawm/obs.py is the
// numpy port.
//
// Include via moba_game.h, which sets up the RNG and includes moba.h first
// (moba.h has no include guard).

#ifndef MOBA_STATE_H
#define MOBA_STATE_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MOBA_HEROES 10
#define MOBA_OBS 510
#define MOBA_ATNS 6
#define MOBA_MAP 128
#define MOBA_SCAN_CAP 64          // target-list entries kept per entity; capture aborts past it
#define MOBA_NONE 255             // empty cell in pids / end of a target list
#define MOBA_N_ACTION_CODES 1176  // 7*7*3*2*2*2

static const int MOBA_ACTION_SIZES[MOBA_ATNS] = {7, 7, 3, 2, 2, 2};

// Per-entity fields that change during a game, in this order. Keep in sync
// with mobawm/shards.py STATE_FIELDS. Everything else in upstream's Entity is
// either constant for the whole game (spawn point, base stats, per-level
// gains, tier) or reset at the start of every step before it is read
// (last_x/last_y, attack_aoe); the markov test perturbs those and checks the
// future does not move.
//
// team, grid_id, lane, entity_type, basic_attack_cd and xp_on_kill are here,
// not in the static table, because creep slots are reused round-robin across
// lanes: spawn_creep rewrites them on every wave.
enum {
    SF_ALIVE, SF_X, SF_Y, SF_HEALTH, SF_MAX_HEALTH, SF_MANA, SF_MAX_MANA,
    SF_DAMAGE, SF_LEVEL, SF_XP, SF_MOVE_MODIFIER, SF_STUN_TIMER, SF_MOVE_TIMER,
    SF_Q_TIMER, SF_W_TIMER, SF_E_TIMER, SF_BASIC_ATTACK_TIMER, SF_IS_HIT,
    SF_WAYPOINT, SF_TARGET_PID, SF_TEAM, SF_GRID_ID, SF_LANE, SF_ENTITY_TYPE,
    SF_BASIC_ATTACK_CD, SF_XP_ON_KILL,
    MOBA_STATE_FIELDS
};

// Per-hero attributes that never change within a game (heroes 0..9 only).
typedef struct {
    int8_t hero_type;         // 0 support, 1 assassin, 2 burst, 3 tank, 4 carry
    int8_t pad[3];
    float  move_speed;
} MobaStatic;

typedef struct {
    int32_t  tick;            // drives creep (150) / neutral (600) spawns and scan phases
    int32_t  creep_idx;       // next creep slot a wave spawn reuses
    int32_t  rng_idx;         // position in the cached jitter table
    int32_t  pad;
    uint64_t rng_state;       // the env's random stream (SplitMix64, moba_game.h)
    float    entity[NUM_ENTITIES][MOBA_STATE_FIELDS];
    float    reward[NUM_PLAYERS][4];            // death, xp, distance, tower (last step)
    uint8_t  grid[MOBA_MAP * MOBA_MAP];         // tile ids, NOT derivable from entities
    uint8_t  pids[MOBA_MAP * MOBA_MAP];         // entity index per cell, MOBA_NONE empty
    uint8_t  scan[NUM_ENTITIES][MOBA_SCAN_CAP]; // target lists, MOBA_NONE-terminated
} MobaState;

// Parts of MobaState, for the necessity test: each entity field, then these.
enum {
    PART_GRID = MOBA_STATE_FIELDS, PART_PIDS, PART_SCAN, PART_REWARD,
    PART_TICK, PART_CREEP_IDX, PART_RNG_IDX, PART_RNG_STATE,
    MOBA_N_PARTS
};

static const char* MOBA_FIELD_NAMES[MOBA_N_PARTS] = {
    "alive", "x", "y", "health", "max_health", "mana", "max_mana", "damage",
    "level", "xp", "move_modifier", "stun_timer", "move_timer", "q_timer",
    "w_timer", "e_timer", "basic_attack_timer", "is_hit", "waypoint",
    "target_pid", "team", "grid_id", "lane", "entity_type", "basic_attack_cd",
    "xp_on_kill",
    "grid", "pids", "scan", "reward", "tick", "creep_idx", "rng_idx", "rng_state",
};

static inline void moba_capture_static(const MOBA* env, MobaStatic st[MOBA_HEROES]) {
    for (int i = 0; i < MOBA_HEROES; i++) {
        memset(&st[i], 0, sizeof(st[i]));
        st[i].hero_type = (int8_t)env->entities[i].hero_type;
        st[i].move_speed = env->entities[i].move_speed;
    }
}

static inline void moba_capture_entities(const MOBA* env, float out[NUM_ENTITIES][MOBA_STATE_FIELDS]) {
    for (int i = 0; i < NUM_ENTITIES; i++) {
        const Entity* e = &env->entities[i];
        float* s = out[i];
        s[SF_ALIVE] = e->pid != -1;
        s[SF_X] = e->x;
        s[SF_Y] = e->y;
        s[SF_HEALTH] = e->health;
        s[SF_MAX_HEALTH] = e->max_health;
        s[SF_MANA] = e->mana;
        s[SF_MAX_MANA] = e->max_mana;
        s[SF_DAMAGE] = e->damage;
        s[SF_LEVEL] = e->level;
        s[SF_XP] = e->xp;
        s[SF_MOVE_MODIFIER] = e->move_modifier;
        s[SF_STUN_TIMER] = e->stun_timer;
        s[SF_MOVE_TIMER] = e->move_timer;
        s[SF_Q_TIMER] = e->q_timer;
        s[SF_W_TIMER] = e->w_timer;
        s[SF_E_TIMER] = e->e_timer;
        s[SF_BASIC_ATTACK_TIMER] = e->basic_attack_timer;
        s[SF_IS_HIT] = e->is_hit;
        s[SF_WAYPOINT] = e->waypoint;
        s[SF_TARGET_PID] = e->target_pid;
        s[SF_TEAM] = e->team;
        s[SF_GRID_ID] = e->grid_id;
        s[SF_LANE] = e->lane;
        s[SF_ENTITY_TYPE] = e->entity_type;
        s[SF_BASIC_ATTACK_CD] = e->basic_attack_cd;
        s[SF_XP_ON_KILL] = e->xp_on_kill;
    }
}

static inline void moba_capture(const MOBA* env, uint64_t rng_state, MobaState* s) {
    memset(s, 0, sizeof(*s));
    s->tick = env->tick;
    s->creep_idx = env->creep_idx;
    s->rng_idx = env->rng->rng_idx;
    s->rng_state = rng_state;
    moba_capture_entities(env, s->entity);
    for (int i = 0; i < NUM_PLAYERS; i++) {
        const Reward* r = &env->reward_components[i];
        s->reward[i][0] = r->death;
        s->reward[i][1] = r->xp;
        s->reward[i][2] = r->distance;
        s->reward[i][3] = r->tower;
    }
    memcpy(s->grid, env->map->grid, sizeof(s->grid));
    for (int i = 0; i < MOBA_MAP * MOBA_MAP; i++)
        s->pids[i] = env->map->pids[i] == -1 ? MOBA_NONE : (uint8_t)env->map->pids[i];
    // Upstream keeps pointers into env->entities and reads each list only up
    // to its first NULL, so that prefix is the whole of the state; anything
    // after it is stale memory that the next scan overwrites before reading.
    for (int r = 0; r < NUM_ENTITIES; r++) {
        memset(s->scan[r], MOBA_NONE, MOBA_SCAN_CAP);
        for (int c = 0; c < 121; c++) {
            Entity* t = env->scanned_targets[r][c];
            if (t == NULL) break;
            if (c >= MOBA_SCAN_CAP - 1) {
                fprintf(stderr, "entity %d: target list longer than MOBA_SCAN_CAP=%d\n", r, MOBA_SCAN_CAP);
                exit(3);
            }
            s->scan[r][c] = (uint8_t)(t - env->entities);
        }
    }
}

// Restore a state into an env built by moba_game_start(). Parts whose bit is
// set in omit are left as the fresh env has them -- used only by the
// necessity test; normal use passes 0.
static inline void moba_restore_partial(MOBA* env, uint64_t* rng_state, const MobaState* s, uint64_t omit) {
#define KEEP(part) (!(omit & (1ULL << (part))))
    if (KEEP(PART_TICK)) env->tick = s->tick;
    if (KEEP(PART_CREEP_IDX)) env->creep_idx = s->creep_idx;
    if (KEEP(PART_RNG_IDX)) env->rng->rng_idx = s->rng_idx;
    if (KEEP(PART_RNG_STATE)) *rng_state = s->rng_state;
    for (int i = 0; i < NUM_ENTITIES; i++) {
        Entity* e = &env->entities[i];
        const float* f = s->entity[i];
        if (KEEP(SF_ALIVE)) e->pid = f[SF_ALIVE] != 0 ? i : -1;
        if (KEEP(SF_X)) e->x = f[SF_X];
        if (KEEP(SF_Y)) e->y = f[SF_Y];
        if (KEEP(SF_HEALTH)) e->health = f[SF_HEALTH];
        if (KEEP(SF_MAX_HEALTH)) e->max_health = f[SF_MAX_HEALTH];
        if (KEEP(SF_MANA)) e->mana = f[SF_MANA];
        if (KEEP(SF_MAX_MANA)) e->max_mana = f[SF_MAX_MANA];
        if (KEEP(SF_DAMAGE)) e->damage = f[SF_DAMAGE];
        if (KEEP(SF_LEVEL)) e->level = (int)f[SF_LEVEL];
        if (KEEP(SF_XP)) e->xp = (int)f[SF_XP];
        if (KEEP(SF_MOVE_MODIFIER)) e->move_modifier = f[SF_MOVE_MODIFIER];
        if (KEEP(SF_STUN_TIMER)) e->stun_timer = (int)f[SF_STUN_TIMER];
        if (KEEP(SF_MOVE_TIMER)) e->move_timer = (int)f[SF_MOVE_TIMER];
        if (KEEP(SF_Q_TIMER)) e->q_timer = (int)f[SF_Q_TIMER];
        if (KEEP(SF_W_TIMER)) e->w_timer = (int)f[SF_W_TIMER];
        if (KEEP(SF_E_TIMER)) e->e_timer = (int)f[SF_E_TIMER];
        if (KEEP(SF_BASIC_ATTACK_TIMER)) e->basic_attack_timer = (int)f[SF_BASIC_ATTACK_TIMER];
        if (KEEP(SF_IS_HIT)) e->is_hit = (int)f[SF_IS_HIT];
        if (KEEP(SF_WAYPOINT)) e->waypoint = (int)f[SF_WAYPOINT];
        if (KEEP(SF_TARGET_PID)) e->target_pid = (int)f[SF_TARGET_PID];
        if (KEEP(SF_TEAM)) e->team = (int)f[SF_TEAM];
        if (KEEP(SF_GRID_ID)) e->grid_id = (int)f[SF_GRID_ID];
        if (KEEP(SF_LANE)) e->lane = (int)f[SF_LANE];
        if (KEEP(SF_ENTITY_TYPE)) e->entity_type = (int)f[SF_ENTITY_TYPE];
        if (KEEP(SF_BASIC_ATTACK_CD)) e->basic_attack_cd = (int)f[SF_BASIC_ATTACK_CD];
        if (KEEP(SF_XP_ON_KILL)) e->xp_on_kill = (int)f[SF_XP_ON_KILL];
    }
    if (KEEP(PART_REWARD))
        for (int i = 0; i < NUM_PLAYERS; i++) {
            Reward* r = &env->reward_components[i];
            r->death = s->reward[i][0];
            r->xp = s->reward[i][1];
            r->distance = s->reward[i][2];
            r->tower = s->reward[i][3];
        }
    if (KEEP(PART_GRID)) memcpy(env->map->grid, s->grid, sizeof(s->grid));
    if (KEEP(PART_PIDS))
        for (int i = 0; i < MOBA_MAP * MOBA_MAP; i++)
            env->map->pids[i] = s->pids[i] == MOBA_NONE ? -1 : s->pids[i];
    if (KEEP(PART_SCAN))
        for (int r = 0; r < NUM_ENTITIES; r++)
            for (int c = 0; c < MOBA_SCAN_CAP; c++) {
                uint8_t t = s->scan[r][c];
                env->scanned_targets[r][c] = t == MOBA_NONE ? NULL : &env->entities[t];
                if (t == MOBA_NONE) break;
            }
    env->should_reset = false;
    compute_observations(env);
#undef KEEP
}

static inline void moba_restore(MOBA* env, uint64_t* rng_state, const MobaState* s) {
    moba_restore_partial(env, rng_state, s, 0);
}

// Ego observations for all 10 heroes, byte-exact with upstream
// compute_observations(). Quirks reproduced deliberately:
//   - channel c of window cell i is written at byte i+c, not 4*i+c, so the
//     next cell's tile overwrites it: bytes 0..120 are tile ids, 121..123 are
//     leftovers from the last cells, 124..483 are always 0;
//   - the window is read at flat index y*128+x with no bounds check, so a
//     window past the left/right edge wraps into the adjacent row;
//   - floats are narrowed to unsigned char by truncation.
static inline void moba_ego_obs(const MobaState* s, const MobaStatic st[MOBA_HEROES],
                                int vision_range, uint8_t out[MOBA_HEROES][MOBA_OBS]) {
    memset(out, 0, MOBA_HEROES * MOBA_OBS);
    int vis = vision_range;
    for (int pid = 0; pid < MOBA_HEROES; pid++) {
        uint8_t* obs_map = out[pid];
        uint8_t* obs_extra = &out[pid][11 * 11 * 4];
        const float* p = s->entity[pid];
        const float* rw = s->reward[pid];

        int y = (int)p[SF_Y];
        int x = (int)p[SF_X];
        obs_extra[0] = x;
        obs_extra[1] = y;
        obs_extra[2] = (int)p[SF_LEVEL];
        obs_extra[3] = 10 * p[SF_HEALTH] / p[SF_MAX_HEALTH];
        obs_extra[4] = 10 * p[SF_MANA] / p[SF_MAX_MANA];
        obs_extra[5] = p[SF_DAMAGE] / 50.0;
        obs_extra[6] = st[pid].move_speed;
        obs_extra[7] = p[SF_MOVE_MODIFIER];
        obs_extra[8] = (int)p[SF_STUN_TIMER];
        obs_extra[9] = (int)p[SF_MOVE_TIMER];
        obs_extra[10] = (int)p[SF_Q_TIMER];
        obs_extra[11] = (int)p[SF_W_TIMER];
        obs_extra[12] = (int)p[SF_E_TIMER];
        obs_extra[13] = (int)p[SF_BASIC_ATTACK_TIMER];
        obs_extra[14] = (int)p[SF_BASIC_ATTACK_CD];
        obs_extra[15] = (int)p[SF_IS_HIT];
        obs_extra[16] = (int)p[SF_TEAM];
        obs_extra[17 + st[pid].hero_type] = 1;
        obs_extra[22] = rw[0] != 0;
        obs_extra[23] = rw[1] != 0;
        obs_extra[24] = rw[2] != 0;
        obs_extra[25] = rw[3] != 0;

        for (int dy = -vis; dy <= vis; dy++) {
            for (int dx = -vis; dx <= vis; dx++) {
                int map_idx = (dy + vis) * 11 + (dx + vis);
                int adr = (y + dy) * MOBA_MAP + (x + dx);
                obs_map[map_idx] = s->grid[adr];
                int t = s->pids[adr];
                if (t == MOBA_NONE)
                    continue;
                const float* e = s->entity[t];
                obs_map[map_idx + 1] = 10 * e[SF_HEALTH] / e[SF_MAX_HEALTH];
                if (e[SF_MAX_MANA] > 0)
                    obs_map[map_idx + 2] = 10 * e[SF_MANA] / e[SF_MAX_MANA];
                obs_map[map_idx + 3] = (int)e[SF_LEVEL];
            }
        }
    }
}

// One hero's 6-head action <-> a code in [0, 1176).
static inline uint16_t moba_pack_action(const float* a) {
    int code = 0;
    for (int h = 0; h < MOBA_ATNS; h++)
        code = code * MOBA_ACTION_SIZES[h] + (int)a[h];
    return (uint16_t)code;
}

static inline void moba_unpack_action(uint16_t code, float* a) {
    for (int h = MOBA_ATNS - 1; h >= 0; h--) {
        a[h] = (float)(code % MOBA_ACTION_SIZES[h]);
        code /= MOBA_ACTION_SIZES[h];
    }
}

static inline uint64_t moba_fnv1a(uint64_t h, const void* data, size_t n) {
    const uint8_t* p = (const uint8_t*)data;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x100000001b3ULL; }
    return h;
}

#define MOBA_FNV_INIT 0xcbf29ce484222325ULL

#endif
