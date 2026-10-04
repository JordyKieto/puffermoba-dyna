// Counterfactual fork generator: amplify a recorded corpus with the real sim.
//
// For each game (seed + recorded action codes) the game is replayed from the
// start. At fork ticks (t % stride == phase) the state S_t is captured and K
// counterfactual siblings are made: S_t is restored into a scratch game, the
// recorded actions of tick t get ONE edit (one hero, one action head, a new
// value), and the scratch game is stepped once. Sibling 0 is the recorded
// action itself (the real next state). Outputs, per fork point:
//
//   hist     the `burn` real states before S_t (+ tick, creep_idx, action codes),
//            to warm the world model's recurrent memory before it branches
//   in       the real state S_t (206 x 26 floats) + tick + creep_idx
//   codes    (K+1) x 10 action codes (sibling 0 = recorded)
//   next     (K+1) x 206 x 26 next states + tick + creep_idx
//
// Single-threaded on purpose: moba_game.h keeps the game's random stream in a
// process-global, so callers parallelize with processes (mobawm/wm/cf.py).
// Built as a shared library (tools/build.sh moba-fork) and called via ctypes.
//
// check=1 also forks sibling 0 through restore + step and requires it to equal
// the real continuation byte-for-byte (entities, tick, creep_idx): the proof
// that restoring onto a reused scratch game is exact.

#include "moba_game.h"

#include <stdint.h>
#include <string.h>

#define NE NUM_ENTITIES
#define NF MOBA_STATE_FIELDS

static uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// One edit: head k, then hero h, then a new value != old for that head. Heads:
// 0 move_y, 1 move_x, 2 target mode, 3 q, 4 w, 5 e. Head weights favour
// abilities and attacks, the rare effects the world model needs most. For an
// ability head the hero is drawn among living heroes whose ability is off
// cooldown (else any hero): an edit on a cooling-down ability changes nothing.
static void pick_edit(const MOBA* env, uint64_t* r, int* hero, int* head) {
    static const int W[MOBA_ATNS] = {2, 2, 2, 3, 3, 3};   // sum 15
    uint64_t x = mix64(*r += 0x9E3779B97F4A7C15ULL);
    int pick = (int)(x % 15), k = 0;
    while (pick >= W[k]) pick -= W[k++];
    int ready[MOBA_HEROES], n = 0;
    if (k >= 3)
        for (int i = 0; i < MOBA_HEROES; i++) {
            const Entity* e = &env->entities[i];
            int timer = k == 3 ? e->q_timer : k == 4 ? e->w_timer : e->e_timer;
            if (e->pid != -1 && timer == 0) ready[n++] = i;
        }
    *head = k;
    *hero = n ? ready[(x >> 24) % (uint64_t)n] : (int)((x >> 24) % MOBA_HEROES);
}

static uint16_t edit_code(uint16_t code, int k, uint64_t* r) {
    float a[MOBA_ATNS];
    moba_unpack_action(code, a);
    uint64_t x = mix64(*r += 0x9E3779B97F4A7C15ULL);
    int n = MOBA_ACTION_SIZES[k];
    a[k] = (float)(((int)a[k] + 1 + (int)(x % (uint64_t)(n - 1))) % n);   // uniform over the other values
    return moba_pack_action(a);
}

// moba_restore_partial(omit = 0) and moba_step_env without compute_observations:
// the world model never reads the ego observations, and they are pure outputs
// (nothing in a step reads them). check=1 compares this path against the real
// upstream step every fork point, so any divergence is caught.
static void fork_restore(MOBA* env, uint64_t* rng_state, const MobaState* s) {
    env->tick = s->tick;
    env->creep_idx = s->creep_idx;
    env->rng->rng_idx = s->rng_idx;
    *rng_state = s->rng_state;
    for (int i = 0; i < NUM_ENTITIES; i++) {
        Entity* e = &env->entities[i];
        const float* f = s->entity[i];
        e->pid = f[SF_ALIVE] != 0 ? i : -1;
        e->x = f[SF_X]; e->y = f[SF_Y];
        e->health = f[SF_HEALTH]; e->max_health = f[SF_MAX_HEALTH];
        e->mana = f[SF_MANA]; e->max_mana = f[SF_MAX_MANA];
        e->damage = f[SF_DAMAGE];
        e->level = (int)f[SF_LEVEL]; e->xp = (int)f[SF_XP];
        e->move_modifier = f[SF_MOVE_MODIFIER];
        e->stun_timer = (int)f[SF_STUN_TIMER]; e->move_timer = (int)f[SF_MOVE_TIMER];
        e->q_timer = (int)f[SF_Q_TIMER]; e->w_timer = (int)f[SF_W_TIMER]; e->e_timer = (int)f[SF_E_TIMER];
        e->basic_attack_timer = (int)f[SF_BASIC_ATTACK_TIMER];
        e->is_hit = (int)f[SF_IS_HIT]; e->waypoint = (int)f[SF_WAYPOINT];
        e->target_pid = (int)f[SF_TARGET_PID]; e->team = (int)f[SF_TEAM];
        e->grid_id = (int)f[SF_GRID_ID]; e->lane = (int)f[SF_LANE];
        e->entity_type = (int)f[SF_ENTITY_TYPE];
        e->basic_attack_cd = (int)f[SF_BASIC_ATTACK_CD]; e->xp_on_kill = (int)f[SF_XP_ON_KILL];
    }
    for (int i = 0; i < NUM_PLAYERS; i++) {
        Reward* r = &env->reward_components[i];
        r->death = s->reward[i][0]; r->xp = s->reward[i][1];
        r->distance = s->reward[i][2]; r->tower = s->reward[i][3];
    }
    memcpy(env->map->grid, s->grid, sizeof(s->grid));
    for (int i = 0; i < MOBA_MAP * MOBA_MAP; i++)
        env->map->pids[i] = s->pids[i] == MOBA_NONE ? -1 : s->pids[i];
    for (int r = 0; r < NUM_ENTITIES; r++)
        for (int c = 0; c < MOBA_SCAN_CAP; c++) {
            uint8_t t = s->scan[r][c];
            env->scanned_targets[r][c] = t == MOBA_NONE ? NULL : &env->entities[t];
            if (t == MOBA_NONE) break;
        }
    env->should_reset = false;
}

static void fork_step(MobaGame* g) {
    MOBA* env = &g->env;
    moba_rng_current = g->rng;
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
    g->rng = moba_rng_current;
}

// Scratch game shared by all forks in this process: built once.
static MobaGame scratch;
static MobaGame real;
static int built = 0;
static MobaState* S_buf = NULL;

static void ensure_built(void) {
    if (built) return;
    moba_game_alloc(&real);
    moba_game_alloc(&scratch);
    moba_game_start(&scratch, 1);
    S_buf = calloc(1, sizeof(MobaState));
    built = 1;
}

static void set_codes(MOBA* env, const uint16_t* codes) {
    for (int a = 0; a < MOBA_HEROES; a++)
        moba_unpack_action(codes[a], &env->actions[a * MOBA_ATNS]);
}

// Returns the number of fork points written (<= cap), or -1 on a check failure.
// codes: all games' recorded codes concatenated, code_off[g] = start of game g
// (n_ticks[g] rows of 10). Fork ticks: t % stride == phase and t < n_ticks - 1
// (so every fork has a real next state that is not terminal).
#define MF_API __attribute__((visibility("default")))

#define MAX_BURN 16

MF_API int64_t mf_run(int n_games, const uint64_t* seeds, const int32_t* n_ticks, const int64_t* code_off,
               const uint16_t* codes, int stride, int phase, int K, uint64_t edit_seed, int check,
               int64_t cap, int burn,
               float* out_hist, int32_t* out_hist_tick, int32_t* out_hist_cidx, uint16_t* out_hist_codes,
               float* out_in, int32_t* out_in_tick, int32_t* out_in_cidx,
               uint16_t* out_codes, int8_t* out_edit_hero, int8_t* out_edit_head,
               float* out_next, int32_t* out_next_tick, int32_t* out_next_cidx,
               int32_t* out_game, int32_t* out_t) {
    ensure_built();
    if (burn < 0 || burn > MAX_BURN) return -2;
    const int S = K + 1;
    static float ring[MAX_BURN][NE][NF];
    static int32_t ring_tick[MAX_BURN], ring_cidx[MAX_BURN];
    int64_t n = 0;
    uint16_t sib[MOBA_HEROES];
    float ent_real[NE][NF], ent_fork[NE][NF];
    for (int g = 0; g < n_games && n < cap; g++) {
        moba_game_start(&real, seeds[g]);
        const uint16_t* gc = codes + code_off[g];
        for (int t = 0; t < n_ticks[g] && n < cap; t++) {
            const uint16_t* ct = gc + (int64_t)t * MOBA_HEROES;
            int fork = (t % stride == phase) && t < n_ticks[g] - 1 && t >= burn;
            if (fork) {
                for (int b = 0; b < burn; b++) {        // ticks t-burn .. t-1, oldest first
                    int tt = t - burn + b, slot = tt % MAX_BURN;
                    memcpy(out_hist + (n * burn + b) * NE * NF, ring[slot], sizeof(ring[slot]));
                    out_hist_tick[n * burn + b] = ring_tick[slot];
                    out_hist_cidx[n * burn + b] = ring_cidx[slot];
                    memcpy(out_hist_codes + (n * burn + b) * MOBA_HEROES, gc + (int64_t)tt * MOBA_HEROES,
                           sizeof(uint16_t) * MOBA_HEROES);
                }
                moba_game_capture(&real, S_buf);
                memcpy(out_in + n * NE * NF, S_buf->entity, sizeof(float) * NE * NF);
                out_in_tick[n] = S_buf->tick;
                out_in_cidx[n] = S_buf->creep_idx;
                out_game[n] = g;
                out_t[n] = t;
                uint64_t r = mix64(edit_seed ^ ((uint64_t)g << 32) ^ (uint64_t)t);
                for (int s = 1; s < S; s++) {
                    memcpy(sib, ct, sizeof(sib));
                    int h, head;
                    pick_edit(&real.env, &r, &h, &head);
                    sib[h] = edit_code(sib[h], head, &r);
                    fork_restore(&scratch.env, &scratch.rng, S_buf);
                    set_codes(&scratch.env, sib);
                    fork_step(&scratch);
                    int64_t o = n * S + s;
                    memcpy(out_codes + o * MOBA_HEROES, sib, sizeof(sib));
                    out_edit_hero[o] = (int8_t)h;
                    out_edit_head[o] = (int8_t)head;
                    moba_capture_entities(&scratch.env, (float(*)[NF])(out_next + o * NE * NF));
                    out_next_tick[o] = scratch.env.tick;
                    out_next_cidx[o] = scratch.env.creep_idx;
                }
                if (check) {        // sibling 0 through the fork path, compared below
                    fork_restore(&scratch.env, &scratch.rng, S_buf);
                    set_codes(&scratch.env, ct);
                    fork_step(&scratch);
                    moba_capture_entities(&scratch.env, ent_fork);
                }
            }
            if (burn) {                                 // this tick's state, for later forks' history
                int slot = t % MAX_BURN;
                moba_capture_entities(&real.env, ring[slot]);
                ring_tick[slot] = real.env.tick;
                ring_cidx[slot] = real.env.creep_idx;
            }
            set_codes(&real.env, ct);
            moba_game_step(&real);
            if (fork) {
                int64_t o = n * S;
                memcpy(out_codes + o * MOBA_HEROES, ct, sizeof(uint16_t) * MOBA_HEROES);
                out_edit_hero[o] = -1;
                out_edit_head[o] = -1;
                moba_capture_entities(&real.env, ent_real);
                memcpy(out_next + o * NE * NF, ent_real, sizeof(ent_real));
                out_next_tick[o] = real.env.tick;
                out_next_cidx[o] = real.env.creep_idx;
                if (check && (memcmp(ent_real, ent_fork, sizeof(ent_real)) != 0
                              || scratch.env.tick != real.env.tick || scratch.env.creep_idx != real.env.creep_idx)) {
                    fprintf(stderr, "mf_run: fork of the recorded action differs from the real step "
                                    "(game %d tick %d)\n", g, t);
                    return -1;
                }
                n++;
            }
        }
    }
    return n;
}

MF_API int mf_sizes(int* ne, int* nf, int* state_bytes) {
    *ne = NE; *nf = NF; *state_bytes = (int)sizeof(MobaState);
    return 0;
}

// ---------------------------------------------------------------------------
// Scenario mining (tower deaths). For one game (seed + recorded codes) the game is
// replayed once; every tower/ancient death at tick te (alive before, dead after the
// step te -> te+1) with te >= MS_PRE is an event. For each event the state at
// t0 = te - MS_PRE is restored into the scratch game and VARIANTS are simulated for
// MS_LEN ticks with the real sim, until the event has quota_die slices where the same
// tower dies inside the window and quota_live where it survives (or max_attempts).
// Variation, scaled by an adaptive strength s in [0,1] (up when survivals are missing,
// down when deaths are missing):
//   rng      a new random stream (always)
//   state    heroes moved by up to +-2 cells (prob 0.5 s, via move_to so the map stays
//            consistent), hero health x (1 +- 0.1 s), the target tower's health x (1 +- 0.05)
//   actions  each hero, with prob 0.6 s, plays uniform-random codes over a random stretch
//            of the window; the rest play the recorded codes
// Output per slice: the perturbed start state (MobaState bytes), the MS_LEN x 10 codes
// played, the length (shorter only when an ancient falls), and the outcome.
#define MS_PRE 96
#define MS_POST 32
#define MS_LEN (MS_PRE + MS_POST)
#define MS_RING (MS_PRE + 1)
#define MS_NCODES 1176                      // 7 * 7 * 3 * 2 * 2 * 2 action combinations

static double ms_u(uint64_t* r) { *r = mix64(*r + 0x9E3779B97F4A7C15ULL); return (*r >> 11) * (1.0 / 9007199254740992.0); }

MF_API int64_t ms_run(uint64_t seed, int n_ticks, const uint16_t* codes, int quota_die, int quota_live,
                      int max_attempts, uint64_t var_seed, int64_t cap,
                      uint8_t* out_state, uint16_t* out_codes, int32_t* out_len, int32_t* out_te,
                      int32_t* out_tower, int8_t* out_died, int32_t* out_death_k, float* out_strength,
                      int32_t* out_events) {
    ensure_built();
    static MobaState ring[MS_RING];
    static uint16_t win[MS_LEN * MOBA_HEROES];
    int64_t n = 0;
    int events = 0;
    moba_game_start(&real, seed);
    for (int t = 0; t < n_ticks - 1 && n < cap; t++) {
        moba_game_capture(&real, &ring[t % MS_RING]);
        int alive0[24];
        for (int k = 0; k < 24; k++) alive0[k] = real.env.entities[TOWER_OFFSET + k].pid != -1;
        set_codes(&real.env, codes + (int64_t)t * MOBA_HEROES);
        fork_step(&real);
        if (t < MS_PRE) continue;
        for (int tw = 0; tw < 24 && n < cap; tw++) {
            if (!alive0[tw] || real.env.entities[TOWER_OFFSET + tw].pid != -1) continue;
            events++;                                       // tower tw fell on the step t -> t+1
            const MobaState* st0 = &ring[(t - MS_PRE) % MS_RING];
            int t0 = t - MS_PRE;
            int n_die = 0, n_live = 0;
            double s = 0.5;
            uint64_t r = mix64(var_seed ^ seed ^ ((uint64_t)t << 20) ^ (uint64_t)tw);
            for (int a = 0; a < max_attempts && (n_die < quota_die || n_live < quota_live) && n < cap; a++) {
                if (n_die >= quota_die && s < 1.0) s = s + 0.05 > 1.0 ? 1.0 : s + 0.05;
                if (n_live >= quota_live && s > 0.0) s = s - 0.05 < 0.0 ? 0.0 : s - 0.05;
                MOBA* env = &scratch.env;
                fork_restore(env, &scratch.rng, st0);
                scratch.rng = mix64(r ^ 0xA5A5A5A5ULL);
                for (int h = 0; h < MOBA_HEROES; h++) {
                    Entity* e = &env->entities[h];
                    if (e->pid == -1) continue;
                    if (ms_u(&r) < 0.5 * s) {
                        float dy = (float)((int)(ms_u(&r) * 5) - 2), dx = (float)((int)(ms_u(&r) * 5) - 2);
                        move_to(env->map, e, e->y + dy, e->x + dx);
                    }
                    float hp = e->health * (float)(1.0 + (ms_u(&r) * 2 - 1) * 0.1 * s);
                    e->health = hp < 1 ? 1 : (hp > e->max_health ? e->max_health : hp);
                }
                Entity* T = &env->entities[TOWER_OFFSET + tw];
                float th = T->health * (float)(1.0 + (ms_u(&r) * 2 - 1) * 0.05);
                T->health = th < 1 ? 1 : (th > T->max_health ? T->max_health : th);
                moba_game_capture(&scratch, (MobaState*)(out_state + n * sizeof(MobaState)));
                // actions: recorded, with random stretches for some heroes
                for (int k = 0; k < MS_LEN; k++) {
                    int tt = t0 + k < n_ticks ? t0 + k : n_ticks - 1;
                    memcpy(win + k * MOBA_HEROES, codes + (int64_t)tt * MOBA_HEROES, sizeof(uint16_t) * MOBA_HEROES);
                }
                for (int h = 0; h < MOBA_HEROES; h++) {
                    if (ms_u(&r) >= 0.6 * s) continue;
                    int lo = (int)(ms_u(&r) * MS_LEN), hi = lo + 1 + (int)(ms_u(&r) * (MS_LEN - lo));
                    for (int k = lo; k < hi && k < MS_LEN; k++) win[k * MOBA_HEROES + h] = (uint16_t)(ms_u(&r) * MS_NCODES);
                }
                int died = 0, death_k = -1, len = MS_LEN;
                for (int k = 0; k < MS_LEN; k++) {
                    set_codes(env, win + k * MOBA_HEROES);
                    fork_step(&scratch);
                    if (!died && T->pid == -1) {
                        died = 1; death_k = k;
                        if (tw >= 22) { len = k + 1; break; }     // an ancient: the game is over
                    }
                }
                if (died ? n_die >= quota_die : n_live >= quota_live) continue;   // class already full
                memcpy(out_codes + n * MS_LEN * MOBA_HEROES, win, sizeof(win));
                out_len[n] = len; out_te[n] = t; out_tower[n] = tw; out_died[n] = (int8_t)died;
                out_death_k[n] = death_k; out_strength[n] = (float)s;
                if (died) n_die++; else n_live++;
                n++;
            }
        }
    }
    *out_events = events;
    return n;
}

// Regenerate a slice: restore its start state and play its codes; writes (len+1) x NE x NF
// entity states (start included) + ticks + creep_idx. Returns the tick at which `tower`
// (0..23) died (-1 if it survived), so a caller can verify the stored outcome.
// ms_replay plus every tick's target lists (out_scan: (len+1) x NE x MOBA_SCAN_CAP, MOBA_NONE-terminated),
// for models whose state carries the stored scan lists (mobawm.wm.data LIST_STATE).
MF_API int ms_replay_scan(const uint8_t* state, const uint16_t* codes, int len, int tower,
                          float* out_ent, int32_t* out_tick, int32_t* out_cidx, uint8_t* out_scan) {
    ensure_built();
    MOBA* env = &scratch.env;
    fork_restore(env, &scratch.rng, (const MobaState*)state);
    scratch.rng = ((const MobaState*)state)->rng_state;
    int death_k = -1;
    for (int k = 0; k <= len; k++) {
        moba_game_capture(&scratch, S_buf);
        memcpy(out_ent + (int64_t)k * NE * NF, S_buf->entity, sizeof(float) * NE * NF);
        memcpy(out_scan + (int64_t)k * NE * MOBA_SCAN_CAP, S_buf->scan, (size_t)NE * MOBA_SCAN_CAP);
        out_tick[k] = S_buf->tick; out_cidx[k] = S_buf->creep_idx;
        if (k == len) break;
        set_codes(env, codes + (int64_t)k * MOBA_HEROES);
        fork_step(&scratch);
        if (death_k < 0 && tower >= 0 && env->entities[TOWER_OFFSET + tower].pid == -1) death_k = k;
    }
    return death_k;
}

MF_API int ms_replay(const uint8_t* state, const uint16_t* codes, int len, int tower,
                     float* out_ent, int32_t* out_tick, int32_t* out_cidx) {
    ensure_built();
    MOBA* env = &scratch.env;
    fork_restore(env, &scratch.rng, (const MobaState*)state);
    scratch.rng = ((const MobaState*)state)->rng_state;
    int death_k = -1;
    for (int k = 0; k <= len; k++) {
        moba_game_capture(&scratch, S_buf);
        memcpy(out_ent + (int64_t)k * NE * NF, S_buf->entity, sizeof(float) * NE * NF);
        out_tick[k] = S_buf->tick; out_cidx[k] = S_buf->creep_idx;
        if (k == len) break;
        set_codes(env, codes + (int64_t)k * MOBA_HEROES);
        fork_step(&scratch);
        if (death_k < 0 && tower >= 0 && env->entities[TOWER_OFFSET + tower].pid == -1) death_k = k;
    }
    return death_k;
}
