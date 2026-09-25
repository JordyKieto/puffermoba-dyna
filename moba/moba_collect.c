// moba_collect - MOBA game collection, full-state replay, and the Markov test.
//
//   build/moba_collect collect --first-game 0 --games 100 --mix --out games.0000000.mbl
//   build/moba_collect collect --games 10 --radiant net --dire script --out x.mbl
//   build/moba_collect replay x.mbl                          # verify only
//   build/moba_collect replay x.mbl --dense x.mbs            # + full state of every tick
//   build/moba_collect markov x.mbl                          # state sufficiency test
//
// All 10 heroes act through the env's 6-head action interface
// (script_opponents = 0), so every tick records 10 actions. Each team is
// driven by one of two controllers:
//   net     the upstream pretrained policy (resources/moba/moba_weights.bin);
//   script  upstream creep_ai's hero behaviour -- chase the nearest non-neutral
//           enemy within 7 cells, else walk the lane -- re-expressed as
//           actions so it is recorded like any other agent (scripted_action).
// --mix cycles game configs by global game index: net/net, net/script,
// script/net (radiant/dire).
//
// collect stores a game as its seed, its config, the 10 action codes per tick
// (zlib) and a hash of the state over every tick including the terminal one.
// replay re-runs each game from that with no controller in the loop, checks
// the hash, checks moba_ego_obs() against the env's own observation on every
// tick, and with --dense writes the full MobaState of every tick in order,
// ending each game with its terminal state.
//
// Determinism. The env's randomness is a per-game SplitMix64 stream whose
// state is part of MobaState (moba_game.h), so a game depends on nothing but
// its seed and its actions, and a state restored mid-game continues exactly.
// The net samples from its own RNG, and the scripted controller reads the
// env without writing to it.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <zlib.h>
#include <unistd.h>
#include <sys/wait.h>

#include "moba_game.h"
#include "puffernet.h"

#define LOG_MAGIC 0x4D424C33u    /* "MBL3" */
#define DENSE_MAGIC 0x4D425333u  /* "MBS3" */
#define FORMAT_VERSION 3

#define CTRL_NET 0
#define CTRL_SCRIPT 1
static const char* CTRL_NAMES[] = {"net", "script"};

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t n_games;
    uint32_t max_ticks;
    uint64_t base_seed;
    uint32_t first_game;      // global index of this file's game 0
    uint32_t vision_range;
    float    agent_speed;
    float    reward_death;
    float    reward_xp;
    float    reward_distance;
    float    reward_tower;
    float    epsilon;
    uint64_t weights_hash;
    uint32_t reserved[4];
} MobaLogHeader;

typedef struct {
    uint64_t game_seed;
    uint32_t game_index;      // global
    uint32_t n_ticks;         // steps taken; the dense replay has n_ticks + 1 records
    uint8_t  winner;          // 0 none (truncated), 1 radiant, 2 dire, 3 both ancients fell
    uint8_t  truncated;
    uint8_t  radiant_ctrl;    // CTRL_*
    uint8_t  dire_ctrl;
    uint32_t zbytes;          // compressed uint16[n_ticks][10] that follows
    uint64_t state_hash;      // FNV-1a over the MobaState of every tick, terminal included
} MobaGameHeader;

// Dense file: MobaDenseHeader, then per game a MobaDenseGame followed by
// zlib(MobaDenseRecord[n_records]). Full states are ~72 KB/tick but nearly
// constant tick to tick (grid, occupancy, target lists), so they compress
// well; readers decompress one game at a time.
//
// Dense record: one tick, in game order. A game of n steps has n + 1 records:
// records 0..n-1 are (s_t, a_t, r_t) with everything captured before the step
// and done set on the step that ends the game; record n is the terminal state
// (final = 1; no action, no reward). So every non-final record's successor
// s_{t+1} is the next record.
typedef struct {
    uint32_t game_index;
    int32_t  tick;
    uint8_t  done;            // this step ended the game
    uint8_t  truncated;
    uint8_t  winner;          // set on the done record and the final record
    uint8_t  final;           // terminal state: no action follows
    uint8_t  radiant_ctrl;
    uint8_t  dire_ctrl;
    uint8_t  pad[2];
    uint8_t  actions[MOBA_HEROES][MOBA_ATNS];
    float    rewards[MOBA_HEROES];
    uint8_t  obs[MOBA_HEROES][MOBA_OBS];      // env's own obs, kept as a check
    MobaState state;
} MobaDenseRecord;

_Static_assert(sizeof(MobaDenseRecord) == 16 + MOBA_HEROES * MOBA_ATNS + 4 * MOBA_HEROES
               + MOBA_HEROES * MOBA_OBS + sizeof(MobaState),
               "MobaDenseRecord has implicit padding; mirror it in mobawm/shards.py");

typedef struct {
    uint32_t game_index;
    uint32_t n_records;       // n_ticks + 1 (the last is the terminal state)
    uint64_t zbytes;
} MobaDenseGame;

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t record_bytes;
    uint32_t n_records;       // total over all games
    uint32_t n_games;
    uint32_t state_bytes;
    uint32_t vision_range;
    uint32_t scan_cap;
    uint32_t state_fields;
    uint32_t reserved;
    MobaStatic statics[MOBA_HEROES];
} MobaDenseHeader;

static uint64_t splitmix64(uint64_t* x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static float rng_uniform(uint64_t* s) {
    return (splitmix64(s) >> 40) * (1.0f / 16777216.0f);
}

static uint64_t game_seed_for(uint64_t base_seed, uint32_t game_index) {
    uint64_t x = base_seed ^ ((uint64_t)game_index * 0xD1B54A32D192ED03ULL);
    return splitmix64(&x);
}

static void net_reset(PufferNet* net) {
    memset(net->mingru->state, 0,
           net->mingru->num_layers * net->mingru->batch_size * net->mingru->hidden_size * sizeof(float));
}

// Upstream forward_puffernet for all 10 heroes, sampling on our RNG instead
// of rand(). The weights were trained as radiant; dire runs the same network
// on its own observations.
static void net_act(PufferNet* net, const uint8_t* obs, float* obs_f, float* actions, uint64_t* rng) {
    for (int i = 0; i < MOBA_HEROES * MOBA_OBS; i++)
        obs_f[i] = (float)obs[i];
    linear(net->encoder, obs_f);
    mingru(net->mingru, net->encoder->output);
    linear(net->decoder, net->mingru->output);
    const float* logits = net->decoder->output;
    int atn_sum = 0;
    for (int h = 0; h < MOBA_ATNS; h++) atn_sum += MOBA_ACTION_SIZES[h];
    for (int b = 0; b < MOBA_HEROES; b++) {
        int in_adr = b * (atn_sum + 1);   // +1: fused value head
        for (int h = 0; h < MOBA_ATNS; h++) {
            int n = MOBA_ACTION_SIZES[h];
            float sum = 0;
            for (int i = 0; i < n; i++) sum += expf(logits[in_adr + i]);
            float u = rng_uniform(rng), acc = 0, a = 0;
            for (int i = 0; i < n; i++) {
                acc += expf(logits[in_adr + i]) / sum;
                if (u < acc) { a = (float)i; break; }
            }
            actions[b * MOBA_ATNS + h] = a;
            in_adr += n;
        }
    }
}

// Upstream creep_ai's decision for a hero, emitted as an action:
//   - nearest enemy that is not a neutral within a radius-7 window (creep_ai
//     scans CREEP_VISION excluding friendlies and neutrals) -> step toward it;
//   - otherwise step toward the next waypoint of the hero's lane;
//   - the step direction is the BFS first move that creep_ai's move_towards
//     takes, mapped to the move heads (-1/0/+1 -> 0/3/6);
//   - target mode 1 (attack the nearest hostile in vision), no skills --
//     creep_ai never casts.
// It differs from creep_ai in mechanics, not intent: movement goes through the
// player move path (no jitter when blocked), attacks through the player attack
// path (any hostile within vision 5), and it rescans every tick rather than
// every 5. It reads the env and writes nothing but the pathing cache (walls
// only, so the cache is identical whoever fills it); its lane progress lives
// in ScriptMemory.
// Controller memory, like the net's recurrent state: not env state, so replay
// (which only feeds recorded actions) never needs it.
typedef struct {
    int waypoint[MOBA_HEROES];
    float deaths[MOBA_HEROES];
} ScriptMemory;

static void script_reset(ScriptMemory* m) {
    for (int i = 0; i < MOBA_HEROES; i++) { m->waypoint[i] = 1; m->deaths[i] = 0; }
}

static void scripted_action(MOBA* env, int pid, float* act, ScriptMemory* mem) {
    Entity* h = &env->entities[pid];
    int hy = (int)h->y, hx = (int)h->x;
    if (env->player_logs[pid].deaths != mem->deaths[pid]) {   // respawned
        mem->deaths[pid] = env->player_logs[pid].deaths;
        mem->waypoint[pid] = 1;
    }

    int best = -1;
    float best_d = 1e9f;
    for (int y = hy - CREEP_VISION; y <= hy + CREEP_VISION; y++) {
        for (int x = hx - CREEP_VISION; x <= hx + CREEP_VISION; x++) {
            if (y < 0 || y >= 128 || x < 0 || x >= 128) continue;
            int t = env->map->pids[map_offset(env->map, y, x)];
            if (t == -1) continue;
            Entity* e = &env->entities[t];
            if (e->team == h->team || e->entity_type == ENTITY_NEUTRAL) continue;
            float d = l1_distance(h->y, h->x, e->y, e->x);
            if (d < best_d) { best_d = d; best = t; }
        }
    }

    int dst_y, dst_x;
    if (best >= 0) {
        dst_y = (int)env->entities[best].y;
        dst_x = (int)env->entities[best].x;
    } else {
        // Lane walking exactly as creep_ai: a waypoint index that starts at 1
        // on (re)spawn and advances within L1 distance 2. It has to be memory:
        // some lanes double back (lane 2 runs x 37 -> 32.5 -> 68.7), so no
        // rule based on position alone can tell which way is forward.
        int lane = h->lane, n = WAYPOINTS_N[lane];
        int* wp = &mem->waypoint[pid];
        float dy0 = WAYPOINTS[lane][*wp][0], dx0 = WAYPOINTS[lane][*wp][1];
        if (l1_distance(h->y, h->x, dy0, dx0) < 2 && *wp < n - 1) (*wp)++;
        dst_y = (int)WAYPOINTS[lane][*wp][0];
        dst_x = (int)WAYPOINTS[lane][*wp][1];
    }

    int adr = ai_offset(dst_y, dst_x, hy, hx);
    int atn = env->ai_paths[adr];
    if (atn == 255) {
        bfs(env->map, &env->ai_paths[ai_offset(dst_y, dst_x, 0, 0)], env->ai_path_buffer, dst_y, dst_x);
        atn = env->ai_paths[adr];
    }
    act[0] = 3;
    act[1] = 3;
    act[2] = 1;
    act[3] = act[4] = act[5] = 0;
    if (atn >= 8) return;

    // creep_ai steps whole cells from an integer position; a hero moves by a
    // continuous velocity from a fractional one, so the naive heading can land
    // in a side cell that is a wall or occupied, where move_to refuses -- and
    // an unchanged state repeats the same refused move forever. So pick the
    // move head pair whose landing point, computed exactly as step_players
    // will, is closest to the centre of the BFS next cell and not blocked.
    float ty = hy + ATN_MAP[0][atn] + 0.5f, tx = hx + ATN_MAP[1][atn] + 0.5f;
    float mm = h->move_timer <= 1 ? 1.0f : h->move_modifier;   // after update_status
    float best_score = (h->y - ty) * (h->y - ty) + (h->x - tx) * (h->x - tx);
    for (int ay = 0; ay < 7; ay++) {
        for (int ax = 0; ax < 7; ax++) {
            float vy = (ay - 3.0f) / 3.0f, vx = (ax - 3.0f) / 3.0f;
            float mag = sqrtf(vy * vy + vx * vx);
            if (mag > 1) { vy /= mag; vx /= mag; }
            float dy = h->y + mm * env->agent_speed * vy;
            float dx = h->x + mm * env->agent_speed * vx;
            int cell = map_offset(env->map, (int)dy, (int)dx);
            if (env->map->grid[cell] != EMPTY && env->map->pids[cell] != pid) continue;
            float score = (dy - ty) * (dy - ty) + (dx - tx) * (dx - tx);
            if (score < best_score) { best_score = score; act[0] = ay; act[1] = ax; }
        }
    }
}

static uint64_t hash_state(uint64_t h, const MobaGame* g, MobaState* buf) {
    moba_game_capture(g, buf);
    return moba_fnv1a(h, buf, sizeof(*buf));
}

static int parse_ctrl(const char* s) {
    if (!strcmp(s, "net")) return CTRL_NET;
    if (!strcmp(s, "script")) return CTRL_SCRIPT;
    return -1;
}

static void usage(void) {
    fprintf(stderr,
        "usage:\n"
        "  moba_collect collect --out FILE [--first-game N] [--games N] [--base-seed N]\n"
        "                       [--mix | --radiant net|script --dire net|script]\n"
        "                       [--max-ticks N] [--eps F] [--weights PATH]\n"
        "  moba_collect replay FILE [--dense OUT] [--games I:J]\n"
        "  moba_collect markov FILE [--games N] [--every K] [--horizon H]\n");
}

static int cmd_collect(int argc, char** argv) {
    uint32_t first_game = 0, games = 1, max_ticks = 20000;
    uint64_t base_seed = 0;
    int mix = 0, fixed_ctrl[2] = {CTRL_NET, CTRL_NET};
    float eps = 0.0f;
    const char* out_path = NULL;
    const char* weights_path = "third_party/pufferlib/resources/moba/moba_weights.bin";
    for (int i = 0; i < argc; i++) {
        if      (!strcmp(argv[i], "--first-game") && i + 1 < argc) first_game = strtoul(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--games") && i + 1 < argc) games = strtoul(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--base-seed") && i + 1 < argc) base_seed = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--max-ticks") && i + 1 < argc) max_ticks = strtoul(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--eps") && i + 1 < argc) eps = atof(argv[++i]);
        else if (!strcmp(argv[i], "--weights") && i + 1 < argc) weights_path = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--mix")) mix = 1;
        else if (!strcmp(argv[i], "--radiant") && i + 1 < argc) fixed_ctrl[0] = parse_ctrl(argv[++i]);
        else if (!strcmp(argv[i], "--dire") && i + 1 < argc) fixed_ctrl[1] = parse_ctrl(argv[++i]);
        else { usage(); return 1; }
    }
    if (!out_path || fixed_ctrl[0] < 0 || fixed_ctrl[1] < 0) { usage(); return 1; }
    // --mix: radiant/dire = net/net, net/script, script/net, by global index.
    static const int MIX[3][2] = {{CTRL_NET, CTRL_NET}, {CTRL_NET, CTRL_SCRIPT}, {CTRL_SCRIPT, CTRL_NET}};

    uint64_t whash = MOBA_FNV_INIT;
    FILE* wf = fopen(weights_path, "rb");
    if (!wf) { perror(weights_path); return 1; }
    uint8_t wb[65536]; size_t nread;
    while ((nread = fread(wb, 1, sizeof(wb), wf)) > 0) whash = moba_fnv1a(whash, wb, nread);
    fclose(wf);
    Weights* weights = load_weights(weights_path);
    PufferNet* net = make_puffernet(weights, MOBA_HEROES, MOBA_OBS, 64, 5, (int*)MOBA_ACTION_SIZES, MOBA_ATNS);

    MobaGame game;
    moba_game_alloc(&game);
    MOBA* env = &game.env;

    char tmp_path[4096];
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", out_path);
    FILE* out = fopen(tmp_path, "wb");
    if (!out) { perror(tmp_path); return 1; }
    MobaLogHeader hdr = {
        .magic = LOG_MAGIC, .version = FORMAT_VERSION, .n_games = 0, .max_ticks = max_ticks,
        .base_seed = base_seed, .first_game = first_game, .vision_range = env->vision_range,
        .agent_speed = env->agent_speed, .reward_death = env->reward_death, .reward_xp = env->reward_xp,
        .reward_distance = env->reward_distance, .reward_tower = env->reward_tower,
        .epsilon = eps, .weights_hash = whash,
    };
    fwrite(&hdr, sizeof(hdr), 1, out);

    uint16_t* codes = malloc(sizeof(uint16_t) * MOBA_HEROES * max_ticks);
    uLongf zcap = compressBound(sizeof(uint16_t) * MOBA_HEROES * max_ticks);
    uint8_t* zbuf = malloc(zcap);
    float obs_f[MOBA_HEROES * MOBA_OBS];
    float net_actions[MOBA_HEROES * MOBA_ATNS];
    MobaState* sbuf = malloc(sizeof(MobaState));
    ScriptMemory mem;
    long total_ticks = 0, total_z = 0;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (uint32_t g = 0; g < games; g++) {
        uint32_t gi = first_game + g;
        const int* ctrl = mix ? MIX[gi % 3] : fixed_ctrl;
        int any_net = ctrl[0] == CTRL_NET || ctrl[1] == CTRL_NET;
        uint64_t seed = game_seed_for(base_seed, gi);
        uint64_t prng = seed ^ 0xA5A5A5A5DEADBEEFULL;
        moba_game_start(&game, seed);
        net_reset(net);
        script_reset(&mem);

        uint64_t h = MOBA_FNV_INIT;
        uint32_t t = 0;
        int result = 0, truncated = 0;
        while (!result && !truncated) {
            if (any_net)
                net_act(net, env->observations, obs_f, net_actions, &prng);
            for (int a = 0; a < MOBA_HEROES; a++) {
                float* act = &env->actions[a * MOBA_ATNS];
                if (ctrl[a >= 5] == CTRL_NET)
                    memcpy(act, &net_actions[a * MOBA_ATNS], sizeof(float) * MOBA_ATNS);
                else
                    scripted_action(env, a, act, &mem);
                if (eps > 0 && rng_uniform(&prng) < eps)
                    for (int k = 0; k < MOBA_ATNS; k++)
                        act[k] = (float)(splitmix64(&prng) % MOBA_ACTION_SIZES[k]);
                codes[t * MOBA_HEROES + a] = moba_pack_action(act);
            }
            h = hash_state(h, &game, sbuf);
            result = moba_game_step(&game);
            t++;
            truncated = !result && t >= max_ticks;
        }
        h = hash_state(h, &game, sbuf);   // the terminal state

        uLongf zlen = zcap;
        if (compress2(zbuf, &zlen, (const Bytef*)codes, sizeof(uint16_t) * MOBA_HEROES * t, 6) != Z_OK) {
            fprintf(stderr, "zlib failed on game %u\n", gi);
            return 1;
        }
        MobaGameHeader gh = {
            .game_seed = seed, .game_index = gi, .n_ticks = t, .winner = (uint8_t)result,
            .truncated = (uint8_t)truncated, .radiant_ctrl = (uint8_t)ctrl[0], .dire_ctrl = (uint8_t)ctrl[1],
            .zbytes = (uint32_t)zlen, .state_hash = h,
        };
        fwrite(&gh, sizeof(gh), 1, out);
        fwrite(zbuf, 1, zlen, out);
        hdr.n_games++;
        total_ticks += t;
        total_z += zlen + sizeof(gh);
        printf("game %u  %s vs %s  %u ticks  %s\n", gi, CTRL_NAMES[ctrl[0]], CTRL_NAMES[ctrl[1]], t,
               result == 1 ? "radiant win" : result == 2 ? "dire win" : result == 3 ? "both ancients" : "truncated");
    }

    fseek(out, 0, SEEK_SET);
    fwrite(&hdr, sizeof(hdr), 1, out);
    fclose(out);
    if (rename(tmp_path, out_path) != 0) { perror("rename"); return 1; }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double secs = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
    printf("%s: %u games (%u..%u), %ld ticks, %.0f ticks/s, %.2f bytes/tick on disk\n",
        out_path, games, first_game, first_game + games - 1, total_ticks, total_ticks / secs,
        (double)total_z / total_ticks);

    free(codes); free(zbuf); free(sbuf);
    free_puffernet(net); free(weights);
    moba_game_free(&game);
    return 0;
}

// A log, opened for reading game by game.
typedef struct {
    FILE* f;
    MobaLogHeader hdr;
    uint16_t* codes;
    uint8_t* zbuf;
    uint32_t next;
} LogReader;

static int log_open(LogReader* r, const char* path) {
    memset(r, 0, sizeof(*r));
    r->f = fopen(path, "rb");
    if (!r->f) { perror(path); return 0; }
    if (fread(&r->hdr, sizeof(r->hdr), 1, r->f) != 1 || r->hdr.magic != LOG_MAGIC
            || r->hdr.version != FORMAT_VERSION) {
        fprintf(stderr, "%s: not a v%d MOBA game log\n", path, FORMAT_VERSION);
        return 0;
    }
    r->codes = malloc(sizeof(uint16_t) * MOBA_HEROES * r->hdr.max_ticks);
    return 1;
}

static int log_next(LogReader* r, MobaGameHeader* gh) {
    if (r->next >= r->hdr.n_games) return 0;
    if (fread(gh, sizeof(*gh), 1, r->f) != 1) { fprintf(stderr, "truncated log\n"); exit(1); }
    r->zbuf = realloc(r->zbuf, gh->zbytes);
    if (fread(r->zbuf, 1, gh->zbytes, r->f) != gh->zbytes) { fprintf(stderr, "truncated log\n"); exit(1); }
    uLongf rawlen = sizeof(uint16_t) * MOBA_HEROES * gh->n_ticks;
    if (uncompress((Bytef*)r->codes, &rawlen, r->zbuf, gh->zbytes) != Z_OK) {
        fprintf(stderr, "zlib failed on game %u\n", gh->game_index);
        exit(1);
    }
    r->next++;
    return 1;
}

static void log_close(LogReader* r) {
    fclose(r->f); free(r->codes); free(r->zbuf);
}

static void set_actions(MOBA* env, const uint16_t* codes_t) {
    for (int a = 0; a < MOBA_HEROES; a++)
        moba_unpack_action(codes_t[a], &env->actions[a * MOBA_ATNS]);
}

static int cmd_replay(int argc, char** argv) {
    if (argc < 1) { usage(); return 1; }
    const char* path = argv[0];
    const char* dense_path = NULL;
    uint32_t g_lo = 0, g_hi = UINT32_MAX;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--dense") && i + 1 < argc) dense_path = argv[++i];
        else if (!strcmp(argv[i], "--games") && i + 1 < argc) {
            if (sscanf(argv[++i], "%u:%u", &g_lo, &g_hi) != 2) { usage(); return 1; }
        }
        else { usage(); return 1; }
    }

    LogReader lr;
    if (!log_open(&lr, path)) return 1;
    MobaGame game;
    moba_game_alloc(&game);
    MOBA* env = &game.env;

    FILE* dense = NULL;
    MobaDenseHeader dh = {
        .magic = DENSE_MAGIC, .version = FORMAT_VERSION, .record_bytes = sizeof(MobaDenseRecord),
        .state_bytes = sizeof(MobaState), .vision_range = env->vision_range, .scan_cap = MOBA_SCAN_CAP,
        .state_fields = MOBA_STATE_FIELDS,
    };
    char dense_tmp[4096];
    if (dense_path) {
        snprintf(dense_tmp, sizeof(dense_tmp), "%s.tmp", dense_path);
        dense = fopen(dense_tmp, "wb");
        if (!dense) { perror(dense_tmp); return 1; }
        fwrite(&dh, sizeof(dh), 1, dense);
    }

    MobaDenseRecord* recs = dense ? malloc(sizeof(MobaDenseRecord) * ((size_t)lr.hdr.max_ticks + 1)) : NULL;
    MobaDenseRecord* one = calloc(1, sizeof(MobaDenseRecord));
    uint8_t* dz = NULL;
    uLongf dz_cap = 0;
    uint64_t raw_bytes = 0, z_bytes = 0;
    MobaStatic statics[MOBA_HEROES];
    uint8_t mine[MOBA_HEROES][MOBA_OBS];
    long ticks = 0, obs_mismatch_ticks = 0;
    int hash_fail = 0, replayed = 0, statics_written = 0;

    MobaGameHeader gh;
    while (log_next(&lr, &gh)) {
        uint32_t local = gh.game_index - lr.hdr.first_game;
        if (local < g_lo || local >= g_hi) continue;

        moba_game_start(&game, gh.game_seed);
        if (!statics_written) {
            moba_capture_static(env, statics);
            memcpy(dh.statics, statics, sizeof(statics));
            statics_written = 1;
        }

        uint64_t h = MOBA_FNV_INIT;
        int result = 0;
        for (uint32_t t = 0; t <= gh.n_ticks; t++) {
            MobaDenseRecord* rec = dense ? &recs[t] : one;
            int final = t == gh.n_ticks;
            if (!final) set_actions(env, &lr.codes[t * MOBA_HEROES]);

            moba_game_capture(&game, &rec->state);
            h = moba_fnv1a(h, &rec->state, sizeof(rec->state));
            moba_ego_obs(&rec->state, statics, env->vision_range, mine);
            if (memcmp(mine, env->observations, sizeof(mine)) != 0)
                obs_mismatch_ticks++;

            rec->game_index = gh.game_index;
            rec->tick = env->tick;
            rec->radiant_ctrl = gh.radiant_ctrl;
            rec->dire_ctrl = gh.dire_ctrl;
            rec->final = final;
            memcpy(rec->obs, env->observations, sizeof(rec->obs));
            memset(rec->actions, 0, sizeof(rec->actions));
            memset(rec->rewards, 0, sizeof(rec->rewards));
            if (!final) {
                for (int a = 0; a < MOBA_HEROES; a++)
                    for (int k = 0; k < MOBA_ATNS; k++)
                        rec->actions[a][k] = (uint8_t)env->actions[a * MOBA_ATNS + k];
                result = moba_game_step(&game);
                ticks++;
                for (int a = 0; a < MOBA_HEROES; a++) rec->rewards[a] = env->rewards[a];
                if (result && t + 1 != gh.n_ticks) {
                    fprintf(stderr, "game %u: ended at step %u, log says %u\n", gh.game_index, t + 1, gh.n_ticks);
                    hash_fail++;
                    break;
                }
            }
            int last_step = t + 1 == gh.n_ticks;
            rec->done = last_step || final;
            rec->truncated = (last_step || final) && gh.truncated;
            rec->winner = (last_step || final) ? gh.winner : 0;
            rec->pad[0] = rec->pad[1] = 0;
        }
        if (dense) {
            uLongf raw = sizeof(MobaDenseRecord) * ((uLongf)gh.n_ticks + 1);
            if (compressBound(raw) > dz_cap) { dz_cap = compressBound(raw); dz = realloc(dz, dz_cap); }
            uLongf zlen = dz_cap;
            if (compress2(dz, &zlen, (const Bytef*)recs, raw, 1) != Z_OK) {
                fprintf(stderr, "zlib failed on dense game %u\n", gh.game_index);
                return 1;
            }
            MobaDenseGame dg = {.game_index = gh.game_index, .n_records = gh.n_ticks + 1, .zbytes = zlen};
            fwrite(&dg, sizeof(dg), 1, dense);
            fwrite(dz, 1, zlen, dense);
            dh.n_records += dg.n_records;
            dh.n_games++;
            raw_bytes += raw;
            z_bytes += zlen;
        }
        if (h != gh.state_hash) {
            hash_fail++;
            fprintf(stderr, "game %u: state hash mismatch (replay diverged)\n", gh.game_index);
        }
        replayed++;
    }

    printf("%s: replayed %d games, %ld steps | state hash mismatches %d | "
           "records where moba_ego_obs != env obs: %ld\n",
        path, replayed, ticks, hash_fail, obs_mismatch_ticks);
    if (dense) {
        fseek(dense, 0, SEEK_SET);
        fwrite(&dh, sizeof(dh), 1, dense);
        fclose(dense);
        if (rename(dense_tmp, dense_path) != 0) { perror("rename"); return 1; }
        printf("wrote %s (%u games, %u records, %.1f KB each; zlib %.1f GB -> %.2f GB, %.1fx)\n",
            dense_path, dh.n_games, dh.n_records, sizeof(MobaDenseRecord) / 1024.0,
            raw_bytes / 1e9, z_bytes / 1e9, (double)raw_bytes / z_bytes);
    }
    log_close(&lr);
    free(recs); free(one); free(dz);
    moba_game_free(&game);
    return (hash_fail || obs_mismatch_ticks) ? 2 : 0;
}

// ---------------------------------------------------------------------------
// Markov / state-sufficiency test.
//
// The original game A runs on upstream c_step (auto-reset and all). At each
// checkpoint its MobaState S is captured and A's next H steps are recorded.
// Then, from S, a series of probes is built in a separate env B -- always a
// freshly started game with a different seed, so anything not restored from
// S is wrong -- and stepped with the same actions through moba_game_step:
//
//   full      restore everything. Must reproduce A exactly: state, all 10
//             observations and rewards, every step (sufficiency; also checks
//             moba_game_step + moba_game_end against upstream c_step).
//   omit P    restore everything except part P, which keeps the fresh game's
//             value. Should diverge (necessity). A part is counted as
//             detected at a checkpoint if the observation at the restore
//             tick or anything in the next H steps differs.
//   perturb   restore everything, then scramble something MobaState leaves
//             out. Must NOT diverge (the omission is safe).
// ---------------------------------------------------------------------------

typedef struct {
    MobaState state;
    uint8_t obs[MOBA_HEROES][MOBA_OBS];
    float rewards[MOBA_HEROES];
} Frame;

static void frame_of(const MobaGame* g, Frame* f) {
    moba_game_capture(g, &f->state);
    memcpy(f->obs, g->env.observations, sizeof(f->obs));
    memcpy(f->rewards, g->env.rewards, sizeof(f->rewards));
}

static int frame_eq(const Frame* a, const Frame* b) {
    return memcmp(&a->state, &b->state, sizeof(a->state)) == 0
        && memcmp(a->obs, b->obs, sizeof(a->obs)) == 0
        && memcmp(a->rewards, b->rewards, sizeof(a->rewards)) == 0;
}

enum { PERT_LAST_XY, PERT_ATTACK_AOE, PERT_ENTITY_REWARD, PERT_LOGS, PERT_STALE_SCAN,
       PERT_OUTPUT_BUFFERS, PERT_PATH_CACHE, N_PERT };
static const char* PERT_NAMES[N_PERT] = {
    "last_x/last_y", "attack_aoe", "entity.reward", "player_logs + log",
    "target-list entries past the terminator", "reward/terminal output buffers",
    "pathing cache (cleared)",
};

static void perturb(MobaGame* g, int kind, uint64_t* r) {
    MOBA* env = &g->env;
    for (int i = 0; i < NUM_ENTITIES; i++) {
        Entity* e = &env->entities[i];
        switch (kind) {
        case PERT_LAST_XY: e->last_x = splitmix64(r) % 128; e->last_y = splitmix64(r) % 128; break;
        case PERT_ATTACK_AOE: e->attack_aoe = splitmix64(r) % 8; break;
        case PERT_ENTITY_REWARD: e->reward = (float)(splitmix64(r) % 100); break;
        case PERT_STALE_SCAN: {
            int c = 0;
            while (c < 121 && env->scanned_targets[i][c] != NULL) c++;
            for (int k = c + 1; k < 121; k++)
                env->scanned_targets[i][k] = &env->entities[splitmix64(r) % NUM_ENTITIES];
            break;
        }
        default: break;
        }
    }
    if (kind == PERT_LOGS) {
        for (int i = 0; i < NUM_PLAYERS; i++) {
            float* pl = (float*)&env->player_logs[i];
            for (size_t k = 0; k < sizeof(PlayerLog) / sizeof(float); k++) pl[k] = (float)(splitmix64(r) % 50);
        }
        env->log.n += 7; env->log.score += 3;
    }
    if (kind == PERT_OUTPUT_BUFFERS)
        for (int i = 0; i < env->num_agents; i++) {
            env->rewards[i] = (float)(splitmix64(r) % 9);
            env->terminals[i] = 1;
        }
    if (kind == PERT_PATH_CACHE)
        memset(env->ai_paths, 255, (size_t)128 * 128 * 128 * 128);
}

// Step B like upstream: moba_game_step, then log + reset if the game ended.
static void probe_step(MobaGame* b) {
    int result = moba_game_step(b);
    if (result) moba_game_end(b, result);
}

typedef struct {
    MobaGame* b;
    const MobaState* s;
    const Frame* future;
    Frame* fb;
    const uint16_t* codes;
    uint32_t t;
    int h_end;
    uint64_t* prng;
} ProbeCtx;

enum { PROBE_SAME, PROBE_DIVERGED, PROBE_CRASHED };   // crashed = upstream exit, abort or hang

// One probe, in a forked child: an inconsistent state (e.g. a part omitted)
// can drive upstream into one of its exit(0)/abort paths or an endless retry
// loop, which must count as "the env noticed", not end or stall the test.
static int run_probe(const ProbeCtx* c, uint64_t omit, int pert_kind) {
    uint64_t pert_seed = splitmix64(c->prng);
    fflush(stdout);
    pid_t pid = fork();
    if (pid == 0) {
        freopen("/dev/null", "w", stdout);
        freopen("/dev/null", "w", stderr);
        alarm(5);   // an impossible state can hang upstream (spawn_player retries forever)
        uint64_t r = pert_seed;
        moba_game_restore(c->b, c->s, omit);
        if (pert_kind >= 0) perturb(c->b, pert_kind, &r);
        int diverged = memcmp(c->b->env.observations, c->future[0].obs, sizeof(c->future[0].obs)) != 0;
        for (int k = 0; k < c->h_end && !diverged; k++) {
            set_actions(&c->b->env, &c->codes[(c->t + k) * MOBA_HEROES]);
            probe_step(c->b);
            frame_of(c->b, c->fb);
            diverged = !frame_eq(c->fb, &c->future[k + 1]);
        }
        _exit(diverged ? 11 : 10);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    if (WIFEXITED(status) && WEXITSTATUS(status) == 10) return PROBE_SAME;
    if (WIFEXITED(status) && WEXITSTATUS(status) == 11) return PROBE_DIVERGED;
    return PROBE_CRASHED;
}

static int cmd_markov(int argc, char** argv) {
    if (argc < 1) { usage(); return 1; }
    const char* path = argv[0];
    uint32_t max_games = UINT32_MAX;
    int every = 250, horizon = 64;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--games") && i + 1 < argc) max_games = strtoul(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--every") && i + 1 < argc) every = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--horizon") && i + 1 < argc) horizon = atoi(argv[++i]);
        else { usage(); return 1; }
    }
    if (every <= horizon) { fprintf(stderr, "--every must exceed --horizon\n"); return 1; }

    LogReader lr;
    if (!log_open(&lr, path)) return 1;
    MobaGame A, B;
    moba_game_alloc(&A);
    moba_game_alloc(&B);
    Frame* future = malloc(sizeof(Frame) * (horizon + 1));   // A at restore tick + H steps
    Frame* fb = malloc(sizeof(Frame));
    MobaState* S = malloc(sizeof(MobaState));
    MobaState* S2 = malloc(sizeof(MobaState));
    uint64_t prng = 0x5EED;

    long checkpoints = 0, full_fail = 0, roundtrip_fail = 0, terminal_windows = 0;
    long omit_hit[MOBA_N_PARTS] = {0}, omit_crash[MOBA_N_PARTS] = {0}, pert_fail[N_PERT] = {0};
    uint32_t done_games = 0;

    MobaGameHeader gh;
    while (done_games < max_games && log_next(&lr, &gh)) {
        moba_game_start(&A, gh.game_seed);
        uint32_t t = 0;
        // Regular checkpoints, plus one whose window contains the terminal
        // step, so the end-of-game path is checked against upstream too.
        uint32_t term_cp = gh.n_ticks > (uint32_t)horizon ? gh.n_ticks - horizon / 2 : 0;
        int term_done = 0;
        while (t < gh.n_ticks) {
            int is_cp = t % every == 0 || (!term_done && t >= term_cp);
            if (!is_cp || t + 1 >= gh.n_ticks) {
                set_actions(&A.env, &lr.codes[t * MOBA_HEROES]);
                moba_game_c_step(&A);
                t++;
                continue;
            }
            // Checkpoint at tick t: capture S, record A's next H steps.
            moba_game_capture(&A, S);
            frame_of(&A, &future[0]);
            int h_end = horizon;
            if (t + h_end > gh.n_ticks) h_end = gh.n_ticks - t;
            for (int k = 0; k < h_end; k++) {
                set_actions(&A.env, &lr.codes[(t + k) * MOBA_HEROES]);
                moba_game_c_step(&A);
                frame_of(&A, &future[k + 1]);
            }
            if (t + h_end == gh.n_ticks) { terminal_windows++; term_done = 1; }
            checkpoints++;

            ProbeCtx ctx = {&B, S, future, fb, lr.codes, t, h_end, &prng};

            // Round trip: restore then capture reproduces S exactly.
            moba_game_restore(&B, S, 0);
            moba_game_capture(&B, S2);
            if (memcmp(S, S2, sizeof(*S)) != 0) roundtrip_fail++;

            full_fail += run_probe(&ctx, 0, -1) != PROBE_SAME;
            for (int p = 0; p < MOBA_N_PARTS; p++) {
                int r = run_probe(&ctx, 1ULL << p, -1);
                omit_hit[p] += r != PROBE_SAME;
                omit_crash[p] += r == PROBE_CRASHED;
            }
            for (int k = 0; k < N_PERT; k++)
                pert_fail[k] += run_probe(&ctx, 0, k) != PROBE_SAME;

            t += h_end;
        }
        done_games++;
        fprintf(stderr, "  game %u (%u ticks): %ld checkpoints so far, full-restore failures %ld\n",
                gh.game_index, gh.n_ticks, checkpoints, full_fail);
    }

    printf("markov test: %u games, %ld checkpoints, horizon %d (%ld windows reach the terminal step)\n",
        done_games, checkpoints, horizon, terminal_windows);
    printf("\nSUFFICIENCY (must be 0)\n");
    printf("  round trip restore -> capture differs:          %ld / %ld\n", roundtrip_fail, checkpoints);
    printf("  full restore diverges from the original game:   %ld / %ld\n", full_fail, checkpoints);
    printf("\nNECESSITY (leave one part at a fresh game's value; diverged within the horizon)\n");
    for (int p = 0; p < MOBA_N_PARTS; p++)
        printf("  %-20s %4ld / %ld%s\n", MOBA_FIELD_NAMES[p], omit_hit[p], checkpoints,
               omit_crash[p] ? "  (some via upstream exit/hang on an impossible state)" : "");
    printf("\nSAFE OMISSIONS (not in MobaState; scrambled after restore; must be 0)\n");
    for (int k = 0; k < N_PERT; k++)
        printf("  %-42s %ld / %ld\n", PERT_NAMES[k], pert_fail[k], checkpoints);

    log_close(&lr);
    free(future); free(fb); free(S); free(S2);
    moba_game_free(&A);
    moba_game_free(&B);
    int fail = roundtrip_fail || full_fail;
    for (int k = 0; k < N_PERT; k++) fail |= pert_fail[k] != 0;
    return fail ? 2 : 0;
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 1; }
    if (!strcmp(argv[1], "collect")) return cmd_collect(argc - 2, argv + 2);
    if (!strcmp(argv[1], "replay")) return cmd_replay(argc - 2, argv + 2);
    if (!strcmp(argv[1], "markov")) return cmd_markov(argc - 2, argv + 2);
    usage();
    return 1;
}
