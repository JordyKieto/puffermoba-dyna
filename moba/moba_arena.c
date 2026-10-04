// Arena: play real MOBA games with one team driven from Python (ctypes) and the
// other by one of the collector's controllers -- the shipped PufferNet weights
// ("net") or the handwritten creep_ai-style bots ("script").
//
// The collector's code is reused as-is: moba_collect.c is included with its
// main() renamed, so net_act / scripted_action / the weights loader are the
// exact ones that generated the corpus. Nothing in moba_collect.c changes.
//
// One game per process (moba_game.h keeps the random stream in a process
// global); mobawm/rl/arena.py runs games in parallel processes.
//
//   ma_init(weights)            load the net once
//   ma_start(seed)              new game, controller memories reset
//   ma_obs(out)                 10 x 510 uint8: the game's own ego observations
//   ma_ctrl_actions(ctrl, out)  10 x 6 float actions from net (0) or script (1)
//   ma_step(actions)            10 x 6 float -> 0 running, 1 radiant win, 2 dire win, 3 both
//   ma_rewards(out)             10 float: this tick's per-hero reward (upstream's)
//   ma_logs(out)                10 x 19 float: upstream PlayerLog per hero (kills, deaths, ...)
//   ma_tick()                   current tick

#define main moba_collect_main
#include "moba_collect.c"
#undef main

#define MA_API __attribute__((visibility("default")))

static MobaGame ma_game;
static PufferNet* ma_net = NULL;
static Weights* ma_weights = NULL;
static ScriptMemory ma_mem;
static uint64_t ma_prng;
static float ma_obs_f[MOBA_HEROES * MOBA_OBS];
static float ma_net_act[MOBA_HEROES * MOBA_ATNS];
static int ma_net_tick = -1;       // the net is advanced once per tick (it is recurrent)

MA_API int ma_init(const char* weights_path) {
    ma_weights = load_weights(weights_path);
    if (!ma_weights) return -1;
    ma_net = make_puffernet(ma_weights, MOBA_HEROES, MOBA_OBS, 64, 5, (int*)MOBA_ACTION_SIZES, MOBA_ATNS);
    moba_game_alloc(&ma_game);
    return 0;
}

MA_API void ma_start(uint64_t seed) {
    moba_game_start(&ma_game, seed);
    net_reset(ma_net);
    script_reset(&ma_mem);
    ma_prng = seed ^ 0xA5A5A5A5DEADBEEFULL;
    ma_net_tick = -1;
}

MA_API void ma_obs(uint8_t* out) {
    memcpy(out, ma_game.env.observations, MOBA_HEROES * MOBA_OBS);
}

MA_API void ma_ctrl_actions(int ctrl, float* out) {
    MOBA* env = &ma_game.env;
    if (ctrl == CTRL_NET) {
        if (ma_net_tick != env->tick) {      // one recurrent step per tick, all 10 heroes
            net_act(ma_net, env->observations, ma_obs_f, ma_net_act, &ma_prng);
            ma_net_tick = env->tick;
        }
        memcpy(out, ma_net_act, sizeof(ma_net_act));
    } else {
        for (int a = 0; a < MOBA_HEROES; a++)
            scripted_action(env, a, &out[a * MOBA_ATNS], &ma_mem);
    }
}

MA_API int ma_step(const float* actions) {
    memcpy(ma_game.env.actions, actions, sizeof(float) * MOBA_HEROES * MOBA_ATNS);
    return moba_game_step(&ma_game);
}

MA_API void ma_rewards(float* out) {
    memcpy(out, ma_game.env.rewards, sizeof(float) * MOBA_HEROES);
}

MA_API void ma_logs(float* out) {
    for (int i = 0; i < MOBA_HEROES; i++)
        memcpy(&out[i * 19], &ma_game.env.player_logs[i], sizeof(float) * 19);
}

MA_API int ma_tick(void) { return ma_game.env.tick; }

// ---------------------------------------------------------------------------------------
// Net pool: the shipped PufferNet acting for MANY games at once (dream PPO vs net).
// One network with batch = n_rows (10 heroes x n games); its MinGRU state is per row, so
// every game keeps its own recurrent memory. Sampling is net_act's, on the pool's own
// SplitMix64 stream. The weights are shared read-only (load_weights once, idx rewound).

static PufferNet* ma_pool = NULL;
static int ma_pool_rows = 0;
static float* ma_pool_obs_f = NULL;
static uint64_t ma_pool_rng = 0x9E3779B97F4A7C15ULL;

MA_API int ma_pool_init(const char* weights_path, int n_rows, uint64_t seed) {
    Weights* w = load_weights(weights_path);
    if (!w) return -1;
    w->idx = 0;
    ma_pool = make_puffernet(w, n_rows, MOBA_OBS, 64, 5, (int*)MOBA_ACTION_SIZES, MOBA_ATNS);
    ma_pool_rows = n_rows;
    ma_pool_obs_f = calloc((size_t)n_rows * MOBA_OBS, sizeof(float));
    ma_pool_rng = seed ^ 0xD1B54A32D192ED03ULL;
    return 0;
}

// Zero the recurrent state of rows [lo, hi) (a game's 10 heroes start a new episode).
MA_API void ma_pool_reset_rows(int lo, int hi) {
    MinGRU* g = ma_pool->mingru;
    for (int l = 0; l < g->num_layers; l++)
        memset(&g->state[((size_t)l * g->batch_size + lo) * g->hidden_size], 0,
               sizeof(float) * (size_t)(hi - lo) * g->hidden_size);
}

// obs: n_rows x 510 uint8 -> actions: n_rows x 6 float (sampled, as net_act).
MA_API void ma_pool_act(const uint8_t* obs, float* actions) {
    int n = ma_pool_rows;
    for (size_t i = 0; i < (size_t)n * MOBA_OBS; i++) ma_pool_obs_f[i] = (float)obs[i];
    linear(ma_pool->encoder, ma_pool_obs_f);
    mingru(ma_pool->mingru, ma_pool->encoder->output);
    linear(ma_pool->decoder, ma_pool->mingru->output);
    const float* logits = ma_pool->decoder->output;
    int atn_sum = 0;
    for (int h = 0; h < MOBA_ATNS; h++) atn_sum += MOBA_ACTION_SIZES[h];
    for (int b = 0; b < n; b++) {
        int adr = b * (atn_sum + 1);   // +1: fused value head
        for (int h = 0; h < MOBA_ATNS; h++) {
            int k = MOBA_ACTION_SIZES[h];
            float sum = 0;
            for (int i = 0; i < k; i++) sum += expf(logits[adr + i]);
            float u = rng_uniform(&ma_pool_rng), acc = 0, a = 0;
            for (int i = 0; i < k; i++) {
                acc += expf(logits[adr + i]) / sum;
                if (u < acc) { a = (float)i; break; }
            }
            actions[b * MOBA_ATNS + h] = a;
            adr += k;
        }
    }
}

// Raw decoder outputs (logits + value), for checking the pool against the collector's net.
MA_API void ma_net_logits(float* out, int n) { memcpy(out, ma_net->decoder->output, sizeof(float) * n); }
MA_API void ma_pool_logits(float* out, int n) { memcpy(out, ma_pool->decoder->output, sizeof(float) * n); }

// ---------------------------------------------------------------------------------------
// Multi-game arena (the world-model planner): n real games in ONE process. Safe because
// moba_game_step swaps each game's random stream into the process global around the call,
// and games are stepped one after another. Each game has its own scripted-bot memory.

static MobaGame* mg = NULL;
static ScriptMemory* mg_mem = NULL;
static int mg_n = 0;
static uint64_t* mg_hash = NULL;     // the collector's FNV-1a state hash per game (for .mbl logs)
static MobaState* mg_sbuf = NULL;

MA_API int ma_multi_init(int n) {
    mg = calloc(n, sizeof(MobaGame));
    mg_mem = calloc(n, sizeof(ScriptMemory));
    for (int i = 0; i < n; i++) moba_game_alloc(&mg[i]);
    mg_n = n;
    mg_hash = calloc(n, sizeof(uint64_t));
    mg_sbuf = calloc(1, sizeof(MobaState));
    return 0;
}

MA_API void ma_multi_start(int g, uint64_t seed) {
    moba_game_start(&mg[g], seed);
    script_reset(&mg_mem[g]);
    mg_hash[g] = MOBA_FNV_INIT;
}

MA_API void ma_multi_obs(int g, uint8_t* out) {
    memcpy(out, mg[g].env.observations, MOBA_HEROES * MOBA_OBS);
}

// The full entity state (206 x 26, moba_state.h field order) + tick + creep_idx: the world model's input.
MA_API void ma_multi_state(int g, float* ent, int32_t* tick, int32_t* cidx) {
    moba_capture_entities(&mg[g].env, (float(*)[MOBA_STATE_FIELDS])ent);
    *tick = mg[g].env.tick;
    *cidx = mg[g].env.creep_idx;
}

MA_API void ma_multi_script(int g, float* out) {
    for (int a = 0; a < MOBA_HEROES; a++)
        scripted_action(&mg[g].env, a, &out[a * MOBA_ATNS], &mg_mem[g]);
}

MA_API int ma_multi_step(int g, const float* actions) {
    memcpy(mg[g].env.actions, actions, sizeof(float) * MOBA_HEROES * MOBA_ATNS);
    mg_hash[g] = hash_state(mg_hash[g], &mg[g], mg_sbuf);   // as the collector: the state before every step
    return moba_game_step(&mg[g]);
}

// The game's state hash as the collector writes it into the log header: every pre-step
// state, then the terminal state. Call once, after the game's last step.
MA_API uint64_t ma_multi_final_hash(int g) { return hash_state(mg_hash[g], &mg[g], mg_sbuf); }

MA_API void ma_multi_rewards(int g, float* out) { memcpy(out, mg[g].env.rewards, sizeof(float) * MOBA_HEROES); }

MA_API void ma_multi_logs(int g, float* out) {
    for (int i = 0; i < MOBA_HEROES; i++)
        memcpy(&out[i * 19], &mg[g].env.player_logs[i], sizeof(float) * 19);
}

// Several independent net pools (id = return value): e.g. the real opponents, and the
// opponents imagined inside the planner's branches.
#define MA_MAX_POOLS 8
static PufferNet* pools[MA_MAX_POOLS];
static float* pools_obs[MA_MAX_POOLS];
static int pools_rows[MA_MAX_POOLS];
static uint64_t pools_rng[MA_MAX_POOLS];
static int n_pools = 0;

MA_API int ma_pools_new(const char* weights_path, int n_rows, uint64_t seed) {
    if (n_pools >= MA_MAX_POOLS) return -1;
    Weights* w = load_weights(weights_path);
    if (!w) return -1;
    w->idx = 0;
    int id = n_pools++;
    pools[id] = make_puffernet(w, n_rows, MOBA_OBS, 64, 5, (int*)MOBA_ACTION_SIZES, MOBA_ATNS);
    pools_obs[id] = calloc((size_t)n_rows * MOBA_OBS, sizeof(float));
    pools_rows[id] = n_rows;
    pools_rng[id] = seed ^ 0xD1B54A32D192ED03ULL;
    return id;
}

MA_API void ma_pools_reset_rows(int id, int lo, int hi) {
    MinGRU* g = pools[id]->mingru;
    for (int l = 0; l < g->num_layers; l++)
        memset(&g->state[((size_t)l * g->batch_size + lo) * g->hidden_size], 0,
               sizeof(float) * (size_t)(hi - lo) * g->hidden_size);
}

// Copy recurrent state rows [src, src+n) of pool a into rows [dst, dst+n) of pool b (branching).
MA_API void ma_pools_copy_rows(int a, int src, int b, int dst, int n) {
    MinGRU* ga = pools[a]->mingru;
    MinGRU* gb = pools[b]->mingru;
    for (int l = 0; l < ga->num_layers; l++)
        memcpy(&gb->state[((size_t)l * gb->batch_size + dst) * gb->hidden_size],
               &ga->state[((size_t)l * ga->batch_size + src) * ga->hidden_size],
               sizeof(float) * (size_t)n * ga->hidden_size);
}

MA_API void ma_pools_act(int id, const uint8_t* obs, float* actions) {
    PufferNet* net = pools[id];
    int n = pools_rows[id];
    for (size_t i = 0; i < (size_t)n * MOBA_OBS; i++) pools_obs[id][i] = (float)obs[i];
    linear(net->encoder, pools_obs[id]);
    mingru(net->mingru, net->encoder->output);
    linear(net->decoder, net->mingru->output);
    const float* logits = net->decoder->output;
    int atn_sum = 0;
    for (int h = 0; h < MOBA_ATNS; h++) atn_sum += MOBA_ACTION_SIZES[h];
    for (int b = 0; b < n; b++) {
        int adr = b * (atn_sum + 1);
        for (int h = 0; h < MOBA_ATNS; h++) {
            int k = MOBA_ACTION_SIZES[h];
            float sum = 0;
            for (int i = 0; i < k; i++) sum += expf(logits[adr + i]);
            float u = rng_uniform(&pools_rng[id]), acc = 0, a = 0;
            for (int i = 0; i < k; i++) {
                acc += expf(logits[adr + i]) / sum;
                if (u < acc) { a = (float)i; break; }
            }
            actions[b * MOBA_ATNS + h] = a;
            adr += k;
        }
    }
}
