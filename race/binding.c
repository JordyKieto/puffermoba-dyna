#include "race.h"

#define OBS_SIZE RACE_OBS_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {3}
#define OBS_TYPE FLOAT
#define ACT_TYPE DOUBLE

#define MY_VEC_INIT
#define Env Race
#include "vecenv.h"

// Upstream's my_init read llw_ang / flw_ang / frw_ang / rrw_ang / render / i
// from kwargs. None of those keys exist in config/whisker_racer.ini or in the
// empty [env] block of config/default.ini, so dict_get() printed
// "dict_get failed to find key: ..." and tripped its assert -- i.e.
// `puffer train whisker_racer` did not start. The whisker angles are derived
// from w_ang inside init() anyway, so they were never meaningful inputs.
static void race_config(Env* env, Dict* kwargs) {
    env->frameskip = dict_get(kwargs, "frameskip")->value;
    env->width = dict_get(kwargs, "width")->value;
    env->height = dict_get(kwargs, "height")->value;
    env->max_whisker_length = dict_get(kwargs, "max_whisker_length")->value;
    env->turn_pi_frac = dict_get(kwargs, "turn_pi_frac")->value;
    env->maxv = dict_get(kwargs, "maxv")->value;
    env->continuous = dict_get(kwargs, "continuous")->value;
    env->reward_yellow = dict_get(kwargs, "reward_yellow")->value;
    env->reward_green = dict_get(kwargs, "reward_green")->value;
    env->gamma = dict_get(kwargs, "gamma")->value;
    env->track_width = dict_get(kwargs, "track_width")->value;
    env->num_radial_sectors = dict_get(kwargs, "num_radial_sectors")->value;
    env->num_points = dict_get(kwargs, "num_points")->value;
    env->bezier_resolution = dict_get(kwargs, "bezier_resolution")->value;
    env->w_ang = dict_get(kwargs, "w_ang")->value;
    env->corner_thresh = dict_get(kwargs, "corner_thresh")->value;
    env->ftmp1 = dict_get(kwargs, "ftmp1")->value;
    env->ftmp2 = dict_get(kwargs, "ftmp2")->value;
    env->ftmp3 = dict_get(kwargs, "ftmp3")->value;
    env->ftmp4 = dict_get(kwargs, "ftmp4")->value;
    env->mode7 = dict_get(kwargs, "mode7")->value;
    env->render_many = dict_get(kwargs, "render_many")->value;
    env->method = dict_get(kwargs, "method")->value;
    env->seed = (unsigned int)dict_get(kwargs, "seed")->value;
    env->max_episode_steps = (int)dict_get(kwargs, "max_episode_steps")->value;
    env->num_agents = 1;
}

void my_init(Env* env, Dict* kwargs) {
    race_config(env, kwargs);
    env->env_idx = (int)dict_get(kwargs, "env_idx")->value;
    init(env);
}

// Upstream relied on the generic my_vec_init in vecenv.h, which does
// `srand(num_envs); envs[num_envs].rng = num_envs;` -- then whisker_racer's
// my_init overwrote env->rng with the config constant, so every env generated
// the *same* track, and episode starts all drew from one global stream.
// Here each env gets its index explicitly and seeds two private PCG streams.
Env* my_vec_init(int* num_envs_out, int* buffer_env_starts, int* buffer_env_counts,
                 Dict* vec_kwargs, Dict* env_kwargs) {
    int total_agents = (int)dict_get(vec_kwargs, "total_agents")->value;
    int num_buffers = (int)dict_get(vec_kwargs, "num_buffers")->value;
    int agents_per_buffer = total_agents / num_buffers;

    Env* envs = (Env*)calloc(total_agents, sizeof(Env));
    for (int i = 0; i < total_agents; i++) {
        Env* env = &envs[i];
        race_config(env, env_kwargs);
        env->env_idx = i;   // seeds (rng_track, rng_episode); see init()
        init(env);
    }

    for (int b = 0; b < num_buffers; b++) {
        buffer_env_starts[b] = b * agents_per_buffer;
        buffer_env_counts[b] = agents_per_buffer;
    }

    *num_envs_out = total_agents;
    return envs;
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "laps", log->laps);
    dict_set(out, "sector_crossings", log->sector_crossings);
    dict_set(out, "crash_rate", log->crash_rate);
    dict_set(out, "wall_crash_rate", log->wall_crash_rate);
    dict_set(out, "offtrack_rate", log->offtrack_rate);
    dict_set(out, "truncated_rate", log->truncated_rate);
    dict_set(out, "n", log->n);
}
