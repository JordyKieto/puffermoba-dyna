// Batched C API over the real Race env, for driving it from Python.
//
// Exists because a planner cannot be exported to a .rpo file -- it needs the
// world model at inference time -- so it cannot be measured by race_eval the
// way BC and Dreamer policies are. Without this, planners could only ever be
// scored inside their own world model, which is precisely the comparison that
// hides a planner that looks good in imagination and fails in the env.
//
// Batched on purpose: stepping one env per Python call would put an
// interpreter round trip in front of a ~1us step. One call advances N envs.

#include <stdlib.h>
#include <string.h>
#include "race.h"

typedef struct {
    Race* envs;
    int n;
    float* obs;        // (n, RACE_OBS_SIZE), contiguous for a numpy view
    float* rewards;
    int* terminated;
    int* truncated;
    int* episode_id;
    int* env_idx;
} RacePool;

RacePool* race_pool_create(int n, unsigned int seed, int track_start,
                           int n_tracks, int frameskip, int max_episode_steps,
                           int width, int height, int track_width,
                           int num_radial_sectors, int num_points,
                           int bezier_resolution, float turn_pi_frac,
                           float maxv, float w_ang, float max_whisker_length,
                           float reward_yellow, float corner_thresh, int method) {
    RacePool* p = (RacePool*)calloc(1, sizeof(RacePool));
    p->n = n;
    p->envs = (Race*)calloc(n, sizeof(Race));
    p->obs = (float*)calloc((size_t)n * RACE_OBS_SIZE, sizeof(float));
    p->rewards = (float*)calloc(n, sizeof(float));
    p->terminated = (int*)calloc(n, sizeof(int));
    p->truncated = (int*)calloc(n, sizeof(int));
    p->episode_id = (int*)calloc(n, sizeof(int));
    p->env_idx = (int*)calloc(n, sizeof(int));

    for (int i = 0; i < n; i++) {
        Race* e = &p->envs[i];
        memset(e, 0, sizeof(Race));
        e->width = width; e->height = height; e->track_width = track_width;
        e->num_radial_sectors = num_radial_sectors;
        e->num_points = num_points; e->bezier_resolution = bezier_resolution;
        e->frameskip = frameskip; e->method = method;
        e->turn_pi_frac = turn_pi_frac; e->maxv = maxv; e->w_ang = w_ang;
        e->max_whisker_length = max_whisker_length;
        e->reward_yellow = reward_yellow; e->corner_thresh = corner_thresh;
        e->max_episode_steps = max_episode_steps;
        e->render = 0; e->render_many = 0; e->mode7 = 0;
        e->seed = seed;
        // Cycle the requested track slice so N envs can exceed N tracks.
        e->env_idx = track_start + (i % n_tracks);
        p->env_idx[i] = e->env_idx;

        // Observations point straight into the shared buffer, so no copy is
        // needed after a step.
        e->observations = p->obs + (size_t)i * RACE_OBS_SIZE;
        e->actions = (double*)calloc(1, sizeof(double));
        e->rewards = p->rewards + i;
        e->terminals = (float*)calloc(1, sizeof(float));
        e->num_agents = 1;
        init(e);
    }
    return p;
}

void race_pool_reset(RacePool* p) {
    for (int i = 0; i < p->n; i++) {
        c_reset(&p->envs[i]);
        p->terminated[i] = 0;
        p->truncated[i] = 0;
        p->episode_id[i] = p->envs[i].episode_id;
    }
}

void race_pool_step(RacePool* p, const int* actions) {
    for (int i = 0; i < p->n; i++) {
        Race* e = &p->envs[i];
        e->actions[0] = (double)actions[i];
        c_step(e);
        p->terminated[i] = e->terminated;
        p->truncated[i] = e->truncated;
        p->episode_id[i] = e->episode_id;
    }
}

float* race_pool_obs(RacePool* p)        { return p->obs; }
float* race_pool_rewards(RacePool* p)    { return p->rewards; }
int*   race_pool_terminated(RacePool* p) { return p->terminated; }
int*   race_pool_truncated(RacePool* p)  { return p->truncated; }
int*   race_pool_episode_id(RacePool* p) { return p->episode_id; }
int*   race_pool_env_idx(RacePool* p)    { return p->env_idx; }
int    race_pool_obs_dim(void)           { return RACE_OBS_SIZE; }
int    race_pool_num_actions(void)       { return 3; }

// Track polylines, so a planner can build its geometry features without
// re-deriving them from a separate dump.
int race_pool_total_points(RacePool* p) { return p->envs[0].track.total_points; }

void race_pool_track(RacePool* p, int i, float* centerline, float* inner, float* outer) {
    Race* e = &p->envs[i];
    for (int k = 0; k < e->track.total_points; k++) {
        centerline[2*k]   = e->track.centerline[k].x;
        centerline[2*k+1] = e->track.centerline[k].y;
        inner[2*k]        = e->track.inner_edge[k].x;
        inner[2*k+1]      = e->track.inner_edge[k].y;
        outer[2*k]        = e->track.outer_edge[k].x;
        outer[2*k+1]      = e->track.outer_edge[k].y;
    }
}

void race_pool_free(RacePool* p) {
    for (int i = 0; i < p->n; i++) {
        free(p->envs[i].actions);
        free(p->envs[i].terminals);
    }
    free(p->envs); free(p->obs); free(p->rewards);
    free(p->terminated); free(p->truncated); free(p->episode_id); free(p->env_idx);
    free(p);
}
