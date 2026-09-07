#pragma once
// Race - PufferLib Ocean "Race", forked from ocean/whisker_racer.
//
// Upstream is non-deterministic (process-global rand()), has a broken reset
// contract, no episode timer, and a 3-float observation that is far from
// Markovian. See docs/AUDIT.md for the full finding list and docs/CONTRACT.md
// for the state contract this file implements.

#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <limits.h>
#include <string.h>
#include "raylib.h"
#include <time.h>

#define LEFT 0
#define NOOP 1
#define RIGHT 2

#define PI2 (PI * 2)

#define MAX_CONTROL_POINTS 32
#define MAX_BEZIER_RESOLUTION 16

// Upstream declared sectors_completed[16] and hardcoded PI2/16 in
// update_radial_progress, so the configurable num_radial_sectors was dead
// (config shipped 180; the sweep over it did nothing). We keep a compile-time
// max so OBS_SIZE stays constant and honor num_radial_sectors at runtime.
#define RACE_MAX_SECTORS 16

// Observation layout version. racewm/contract.py asserts on a hash of this.
#define RACE_STATE_VERSION 1
#define RACE_OBS_SIZE (14 + RACE_MAX_SECTORS)

#define RACE_CRASH_NONE 0
#define RACE_CRASH_WALL 1
#define RACE_CRASH_OFFTRACK 2

// Whisker fraction below which the car counts as having hit a wall.
#define RACE_CRASH_EPS 0.05f

// PCG32. Upstream used the process-global libc srand()/rand(), so track
// generation and episode starts were unreproducible under the threaded vec
// backend and interfered across envs. Each env owns two independent streams.
typedef struct { uint64_t state; uint64_t inc; } RaceRng;

static inline uint32_t rng_u32(RaceRng* r) {
    uint64_t old = r->state;
    r->state = old * 6364136223846793005ULL + r->inc;
    uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    uint32_t rot = (uint32_t)(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31u));
}

static inline void rng_seed(RaceRng* r, uint64_t seed, uint64_t seq) {
    r->state = 0u;
    r->inc = (seq << 1u) | 1u;
    rng_u32(r);
    r->state += seed;
    rng_u32(r);
}

// Bounded draw. Matches upstream `rand() % n` semantics (incl. modulo bias).
static inline int rng_below(RaceRng* r, int n) {
    return (int)(rng_u32(r) % (uint32_t)n);
}

// Drop-in for upstream's rand() inside track generation, which always has a
// local `env`. Track geometry draws from rng_track only.
#define RACE_RAND(e) ((int)(rng_u32(&(e)->rng_track) & 0x7fffffff))

typedef struct {
    Vector2 position;
} ControlPoint;

typedef struct {
    ControlPoint controls[MAX_CONTROL_POINTS];
    int num_points;
    Vector2 centerline[MAX_CONTROL_POINTS * MAX_BEZIER_RESOLUTION];
    Vector2 inner_edge[MAX_CONTROL_POINTS * MAX_BEZIER_RESOLUTION];
    Vector2 outer_edge[MAX_CONTROL_POINTS * MAX_BEZIER_RESOLUTION];
    int total_points;
    Vector2 curbs[MAX_CONTROL_POINTS][4];
    int curb_count;
} Track;

typedef struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    // Behavioral metrics. Upstream logged none of these, and `perf` was
    // score/max_score with max_score never assigned -> inf/nan.
    float laps;
    float sector_crossings;
    float crash_rate;
    float wall_crash_rate;
    float offtrack_rate;
    float truncated_rate;
    float n;
} Log;

typedef struct Client {
    float width;   // 640
    float height;  // 480
    //float llw_ang; // left left whisker angle
    float flw_ang; // front left whisker angle
    float frw_ang; // front right whisker angle
    //float rrw_ang; // right right whisker angle
    float max_whisker_length;
    float turn_pi_frac; //  (pi / turn_pi_frac is the turn angle)
    float maxv;    // 5
    int render;
    int debug;
} Client;

typedef struct Race {
    Client* client;
    Log log;
    float* observations;
    double* actions;
    float* rewards;
    float* terminals;
    int num_agents;

    int debug;
    int render_many;

    float corner_thresh;
    float ftmp1;
    float ftmp2;
    float ftmp3;
    float ftmp4;
    int method;

    float reward_yellow;
    float reward_green;
    float gamma;

    // Game
    int width;
    int height;
    float score;
    int tick;
    int frameskip;
    int render;
    int continuous;
    int current_sector;
    int sectors_completed[RACE_MAX_SECTORS];
    int total_sectors_crossed;
    int track_width;
    int num_radial_sectors;
    int num_points;
    int bezier_resolution;
    Track track;

    // Car
    float px;
    float py;
    float ang;
    float vx;
    float vy;
    float v;
    int near_point_idx;
    float maxv;
    float turn_pi_frac;

    // Whiskers
    int num_whiskers;
    Vector2 whisker_dirs[2];
    float w_ang;
    float llw_ang; // left left whisker angle
    float flw_ang; // front left whisker angle
    float frw_ang; // front right whisker angle
    float rrw_ang; // right right whisker angle
    float llw_length;
    float flw_length;
    float ffw_length;
    float frw_length;
    float rrw_length;
    float max_whisker_length;

    // Math
    float inv_width;
    float inv_height;
    float inv_maxv;
    float inv_pi2;
    float inv_bezier_res;

    Texture2D puffer;

    int texture_initialized;
    int mode7;

    // --- Race additions; see docs/CONTRACT.md ---
    RaceRng rng_track;      // track geometry, drawn once in init()
    RaceRng rng_episode;    // episode start pose, advanced per reset
    unsigned int seed;
    int env_idx;
    int episode_id;

    int max_episode_steps;  // truncation; upstream had no time limit at all
    int crashed;
    int crash_cause;
    int terminated;
    int truncated;

    int sector_crossings;   // every forward crossing (reward-aligned)
    int lap_count;
} Race;

void update_nearest_point(Race* env);
void compute_whiskers(Race* env);
void update_radial_progress(Race* env);
void compute_observations(Race* env);
void GenerateRandomTrack(Race* env);
void reset_episode(Race* env);

void c_close(Race* env) {
    //unload_track();
}

void free_allocated(Race* env) {
    free(env->actions);
    free(env->observations);
    free(env->terminals);
    free(env->rewards);
    c_close(env);
}

void add_log(Race* env) {
    env->log.episode_length += (float)env->tick;
    env->log.episode_return += env->score;
    env->log.score += env->score;
    // perf = fractional laps completed. Upstream divided by an unassigned
    // max_score (always 0), so the headline metric was inf/nan.
    env->log.perf += (float)env->sector_crossings / (float)env->num_radial_sectors;
    env->log.laps += (float)env->lap_count;
    env->log.sector_crossings += (float)env->sector_crossings;
    env->log.crash_rate += (env->crash_cause != RACE_CRASH_NONE) ? 1.0f : 0.0f;
    env->log.wall_crash_rate += (env->crash_cause == RACE_CRASH_WALL) ? 1.0f : 0.0f;
    env->log.offtrack_rate += (env->crash_cause == RACE_CRASH_OFFTRACK) ? 1.0f : 0.0f;
    env->log.truncated_rate += (float)env->truncated;
    env->log.n += 1;
}

// The observation IS the Markov state, normalized. Upstream shipped 3 floats
// (two whiskers + score/100), omitting heading, position, the nearest-segment
// index that the whisker raycast depends on, the sector bookkeeping that
// determines all reward, and any notion of time.
void compute_observations(Race* env) {
    float* obs = env->observations;
    const int S = env->num_radial_sectors;

    obs[0] = env->flw_length;
    obs[1] = env->frw_length;
    // Heading as sin/cos: ang wraps at 2pi, so regressing it directly breaks.
    obs[2] = sinf(env->ang);
    obs[3] = cosf(env->ang);
    obs[4] = env->px * env->inv_width;
    obs[5] = env->py * env->inv_height;
    obs[6] = env->v * env->inv_maxv;

    // current_sector and near_point_idx are both cyclic indices.
    float sector_ang = PI2 * (float)env->current_sector / (float)S;
    obs[7] = sinf(sector_ang);
    obs[8] = cosf(sector_ang);

    float near_ang = PI2 * (float)env->near_point_idx / (float)env->track.total_points;
    obs[9] = sinf(near_ang);
    obs[10] = cosf(near_ang);

    obs[11] = (float)env->total_sectors_crossed / (float)S;
    obs[12] = (float)env->tick / (float)env->max_episode_steps;
    obs[13] = env->score / 100.0f;

    for (int i = 0; i < RACE_MAX_SECTORS; i++) {
        obs[14 + i] = (i < S) ? (float)env->sectors_completed[i] : 0.0f;
    }
}

Client* make_client(Race* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = env->width;
    client->height = env->height;
    //client->llw_ang = env->llw_ang;
    client->flw_ang = env->flw_ang;
    client->frw_ang = env->frw_ang;
    //client->rrw_ang = env->rrw_ang;
    client->max_whisker_length = env->max_whisker_length;
    client->turn_pi_frac = env->turn_pi_frac;
    client->maxv = env->maxv;

    InitWindow(env->width, env->height, "PufferLib Whisker Racer");
    if (env->render_many) SetTargetFPS(10 / env->frameskip);
    else SetTargetFPS(60 / env->frameskip);
    env->puffer = LoadTexture("resources/shared/puffers_128.png");

    return client;
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}

void get_random_start(Race* env) {
    int start_idx = rng_below(&env->rng_episode, env->track.total_points);
    env->near_point_idx = start_idx;

    env->px = env->track.centerline[start_idx].x;
    env->py = env->track.centerline[start_idx].y;

    int next_idx = (start_idx + 1) % env->track.total_points;
    float dx = env->track.centerline[next_idx].x - env->px;
    float dy = env->track.centerline[next_idx].y - env->py;
    env->ang = atan2f(dy, dx);
    if (env->ang < 0) env->ang += PI2;

    env->whisker_dirs[0] = (Vector2){cosf(env->ang + env->flw_ang), sinf(env->ang + env->flw_ang)};
    env->whisker_dirs[1] = (Vector2){cosf(env->ang + env->frw_ang), sinf(env->ang + env->frw_ang)};

    env->v = env->maxv;
    env->vx = env->v * cosf(env->ang);
    env->vy = env->v * sinf(env->ang);
}

void reset_radial_progress(Race* env) {
    float center_x = env->width * 0.5f;
    float center_y = env->height * 0.5f;
    const int S = env->num_radial_sectors;

    float angle = atan2f(env->py - center_y, env->px - center_x);
    if (angle < 0) angle += PI2;

    env->current_sector = (int)(angle / (PI2 / (float)S)) % S;

    for (int i = 0; i < RACE_MAX_SECTORS; i++) env->sectors_completed[i] = 0;
    env->total_sectors_crossed = 0;
    env->sector_crossings = 0;
    env->lap_count = 0;
}

void reset_episode(Race* env) {
    env->score = 0.0f;
    env->tick = 0;
    env->crashed = 0;
    env->crash_cause = RACE_CRASH_NONE;
    env->episode_id += 1;

    get_random_start(env);
    reset_radial_progress(env);

    // Upstream hardcoded the post-reset whiskers to 0.5. Raycast instead, so
    // the first observation of an episode is a true sensor reading.
    update_nearest_point(env);
    compute_whiskers(env);
}

void c_reset(Race* env) {
    // Upstream called compute_observations() BEFORE zeroing score and before
    // reset_round(), so every episode opened on stale data from the previous
    // episode. Observe last.
    env->episode_id = 0;
    reset_episode(env);
    compute_observations(env);
}

// Line segment intersection helper function
// Returns 1 if intersection found, 0 otherwise
// If intersection found, stores the parameter t in *t_out (0 <= t <= 1 along the whisker ray)
static inline int line_segment_intersect(Vector2 ray_start, Vector2 ray_dir, float ray_length,
                                       Vector2 seg_start, Vector2 seg_end, float* t_out) {
    Vector2 seg_dir = {seg_end.x - seg_start.x, seg_end.y - seg_start.y};
    Vector2 diff = {seg_start.x - ray_start.x, seg_start.y - ray_start.y};

    float cross_rd_sd = ray_dir.x * seg_dir.y - ray_dir.y * seg_dir.x;

    // Lines are parallel
    if (fabsf(cross_rd_sd) < 1e-3f) {
        return 0;
    }

    float cross_diff_sd = diff.x * seg_dir.y - diff.y * seg_dir.x;
    float cross_diff_rd = diff.x * ray_dir.y - diff.y * ray_dir.x;

    float t = cross_diff_sd / cross_rd_sd;
    float u = cross_diff_rd / cross_rd_sd;

    // Check if intersection is within both line segments
    if (t >= 0.0f && t <= ray_length && u >= 0.0f && u <= 1.0f) {
        *t_out = t;
        return 1;
    }

    return 0;
}

void update_nearest_point(Race* env) {
    float min_dist_sq = 100000;
    int closest_seg = env->near_point_idx;
    Vector2 car_pos = {env->px, env->py};

    int search_range = 3;
    for (int offset = 0; offset <= search_range; offset++) {
        int i = (env->near_point_idx + offset + env->track.total_points) % env->track.total_points;

        Vector2 center = env->track.centerline[i];
        float dx = car_pos.x - center.x;
        float dy = car_pos.y - center.y;
        float dist_sq = dx * dx + dy * dy;

        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            closest_seg = i;
        }
    }

    env->near_point_idx = closest_seg;
}

// Pure sensor read. Upstream folded crash detection AND an in-place c_reset()
// into this loop, which could reset twice in one call (the w=1 iteration ran
// against a already-reset car) and double-count episodes in the log. Callers
// now run update_nearest_point() first; c_step owns episode boundaries.
void compute_whiskers(Race* env) {
    float max_len = env->max_whisker_length;
    float inv_max_len = 1.0f / max_len;

    float* lengths[2] = {&env->flw_length, &env->frw_length};
    Vector2 car_pos = {env->px, env->py};

    for (int w = 0; w < 2; ++w) {
        Vector2 whisker_dir = env->whisker_dirs[w];
        float min_hit_distance = max_len;

        int window_size = 10;
        for (int offset = -window_size/2; offset <= window_size/2; offset++) {
            int i = (env->near_point_idx + offset + env->track.total_points) % env->track.total_points;
            int next_i = (i + 1) % env->track.total_points;

            float t;

            if (line_segment_intersect(car_pos, whisker_dir, max_len,
                                     env->track.inner_edge[i], env->track.inner_edge[next_i], &t)) {
                if (t < min_hit_distance) {
                    min_hit_distance = t;
                }
                if (t < 0.05) break;
            }

            if (line_segment_intersect(car_pos, whisker_dir, max_len,
                                     env->track.outer_edge[i], env->track.outer_edge[next_i], &t)) {
                if (t < min_hit_distance) {
                    min_hit_distance = t;
                }
                if (t < 0.05) break;
            }
        }

        *lengths[w] = fminf(1.0f, fmaxf(0.0f, min_hit_distance * inv_max_len));
    }
}

void update_radial_progress(Race* env) {
    float center_x = env->width * 0.5f;
    float center_y = env->height * 0.5f;
    const int S = env->num_radial_sectors;

    float angle = atan2f(env->py - center_y, env->px - center_x);
    if (angle < 0) angle += PI2;

    // Upstream hardcoded PI2/16.0f here and then took `% num_radial_sectors`,
    // which was a no-op for any S >= 16. Sector count is now honored.
    int sector = (int)(angle / (PI2 / (float)S)) % S;

    if (sector != env->current_sector) {
        int expected_next = (env->current_sector + 1) % S;
        if (sector == expected_next) {
            // Reward is reward_yellow on ANY forward crossing, first time or
            // not - unchanged from upstream. sectors_completed only gates
            // total_sectors_crossed, so lap telemetry is tracked separately.
            if (!env->sectors_completed[sector]) {
                env->sectors_completed[sector] = 1;
                env->total_sectors_crossed++;
            }
            env->sector_crossings++;
            if (env->sector_crossings % S == 0) env->lap_count++;
            env->rewards[0] += env->reward_yellow;
            env->score += env->reward_yellow;
        }
        env->current_sector = sector;
    }
}

Vector2 EvaluateCubicBezier(Vector2 p0, Vector2 p1, Vector2 p2, Vector2 p3, float t) {
    float u = 1.0f - t;
    float tt = t * t;
    float uu = u * u;
    float uuu = uu * u;
    float ttt = tt * t;

    Vector2 result;
    result.x = uuu * p0.x + 3 * uu * t * p1.x + 3 * u * tt * p2.x + ttt * p3.x;
    result.y = uuu * p0.y + 3 * uu * t * p1.y + 3 * u * tt * p2.y + ttt * p3.y;
    return result;
}

Vector2 GetBezierDerivative(Vector2 p0, Vector2 p1, Vector2 p2, Vector2 p3, float t) {
    float u = 1.0f - t;
    float tt = t * t;
    float uu = u * u;

    Vector2 result;
    result.x = -3 * uu * p0.x + 3 * uu * p1.x - 6 * u * t * p1.x + 6 * u * t * p2.x - 3 * tt * p2.x + 3 * tt * p3.x;
    result.y = -3 * uu * p0.y + 3 * uu * p1.y - 6 * u * t * p1.y + 6 * u * t * p2.y - 3 * tt * p2.y + 3 * tt * p3.y;
    return result;
}

Vector2 NormalizeVector(Vector2 v) {
    float length = sqrtf(v.x * v.x + v.y * v.y);
    if (length < 0.00001f) return (Vector2){0, 0};
    return (Vector2){v.x / length, v.y / length};
}

Vector2 GetPerpendicular(Vector2 v) {
    return (Vector2){-v.y, v.x};
}

void GenerateRandomControlPoints(Race* env) {
    float center_x = env->width * 0.5f;
    float center_y = env->height * 0.5f;

    int n = env->num_points;

    if (env->method == -1) {
        env->method = RACE_RAND(env) % 3;
    }

    if (env->method == 0) {
        // Randomly choose distinct, non-adjacent indices for tight and medium corners
        int opt1 = RACE_RAND(env) % n;
        int opt2;
        do {
            opt2 = RACE_RAND(env) % n;
        } while (opt2 == opt1 || abs(opt2 - opt1) == 1 || abs(opt2 - opt1) == n - 1);

        int opt3, opt4;
        do {
            opt3 = RACE_RAND(env) % n;
        } while (opt3 == opt1 || opt3 == opt2);

        do {
            opt4 = RACE_RAND(env) % n;
        } while (opt4 == opt1 || opt4 == opt2 || opt4 == opt3 || abs(opt4 - opt3) == 1 || abs(opt4 - opt3) == n - 1);

        // Generate control points
        for (int i = 0; i < n; i++) {
            float angle = (PI2 * i) / n;

            float dist_from_center;
            if (i == opt1) {
                dist_from_center = env->height * 0.2 + (RACE_RAND(env) % 30);
            } else if (i == opt2 || i == opt3) {
                dist_from_center = env->height * 0.3 + (RACE_RAND(env) % 40);
            } else {
                dist_from_center = env->height * 0.5 + (RACE_RAND(env) % 30);
            }

            env->track.controls[i].position.x = center_x + dist_from_center * cosf(angle);
            env->track.controls[i].position.y = center_y + dist_from_center * 0.8f * sinf(angle);
        }
    } // end method 0
    else if (env->method == 1) {

        int corner_types[n];
        int assigned[n];

        // type 0 is close to center, 1 is medium, 2 is far from center
        for (int i = 0; i < n; i++) {
            corner_types[i] = 2;
            assigned[i] = 0;
        }

        int num_med = (n + 2) / 5;
        for (int placed = 0; placed < num_med; placed++) {
            int attempts = 0;
            int pos;
            do {
                pos = RACE_RAND(env) % n;
                bool valid = (assigned[pos] == 0);
                if (valid) break;
                attempts++;
            } while (attempts < 50);

            if (attempts < 50) {
                corner_types[pos] = 1;
                assigned[pos] = 1;
            }
        }

        int num_close = (n + 2) / 8;
        for (int placed = 0; placed < num_close; placed++) {
            int attempts = 0;
            int pos;
            do {
                pos = RACE_RAND(env) % n;
                int prev_prev = (pos - 2 + n) % n;
                int prev = (pos - 1 + n) % n;
                int next = (pos + 1) % n;

                bool valid = (corner_types[pos] == 2) &&
                            (corner_types[prev_prev] > 0) &&
                            (corner_types[prev] > 0) &&
                            (corner_types[next] > 0) &&
                            //(corner_types[next_next] > 0) &&
                            (assigned[pos] == 0);

                if (valid) break;
                attempts++;
            } while (attempts < 50);

            if (attempts < 50) {
                corner_types[pos] = 0;
                assigned[pos] = 1;
            }
        }

        for (int i = 0; i < n; i++ ) {
            int prev = (i - 1 + n) % n;
            int next = (i + 1 + n) % n;
            if (corner_types[prev] == 0 && corner_types[i] == 2 && corner_types[next] == 0) {
                corner_types[i] = 1;
            }
        }

        // Generate control points
        for (int i = 0; i < n; i++) {
            float angle = (PI2 * i) / n;

            float dist_from_center;
            if (corner_types[i] == 0) {
                dist_from_center = env->height * 0.35 + (RACE_RAND(env) % 30);
            } else if (corner_types[i] == 1) {
                dist_from_center = env->height * 0.45 + (RACE_RAND(env) % 40);
            } else {
                dist_from_center = env->height * 0.6 + (RACE_RAND(env) % 30);
            }

            env->track.controls[i].position.x = center_x + dist_from_center * 1.2f * cosf(angle);
            env->track.controls[i].position.y = center_y + dist_from_center * 0.7f * sinf(angle);
        }

    } // end method 1
    else {
        float base_radius = env->height * 0.5f;
        float variation_strength = 0.5;
        float track_stretch_x = 1.0;
        float track_stretch_y = 0.6;

        float freq1 = 2.0f + (RACE_RAND(env) % 5);
        float amp1 = (1.0f / freq1) * (0.9f + 0.2f * (RACE_RAND(env) % 100) / 100.0f);
        float phase1 = PI2 * (RACE_RAND(env) % 100) / 100.0f;

        float freq2 = 1.0f + (RACE_RAND(env) % 2);
        float amp2 = 0.2f + 0.2f * (RACE_RAND(env) % 100) / 100.0f;
        float phase2 = PI2 * (RACE_RAND(env) % 100) / 100.0f;

        float freq3 = 10.0f + 0.5f * (RACE_RAND(env) % 3);
        float amp3 = 0.3f + 0.1f * (RACE_RAND(env) % 100) / 100.0f;
        float phase3 = PI2 * (RACE_RAND(env) % 100) / 100.0f;

        for (int i = 0; i < n; i++) {
            float angle = (PI2 * i) / n;

            float radius_variation = amp1 * cosf(freq1 * angle + phase1) +
                                amp2 * cosf(freq2 * angle + phase2) +
                                amp3 * cosf(freq3 * angle + phase3);

            float radius = base_radius + (base_radius * variation_strength * radius_variation);

            env->track.controls[i].position.x = center_x + radius * track_stretch_x * cosf(angle);
            env->track.controls[i].position.y = center_y + radius * track_stretch_y * sinf(angle);
        }
    } // end method 2

    float tw2 = env->track_width * 0.5f;

    for (int i = 0; i < n; i++) {
        if (env->track.controls[i].position.x < tw2) env->track.controls[i].position.x = tw2;
        if (env->track.controls[i].position.x > env->width - tw2) env->track.controls[i].position.x = env->width - tw2;
        if (env->track.controls[i].position.y < tw2) env->track.controls[i].position.y = tw2;
        if (env->track.controls[i].position.y > env->height - tw2) env->track.controls[i].position.y = env->height - tw2;

        Vector2 prev = env->track.controls[(i - 1 + n) % n].position;
        Vector2 curr = env->track.controls[i].position;
        Vector2 next = env->track.controls[(i + 1) % n].position;


        float vx1 = prev.x - curr.x;
        float vy1 = prev.y - curr.y;
        float vx2 = next.x - curr.x;
        float vy2 = next.y - curr.y;

        float dot = vx1 * vx2 + vy1 * vy2;
        float mag1 = sqrtf(vx1 * vx1 + vy1 * vy1);
        float mag2 = sqrtf(vx2 * vx2 + vy2 * vy2);

        if (mag1 < 1e-3f || mag2 < 1e-3f) continue;

        float angle_cos = dot / (mag1 * mag2);

        if (angle_cos > env->corner_thresh) {
            float dx = curr.x - center_x;
            float dy = curr.y - center_y;
            float dist = sqrtf(dx * dx + dy * dy);

            float adjust_scale = 0.0f;
            if (dist > 200) {
                adjust_scale = 0.3f * angle_cos;
            }

            env->track.controls[i].position.x = env->track.controls[i].position.x - dx * adjust_scale;
            env->track.controls[i].position.y = env->track.controls[i].position.y - dy * adjust_scale;
        }
    }
}

void GenerateTrackCenterline(Race* env) {
    int point_index = 0;

    for (int i = 0; i < env->num_points; i++) {
        Vector2 p0 = env->track.controls[i].position;
        Vector2 p3 = env->track.controls[(i + 1) % env->num_points].position;

        Vector2 prev = env->track.controls[(i - 1 + env->num_points) % env->num_points].position;
        Vector2 next = env->track.controls[(i + 2) % env->num_points].position;

        Vector2 dir1 = NormalizeVector((Vector2){p3.x - prev.x, p3.y - prev.y});
        Vector2 dir2 = NormalizeVector((Vector2){next.x - p0.x, next.y - p0.y});

        float dist = sqrtf((p3.x - p0.x) * (p3.x - p0.x) + (p3.y - p0.y) * (p3.y - p0.y));

        float control_length;
        if (i == 1 || i == 3) {
            control_length = dist * 0.2f;
        } else if (i == 0 || i == 4) {
            control_length = dist * 0.3f;
        } else {
            control_length = dist * 0.4f;
        }

        Vector2 p1 = (Vector2){p0.x + dir1.x * control_length, p0.y + dir1.y * control_length};
        Vector2 p2 = (Vector2){p3.x - dir2.x * control_length, p3.y - dir2.y * control_length};

        for (int j = 0; j < env->bezier_resolution && point_index < MAX_CONTROL_POINTS * env->bezier_resolution - 1; j++) {
            float t = (float)j * env->inv_bezier_res;
            env->track.centerline[point_index] = EvaluateCubicBezier(p0, p1, p2, p3, t);
            point_index++;
        }
    }
    env->track.total_points = point_index;
}

void GenerateTrackEdges(Race* env) {
    for (int i = 0; i < env->track.total_points; i++) {
        Vector2 current = env->track.centerline[i];
        Vector2 next = env->track.centerline[(i + 1) % env->track.total_points];

        Vector2 tangent = NormalizeVector((Vector2){next.x - current.x, next.y - current.y});
        Vector2 normal = GetPerpendicular(tangent);

        // Create inner and outer edges
        float half_width = env->track_width * 0.5f;
        env->track.inner_edge[i] = (Vector2){current.x - normal.x * half_width, current.y - normal.y * half_width};
        env->track.outer_edge[i] = (Vector2){current.x + normal.x * half_width, current.y + normal.y * half_width};
    }
}

void GenerateCurbs(Race* env) {
    env->track.curb_count = 0;

    for (int i = 0; i < env->num_points; i++) {
        Vector2 prev = env->track.controls[(i - 1 + env->num_points) % env->num_points].position;
        Vector2 curr = env->track.controls[i].position;
        Vector2 next = env->track.controls[(i + 1) % env->num_points].position;

        float vx1 = prev.x - curr.x;
        float vy1 = prev.y - curr.y;
        float vx2 = next.x - curr.x;
        float vy2 = next.y - curr.y;

        Vector2 to_prev = {vx1, vy1};
        Vector2 to_next = {vx2, vy2};

        float cross = to_prev.x * to_next.y - to_prev.y * to_next.x;
        float dot = vx1 * vx2 + vy1 * vy2;

        float mag1 = sqrtf(vx1 * vx1 + vy1 * vy1);
        float mag2 = sqrtf(vx2 * vx2 + vy2 * vy2);

        if (mag1 < 1e-3f || mag2 < 1e-3f) continue;

        float angle_cos = dot / (mag1 * mag2);

        if (angle_cos > -0.8f) {
            int apex_idx = i * env->bezier_resolution;

            Vector2* edge_points = (cross > 0) ? env->track.inner_edge : env->track.outer_edge;

            for (int j = 0; j < 4; j++) {
                int idx = (apex_idx - 1 + j + env->track.total_points) % env->track.total_points; // -2? todo
                env->track.curbs[env->track.curb_count][j] = edge_points[idx];
            }
            env->track.curb_count++;
        }
    }
}

void GenerateRandomTrack(Race* env) {
    GenerateRandomControlPoints(env);
    GenerateTrackCenterline(env);
    GenerateTrackEdges(env);
    GenerateCurbs(env);
}

void TopDownTexture(Race* env, RenderTexture2D* mode7RenderTexture, Vector2* center_points) {
    if (env->texture_initialized == 0) {
        *mode7RenderTexture = LoadRenderTexture(env->width, env->height);

        BeginTextureMode(*mode7RenderTexture);
        ClearBackground(DARKGREEN);
        SetConfigFlags(FLAG_MSAA_4X_HINT);
        ClearBackground(DARKGREEN);
        DrawSplineBasis(center_points, env->track.total_points + 3, env->track_width, BLACK);
        //DrawSplineBasis(center_points, env->track.total_points + 3, 2, WHITE);

        for (int i = 0; i < env->track.curb_count; i++) {
            Vector2 curb_points[4];
            for (int j = 0; j < 4; j++) {
                curb_points[j] = env->track.curbs[i][j];
                curb_points[j].y = env->height - curb_points[j].y; // Flip Y coordinate
            }
            DrawSplineBasis(curb_points, 4, 5.0f, RED); // 5 pixel wide red curbs
        }

        EndTextureMode();
        env->texture_initialized = 1;
    }
}

void Mode7(Race* env, RenderTexture2D mode7RenderTexture) {
    BeginDrawing();
    ClearBackground(SKYBLUE);

    Texture2D scene = mode7RenderTexture.texture;
    float camX = env->px;
    float camY = env->py;
    float camAngle = env->ang;

    Image sceneImage = LoadImageFromTexture(scene);
    Color* pixels = LoadImageColors(sceneImage);

    int height = env->height;
    int width = env->width;
    float inv_width = env->inv_width;
    float inv_height = env->inv_height;

    int horizon = height / 2;

    float cos_ang = cosf(camAngle);
    float sin_ang = sinf(camAngle);

    float cos_ang2 = -2.0f * cos_ang;
    float sin_ang2 = 2.0f * sin_ang;

    float height9 = 9.0f * height;

    DrawRectangle(0, horizon, width, height-horizon, DARKGREEN);

    for (int screenY = horizon; screenY < height; screenY += 3) {
        float row = (float)(screenY - horizon);

        if (row < 0.0001) row = 0.0001;
        float z = height9 / row;

        float dx = -sin_ang * z;
        float dy =  cos_ang * z;

        float sx = camX + dy + dx;
        float sy = camY - dx + dy;

        dx = (sin_ang2 * z) * inv_width * 3.0f;
        dy = (cos_ang2 * z) * inv_width * 3.0f;

        for (int screenX = 0; screenX < width; screenX += 3) {
            int srcX = (int)sx;
            int srcY = (int)sy;

            if (srcX >= 0 && srcX < width &&
                srcY >= 0 && srcY < height) {
                Color color = pixels[srcY * width + srcX];
                DrawRectangle(screenX, screenY, 3, 3, color);
            }

            sx += dx;
            sy += dy;
        }
    }
    UnloadImageColors(pixels);
    UnloadImage(sceneImage);

    int minimap_width = width * 0.333f;
    int minimap_height = height * 0.333f;
    int minimap_x = width - minimap_width;
    int minimap_y = 0;

    DrawTexturePro(
        scene,
        (Rectangle){0, 0, width, -height},
        (Rectangle){minimap_x, minimap_y, minimap_width, minimap_height},
        (Vector2){0, 0},
        0,
        WHITE
    );

    float minimap_puffer_size = 8.0f;
    float minimap_px = minimap_x + (env->px * inv_width) * minimap_width;
    float minimap_py = minimap_y + ((height - env->py) * inv_height) * minimap_height;

    DrawTexturePro(
        env->puffer,
        (Rectangle){0, 0, 128, 128},
        (Rectangle){minimap_px, minimap_py, minimap_puffer_size, minimap_puffer_size},
        (Vector2){minimap_puffer_size / 2.0f, minimap_puffer_size / 2.0f},
        (-env->ang * 180.0f / PI) - 10,
        WHITE
    );

    EndDrawing();
}

void Draw(Race* env, Vector2* center_points) {
    BeginDrawing();
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    ClearBackground(DARKGREEN);
    DrawSplineBasis(center_points, env->track.total_points + 3, env->track_width, BLACK);
    //DrawSplineBasis(center_points, env->track.total_points + 3, 2, WHITE);
    for (int i = 0; i < env->track.curb_count; i++) {
        Vector2 curb_points[4];
        for (int j = 0; j < 4; j++) {
            curb_points[j] = env->track.curbs[i][j];
            curb_points[j].y = env->height - curb_points[j].y; // Flip Y coordinate
        }
        DrawSplineBasis(curb_points, 4, 5.0f, RED); // 5 pixel wide red curbs
    }

    float puffer_width = 48.0f;
    float puffer_height = 48.0f;
    float puffer_x = env->px;
    float puffer_y = env->height - env->py;
    Vector2 origin = {puffer_width / 2.0f, puffer_height / 2.0f};

    DrawTexturePro(
        env->puffer,
        (Rectangle){0, 0, 128, 128},
        (Rectangle){puffer_x, puffer_y, puffer_width, puffer_height},
        origin,
        (-env->ang * 180.0f / PI) - 10,
        (Color){255, 255, 255, 255}
    );

    EndDrawing();
}

void c_render(Race* env) {

    static RenderTexture2D mode7RenderTexture;

    env->render = 1;
    if (env->client == NULL) {
        env->client = make_client(env);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    if (IsKeyPressed(KEY_TAB)) {
        ToggleFullscreen();
    }

    if (IsKeyDown(KEY_M)) {
        if (env->mode7 == 1) {
  	    env->mode7 = 0;
        }
        else {
   	    env->mode7 = 1;
        }
    }

    if (env->render_many)
    {
        env->method = RACE_RAND(env) % 3;
        GenerateRandomTrack(env);
    }

    Vector2* center_points = malloc(sizeof(Vector2) * (env->track.total_points + 3));
    for (int i = 0; i < env->track.total_points; i++) {
        center_points[i] = env->track.centerline[i];
        center_points[i].y = env->height - center_points[i].y;
    }

    // Without enough overlap it draws a C rather than an O
    center_points[env->track.total_points] = center_points[0];
    center_points[env->track.total_points + 1] = center_points[1];
    center_points[env->track.total_points + 2] = center_points[2];

    if (env->mode7 == 1) {
        TopDownTexture(env, &mode7RenderTexture, center_points);
        Mode7(env, mode7RenderTexture);
    }
    else {
        Draw(env, center_points);
    }

    free(center_points);
}

void init(Race* env) {
    env->tick = 0;
    env->debug = 0;

    if (env->num_radial_sectors < 1 || env->num_radial_sectors > RACE_MAX_SECTORS) {
        env->num_radial_sectors = RACE_MAX_SECTORS;
    }
    if (env->max_episode_steps <= 0) env->max_episode_steps = 2000;
    if (env->frameskip < 1) env->frameskip = 1;

    env->inv_width = 1.0f / env->width;
    env->inv_height = 1.0f / env->height;
    env->inv_maxv = 1.0f / env->maxv;
    env->inv_pi2 = 1.0f / PI2;
    env->inv_bezier_res = 1.0f / env->bezier_resolution;

    env->flw_ang = -env->w_ang;
    env->frw_ang = env->w_ang;

    env->texture_initialized = 0;

    // Upstream: srand(env->rng + env->i) on the process-global stream, with
    // env->rng overwritten from config so every env got the same track.
    rng_seed(&env->rng_track,   (uint64_t)env->seed, 2ull * (uint64_t)env->env_idx);
    rng_seed(&env->rng_episode, (uint64_t)env->seed, 2ull * (uint64_t)env->env_idx + 1ull);

    env->episode_id = 0;
    env->crashed = 0;
    env->crash_cause = RACE_CRASH_NONE;
    env->terminated = 0;
    env->truncated = 0;
    env->sector_crossings = 0;
    env->lap_count = 0;

    GenerateRandomTrack(env);
}

void allocate(Race* env) {
    init(env);
    env->observations = (float*)calloc(RACE_OBS_SIZE, sizeof(float));
    env->actions = (double*)calloc(1, sizeof(double));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (float*)calloc(1, sizeof(float));
}

void step_frame(Race* env, float action) {
    if (action == LEFT) {
        env->ang += PI / env->turn_pi_frac;
    } else if (action == RIGHT) {
        env->ang -= PI / env->turn_pi_frac;
    }
    if (env->ang > PI2) {
        env->ang -= PI2;
    } else if (env->ang < 0) {
        env->ang += PI2;
    }

    env->whisker_dirs[0] = (Vector2){cosf(env->ang + env->flw_ang), sinf(env->ang + env->flw_ang)};
    env->whisker_dirs[1] = (Vector2){cosf(env->ang + env->frw_ang), sinf(env->ang + env->frw_ang)};

    env->vx = env->v * cosf(env->ang);
    env->vy = env->v * sinf(env->ang);
    env->px = env->px + env->vx;
    env->py = env->py + env->vy;
    if (env->px < 0) env->px = 0;
    else if (env->px > env->width) env->px = env->width;
    if (env->py < 0) env->py = 0;
    else if (env->py > env->height) env->py = env->height;

    update_nearest_point(env);
    compute_whiskers(env);

    if (env->flw_length < RACE_CRASH_EPS || env->frw_length < RACE_CRASH_EPS) {
        env->flw_length = 0.0f;
        env->frw_length = 0.0f;
        env->crashed = 1;
        env->crash_cause = RACE_CRASH_WALL;
        return;
    }
    if (env->flw_length >= 0.99f && env->frw_length >= 0.99f) {
        // Both whiskers see nothing: car has left the track entirely.
        env->flw_length = 0.0f;
        env->frw_length = 0.0f;
        env->crashed = 1;
        env->crash_cause = RACE_CRASH_OFFTRACK;
        return;
    }

    update_radial_progress(env);
}

void c_step(Race* env) {
    env->terminals[0] = 0;
    env->rewards[0] = 0.0f;
    env->terminated = 0;
    env->truncated = 0;

    float action = (float)env->actions[0];
    for (int i = 0; i < env->frameskip; i++) {
        env->tick += 1;
        step_frame(env, action);
        // Upstream kept sub-stepping a freshly-reset episode after a crash.
        if (env->crashed) break;
    }

    if (env->crashed) {
        env->terminated = 1;
    } else if (env->tick >= env->max_episode_steps) {
        env->truncated = 1;
    }

    if (env->terminated || env->truncated) {
        // PufferLib 4.0 has no truncation channel - both map onto terminals.
        // The distinction is preserved in the log and in env->truncated.
        env->terminals[0] = 1;
        add_log(env);
        reset_episode(env);
    }

    compute_observations(env);
}
