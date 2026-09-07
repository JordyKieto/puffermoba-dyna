// Scripted baseline policies for Race.
//
// These drive the eval CLI and the BC corpus. They are deterministic given
// their own RNG stream, which is seeded independently of the env's two streams
// so that changing the policy never perturbs track geometry or start poses.

#ifndef RACE_BOTS_H
#define RACE_BOTS_H

#include "race.h"

enum {
    BOT_WHISKER = 0,   // greedy wall-avoider on the two whisker readings
    BOT_STRAIGHT,      // always NOOP; the trivial floor
    BOT_RANDOM,        // uniform over {LEFT, NOOP, RIGHT}
    BOT_CENTERLINE,    // privileged: steer onto the local track tangent
    BOT_KIND_COUNT
};

static const char* BOT_NAMES[BOT_KIND_COUNT] = {
    "whisker", "straight", "random", "centerline"
};

typedef struct {
    int kind;
    float epsilon;    // per-step chance of a uniform random action
    float deadband;   // whisker difference below which the bot goes straight
    float lookahead;  // pure-pursuit lookahead, in pixels
    int track_idx;    // persistent progress index; -1 = needs re-acquire
    RaceRng rng;
} Bot;

static inline void bot_init(Bot* bot, int kind, float epsilon, uint64_t seed, uint64_t seq) {
    bot->kind = kind;
    bot->epsilon = epsilon;
    bot->deadband = 0.02f;
    bot->lookahead = 60.0f;
    bot->track_idx = -1;
    rng_seed(&bot->rng, seed, seq);
}

static inline float rng_float(RaceRng* r) {
    return (float)(rng_u32(r) >> 8) / (float)(1u << 24);
}

// Steer away from whichever whisker reads closer.
//
// Sign convention, which the upstream field names actively work against:
// flw_ang = -w_ang and frw_ang = +w_ang, and the car integrates as
// px += v*cos(ang), py += v*sin(ang) -- a standard math frame where +ang is
// counter-clockwise (the renderer flips y for display, the physics does not).
// So whisker_dirs[0] ("front-left") actually points to the geometric RIGHT and
// whisker_dirs[1] to the geometric LEFT. The LEFT action does ang += ..., i.e.
// it rotates toward whisker_dirs[1]. A near obstacle on flw therefore calls
// for LEFT, not RIGHT.
static inline int bot_whisker(Race* env) {
    float diff = env->flw_length - env->frw_length;
    if (diff < -0.02f) return LEFT;
    if (diff > 0.02f) return RIGHT;
    return NOOP;
}

// Privileged baseline: pure pursuit against the centerline.
//
// Two things this has to get right that the naive version did not:
//
// 1. Progress index. env->near_point_idx is deliberately cheap -- it scans
//    only offsets 0..+3 from its previous value (see update_nearest_point)
//    because it only needs a window for the whisker raycast. It therefore lags
//    the true nearest point, and a lookahead measured from a lagging index can
//    land behind the car. A plain global argmin is worse still: on a closed
//    loop it snaps across the track wherever the loop passes near itself. So
//    the bot keeps its own index and advances it forward through a local
//    window, re-acquiring globally only at episode start.
//
// 2. Lookahead in pixels, not indices. GenerateTrackCenterline varies the
//    Bezier control length per segment (dist * 0.2 / 0.3 / 0.4 depending on i),
//    so arc length per index is far from uniform and a fixed index offset is a
//    wildly varying distance.
static inline int bot_centerline(Bot* bot, Race* env) {
    const int n = env->track.total_points;

    if (bot->track_idx < 0) {
        float best = 1e30f;
        for (int i = 0; i < n; i++) {
            float dx = env->px - env->track.centerline[i].x;
            float dy = env->py - env->track.centerline[i].y;
            float d2 = dx * dx + dy * dy;
            if (d2 < best) { best = d2; bot->track_idx = i; }
        }
    } else {
        float best = 1e30f;
        int found = bot->track_idx;
        for (int off = 0; off <= 8; off++) {
            int i = (bot->track_idx + off) % n;
            float dx = env->px - env->track.centerline[i].x;
            float dy = env->py - env->track.centerline[i].y;
            float d2 = dx * dx + dy * dy;
            if (d2 < best) { best = d2; found = i; }
        }
        bot->track_idx = found;
    }

    // Walk forward by arc length until we are `lookahead` pixels ahead.
    int j = bot->track_idx;
    float acc = 0.0f;
    for (int step = 0; step < n && acc < bot->lookahead; step++) {
        int k = (j + 1) % n;
        float dx = env->track.centerline[k].x - env->track.centerline[j].x;
        float dy = env->track.centerline[k].y - env->track.centerline[j].y;
        acc += sqrtf(dx * dx + dy * dy);
        j = k;
    }

    float dx = env->track.centerline[j].x - env->px;
    float dy = env->track.centerline[j].y - env->py;
    float target = atan2f(dy, dx);

    float err = target - env->ang;
    while (err > PI) err -= PI2;
    while (err < -PI) err += PI2;

    // One action rotates frameskip * PI/turn_pi_frac radians, so a tolerance
    // finer than half that only makes the bot chatter.
    float tol = 0.5f * (float)env->frameskip * PI / env->turn_pi_frac;
    if (err > tol) return LEFT;
    if (err < -tol) return RIGHT;
    return NOOP;
}

static inline int bot_act(Bot* bot, Race* env) {
    if (env->tick == 0) bot->track_idx = -1;   // new episode: re-acquire
    if (bot->epsilon > 0.0f && rng_float(&bot->rng) < bot->epsilon) {
        return rng_below(&bot->rng, 3);
    }
    switch (bot->kind) {
        case BOT_WHISKER:    return bot_whisker(env);
        case BOT_STRAIGHT:   return NOOP;
        case BOT_RANDOM:     return rng_below(&bot->rng, 3);
        case BOT_CENTERLINE: return bot_centerline(bot, env);
        default:             return NOOP;
    }
}

#endif
