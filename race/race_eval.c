// race_eval - headless, deterministic, parallel evaluation for Race.
//
// Runs bots x tracks x episodes with no window and no Python, and reports
// behavioral metrics rather than just return. Optionally dumps transition
// shards for world-model training.
//
//   build/race_eval --bots whisker,centerline,random --tracks 64 --episodes 20
//   build/race_eval --bots whisker --tracks 8 --episodes 4 --dump data/shard
//
// Determinism: results depend only on (--seed, --tracks, --episodes, --bots,
// --epsilon, env config). Thread count never changes the numbers -- each
// (bot, track) job owns its env and its RNG streams.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>

#include "race.h"
#include "race_bots.h"
#include "race_hash.h"
#include "race_replay.h"
#include "race_shard.h"
#include "race_policy.h"

#define MAX_BOTS 8
#define MAX_THREADS 64

// ----------------------------------------------------------------- tracks ---
// Track geometry is episode-constant but the dynamics depend on it: the
// whisker raycast intersects the edge polylines and update_nearest_point walks
// the centerline. It is part of the Markov state -- the part that does not
// change within an episode -- so it is dumped once per env_idx, not per step.
#define RACE_TRACK_MAGIC 0x52434b31u /* "RCK1" */

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t n_tracks;
    uint32_t seed;
} TrackFileHeader;

// --------------------------------------------------------------- metrics ---
// Grouped so like sits with like: an outcome number and a control-smoothness
// number answer different questions and should not be read off the same row.
typedef struct {
    double n;
    // OUTCOME
    double ret, len, survived;
    // FAILURE
    double crashed, wall, offtrack, steps_to_crash;
    // PROGRESS
    double laps, crossings, sectors_per_100, steps_to_first_lap, lap1_count, coverage;
    // CONTROL
    double idle_frac, left_frac, right_frac, reversal_rate;
    // SAFETY
    double dist_mean, dist_max, min_whisker, mean_whisker;
} Agg;

// Agg is all doubles and is summed field-wise. Derive the count from the
// struct rather than hardcoding it -- a hardcoded length silently drops the
// last field whenever a metric is added, which is exactly what happened to
// mean_whisker.
static void agg_merge(Agg* d, const Agg* s) {
    double* a = (double*)d; const double* b = (const double*)s;
    for (size_t i = 0; i < sizeof(Agg) / sizeof(double); i++) a[i] += b[i];
}

// ------------------------------------------------------------------ config -
typedef struct {
    unsigned int seed;
    int tracks, track_start, episodes, max_steps, threads;
    float epsilon;
    int bots[MAX_BOTS], n_bots;
    const char* dump_prefix;
    const char* record_prefix;
    int record_episodes;
    const char* policy_path;
    Policy* policy;          // loaded once in main, read-only in workers
    // env config, mirrored from config/race.ini defaults
    int width, height, track_width, num_radial_sectors, num_points;
    int bezier_resolution, frameskip, method;
    float turn_pi_frac, maxv, w_ang, max_whisker_length, reward_yellow, corner_thresh;
} Config;

typedef struct {
    const Config* cfg;
    int thread_id;
    int job_start, job_end;   // over [0, n_bots * tracks)
    Agg* agg;                 // n_bots entries
    uint64_t* hashes;         // shared, indexed by job_id (disjoint writes)
    long records_written;
} Job;

static void env_configure(Race* env, const Config* cfg, int env_idx) {
    memset(env, 0, sizeof(*env));
    env->width = cfg->width;
    env->height = cfg->height;
    env->track_width = cfg->track_width;
    env->num_radial_sectors = cfg->num_radial_sectors;
    env->num_points = cfg->num_points;
    env->bezier_resolution = cfg->bezier_resolution;
    env->frameskip = cfg->frameskip;
    env->method = cfg->method;
    env->turn_pi_frac = cfg->turn_pi_frac;
    env->maxv = cfg->maxv;
    env->w_ang = cfg->w_ang;
    env->max_whisker_length = cfg->max_whisker_length;
    env->reward_yellow = cfg->reward_yellow;
    env->corner_thresh = cfg->corner_thresh;
    env->max_episode_steps = cfg->max_steps;
    env->render = 0;
    env->render_many = 0;
    env->mode7 = 0;
    env->seed = cfg->seed;
    env->env_idx = env_idx;
}

// Written serially from main() before workers start: the same env_idx is
// visited by every bot, and the geometry is identical each time.
static int dump_tracks(const Config* cfg, const char* prefix) {
    char path[512];
    snprintf(path, sizeof(path), "%s.tracks.bin", prefix);
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }

    TrackFileHeader hdr = {RACE_TRACK_MAGIC, RACE_STATE_VERSION,
                           (uint32_t)cfg->tracks, cfg->seed};
    fwrite(&hdr, sizeof(hdr), 1, f);

    for (int i = 0; i < cfg->tracks; i++) {
        int env_idx = cfg->track_start + i;
        Race env;
        env_configure(&env, cfg, env_idx);
        init(&env);

        int32_t meta[2] = {env_idx, env.track.total_points};
        fwrite(meta, sizeof(meta), 1, f);
        for (int i = 0; i < env.track.total_points; i++) {
            float xy[2] = {env.track.centerline[i].x, env.track.centerline[i].y};
            fwrite(xy, sizeof(xy), 1, f);
        }
        for (int i = 0; i < env.track.total_points; i++) {
            float xy[2] = {env.track.inner_edge[i].x, env.track.inner_edge[i].y};
            fwrite(xy, sizeof(xy), 1, f);
        }
        for (int i = 0; i < env.track.total_points; i++) {
            float xy[2] = {env.track.outer_edge[i].x, env.track.outer_edge[i].y};
            fwrite(xy, sizeof(xy), 1, f);
        }
        c_close(&env);
    }
    fclose(f);
    return 0;
}

static void* worker(void* arg) {
    Job* job = (Job*)arg;
    const Config* cfg = job->cfg;

    FILE* shard = NULL;
    ShardHeader hdr = {0};
    if (cfg->dump_prefix) {
        char path[512];
        snprintf(path, sizeof(path), "%s.%03d.bin", cfg->dump_prefix, job->thread_id);
        shard = fopen(path, "wb");
        if (!shard) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
        hdr.magic = RACE_SHARD_MAGIC;
        hdr.state_version = RACE_STATE_VERSION;
        hdr.obs_dim = RACE_OBS_SIZE;
        hdr.record_bytes = sizeof(Record);
        hdr.seed = cfg->seed;
        hdr.num_radial_sectors = cfg->num_radial_sectors;
        hdr.max_episode_steps = cfg->max_steps;
        hdr.width = (float)cfg->width;
        hdr.height = (float)cfg->height;
        hdr.maxv = cfg->maxv;
        hdr.frameskip = (uint32_t)cfg->frameskip;
        hdr.total_points = (uint32_t)(cfg->num_points * cfg->bezier_resolution);
        fwrite(&hdr, sizeof(hdr), 1, shard);   // n_records backfilled at close
    }

    for (int job_id = job->job_start; job_id < job->job_end; job_id++) {
        int bot_slot = job_id / cfg->tracks;
        int env_idx = cfg->track_start + (job_id % cfg->tracks);
        int bot_kind = cfg->bots[bot_slot];

        Race env;
        env_configure(&env, cfg, env_idx);
        allocate(&env);

        Bot bot;
        // Bot stream is keyed on (bot_kind, env_idx) and is independent of the
        // env's streams, so swapping policies cannot change track geometry.
        bot_init(&bot, bot_kind, cfg->epsilon, (uint64_t)cfg->seed,
                 0x9E3779B9ull * (uint64_t)(bot_kind + 1) + (uint64_t)env_idx);
        bot.policy = cfg->policy;   // shared, inference is read-only

        c_reset(&env);
        uint64_t h = race_hash_begin(&env);

        // Replays store only the action stream: the env is fully determined by
        // (config, seed, env_idx), which is what test_determinism.sh proves.
        uint8_t* rec_actions = NULL;
        uint32_t rec_n = 0;
        uint64_t rec_hash = 0;
        int rec_cap = 0;
        if (cfg->record_prefix) {
            rec_cap = cfg->record_episodes * cfg->max_steps + 8;
            rec_actions = (uint8_t*)malloc(rec_cap);
        }

        for (int ep = 0; ep < cfg->episodes; ep++) {
            double ret = 0, dist_sum = 0, dist_max = 0, min_whisk = 1.0, whisk_sum = 0;
            int steps = 0, reversals = 0, prev_turn = NOOP;
            int act_count[3] = {0, 0, 0};
            int first_crash = -1, first_lap_step = -1;
            int visited[RACE_MAX_SECTORS] = {0};
            int start_episode = env.episode_id;
            int S = cfg->num_radial_sectors;

            while (env.episode_id == start_episode) {
                int a = bot_act(&bot, &env);

                Record rec;
                if (shard) {
                    rec.env_idx = env_idx;
                    rec.episode_id = env.episode_id;
                    rec.t = env.tick;
                    rec.action = a;
                    rec.bot_kind = (uint8_t)bot_kind;
                    rec.pad = 0;
                    memcpy(rec.obs, env.observations, sizeof(rec.obs));
                }

                // Behavioral telemetry sampled on s_t, before the transition.
                float cx = env.track.centerline[env.near_point_idx].x;
                float cy = env.track.centerline[env.near_point_idx].y;
                double d = sqrt((env.px - cx) * (env.px - cx) + (env.py - cy) * (env.py - cy));
                dist_sum += d;
                if (d > dist_max) dist_max = d;
                double mw = env.flw_length < env.frw_length ? env.flw_length : env.frw_length;
                if (mw < min_whisk) min_whisk = mw;
                whisk_sum += 0.5 * (env.flw_length + env.frw_length);
                if (env.current_sector >= 0 && env.current_sector < RACE_MAX_SECTORS) {
                    visited[env.current_sector] = 1;
                }
                if (a >= 0 && a < 3) act_count[a]++;
                if ((a == LEFT && prev_turn == RIGHT) || (a == RIGHT && prev_turn == LEFT)) {
                    reversals++;
                }
                if (a != NOOP) prev_turn = a;

                env.actions[0] = (double)a;
                c_step(&env);
                steps++;
                ret += env.rewards[0];

                h = race_hash_step(h, &env, a);
                if (rec_actions && ep < cfg->record_episodes && (int)rec_n < rec_cap) {
                    rec_actions[rec_n++] = (uint8_t)a;
                    rec_hash = h;   // hash prefix covering exactly the recorded steps
                }

                if (shard) {
                    rec.reward = env.rewards[0];
                    rec.terminated = (uint8_t)env.terminated;
                    rec.truncated = (uint8_t)env.truncated;
                    fwrite(&rec, sizeof(rec), 1, shard);
                    job->records_written++;
                }

                // Completion time: steps to finish one full lap. Only counted
                // over episodes that actually managed a lap, so it is not
                // silently inflated by episodes that crashed early.
                if (first_lap_step < 0 && env.sector_crossings >= S) first_lap_step = steps;
                if (env.terminated && first_crash < 0) first_crash = steps;
                if (env.terminated || env.truncated) break;
            }

            // c_step already logged and reset; read the outcome off the log
            // delta rather than the (now reset) live fields.
            int crashed = (first_crash >= 0);
            int cov = 0;
            for (int i = 0; i < S; i++) cov += visited[i];
            double crossings = ret / (cfg->reward_yellow > 0 ? cfg->reward_yellow : 1.0);
            double inv = steps ? 1.0 / steps : 0.0;

            Agg* g = &job->agg[bot_slot];
            g->n += 1;
            g->ret += ret;
            g->len += steps;
            g->survived += crashed ? 0 : 1;
            g->crashed += crashed ? 1 : 0;
            g->steps_to_crash += crashed ? first_crash : cfg->max_steps;
            g->laps += crossings / (double)S;
            g->crossings += crossings;
            g->sectors_per_100 += crossings * inv * 100.0;
            if (first_lap_step >= 0) { g->steps_to_first_lap += first_lap_step; g->lap1_count += 1; }
            g->coverage += (double)cov / S;
            g->idle_frac += act_count[NOOP] * inv;
            g->left_frac += act_count[LEFT] * inv;
            g->right_frac += act_count[RIGHT] * inv;
            g->reversal_rate += reversals * inv;
            g->dist_mean += dist_sum * inv;
            g->dist_max += dist_max;
            g->min_whisker += min_whisk;
            g->mean_whisker += whisk_sum * inv;
        }

        // Cause-of-death split comes from the env log, which c_step maintains.
        job->agg[bot_slot].wall += env.log.wall_crash_rate;
        job->agg[bot_slot].offtrack += env.log.offtrack_rate;
        job->hashes[job_id] = h;

        if (rec_actions) {
            ReplayHeader rh = {0};
            rh.magic = RACE_REPLAY_MAGIC;
            rh.state_version = RACE_STATE_VERSION;
            rh.seed = cfg->seed;
            rh.env_idx = env_idx;
            rh.n_steps = rec_n;
            rh.traj_hash = rec_hash;
            rh.width = cfg->width; rh.height = cfg->height;
            rh.track_width = cfg->track_width;
            rh.num_radial_sectors = cfg->num_radial_sectors;
            rh.num_points = cfg->num_points;
            rh.bezier_resolution = cfg->bezier_resolution;
            rh.frameskip = cfg->frameskip; rh.method = cfg->method;
            rh.max_episode_steps = cfg->max_steps;
            rh.turn_pi_frac = cfg->turn_pi_frac; rh.maxv = cfg->maxv;
            rh.w_ang = cfg->w_ang;
            rh.max_whisker_length = cfg->max_whisker_length;
            rh.reward_yellow = cfg->reward_yellow;
            rh.corner_thresh = cfg->corner_thresh;
            rh.bot_kind = (uint8_t)bot_kind;

            char path[512];
            snprintf(path, sizeof(path), "%s.%s.%03d.rr",
                     cfg->record_prefix, BOT_NAMES[bot_kind], env_idx);
            if (replay_write(path, &rh, rec_actions) != 0) {
                fprintf(stderr, "failed to write %s\n", path);
            }
            free(rec_actions);
        }

        free_allocated(&env);
    }

    if (shard) {
        hdr.n_records = (uint32_t)job->records_written;
        fseek(shard, 0, SEEK_SET);
        fwrite(&hdr, sizeof(hdr), 1, shard);
        fclose(shard);
    }
    return NULL;
}

// ------------------------------------------------------------------- main ---
static int parse_bot(const char* name) {
    for (int i = 0; i < BOT_KIND_COUNT; i++) {
        if (strcmp(name, BOT_NAMES[i]) == 0) return i;
    }
    return -1;
}

static void usage(void) {
    fprintf(stderr,
        "usage: race_eval [options]\n"
        "  --bots LIST      comma separated: whisker,straight,random,centerline\n"
        "  --seed N         global seed (default 42)\n"
        "  --tracks N       number of distinct tracks / env_idx values (default 32)\n"
        "  --track-start N  first env_idx (default 0). With --tracks this cuts a\n"
        "                   disjoint slice of track space, so train and holdout\n"
        "                   corpora cannot overlap.\n"
        "  --episodes N     episodes per (bot, track) (default 10)\n"
        "  --max-steps N    episode timer (default 2000)\n"
        "  --epsilon F      per-step random action probability (default 0)\n"
        "  --threads N      worker threads (default 8); never affects results\n"
        "  --dump PREFIX    write transition shards to PREFIX.NNN.bin\n"
        "  --hash           print per-(bot,track) trajectory hashes and exit\n"
        "  --policy FILE    load a .rpo policy; enables the `policy` bot name\n"
        "  --record PREFIX  write replays to PREFIX.<bot>.<track>.rr\n"
        "  --record-episodes N   episodes per replay (default 1)\n"
        "  --json PATH      write metrics as JSON (default: stdout table)\n");
}

int main(int argc, char** argv) {
    Config cfg = {
        .seed = 42, .tracks = 32, .track_start = 0, .episodes = 10, .max_steps = 2000,
        .threads = 8, .epsilon = 0.0f, .dump_prefix = NULL,
        .record_prefix = NULL, .record_episodes = 1,
        .policy_path = NULL, .policy = NULL,
        .width = 1080, .height = 720, .track_width = 75,
        .num_radial_sectors = 16, .num_points = 16, .bezier_resolution = 4,
        .frameskip = 4, .method = 2,
        .turn_pi_frac = 40, .maxv = 5, .w_ang = 0.777f,
        .max_whisker_length = 100, .reward_yellow = 0.2f, .corner_thresh = 0.5f,
    };
    const char* json_path = NULL;
    int want_hash = 0;
    char bots_arg[256] = "whisker,straight,random,centerline";

    for (int i = 1; i < argc; i++) {
        const char* a = argv[i];
        #define NEXT() (++i < argc ? argv[i] : (usage(), exit(1), ""))
        if      (!strcmp(a, "--bots"))      snprintf(bots_arg, sizeof(bots_arg), "%s", NEXT());
        else if (!strcmp(a, "--seed"))      cfg.seed = (unsigned)atoi(NEXT());
        else if (!strcmp(a, "--tracks"))    cfg.tracks = atoi(NEXT());
        else if (!strcmp(a, "--track-start")) cfg.track_start = atoi(NEXT());
        else if (!strcmp(a, "--episodes"))  cfg.episodes = atoi(NEXT());
        else if (!strcmp(a, "--max-steps")) cfg.max_steps = atoi(NEXT());
        else if (!strcmp(a, "--epsilon"))   cfg.epsilon = (float)atof(NEXT());
        else if (!strcmp(a, "--threads"))   cfg.threads = atoi(NEXT());
        else if (!strcmp(a, "--dump"))      cfg.dump_prefix = NEXT();
        else if (!strcmp(a, "--record"))    cfg.record_prefix = NEXT();
        else if (!strcmp(a, "--policy"))    cfg.policy_path = NEXT();
        else if (!strcmp(a, "--record-episodes")) cfg.record_episodes = atoi(NEXT());
        else if (!strcmp(a, "--json"))      json_path = NEXT();
        else if (!strcmp(a, "--sectors"))   cfg.num_radial_sectors = atoi(NEXT());
        else if (!strcmp(a, "--frameskip")) cfg.frameskip = atoi(NEXT());
        else if (!strcmp(a, "--method"))    cfg.method = atoi(NEXT());
        else if (!strcmp(a, "--hash"))      want_hash = 1;
        else { usage(); return 1; }
        #undef NEXT
    }

    char* tok = strtok(bots_arg, ",");
    while (tok && cfg.n_bots < MAX_BOTS) {
        int k = parse_bot(tok);
        if (k < 0) { fprintf(stderr, "unknown bot: %s\n", tok); return 1; }
        cfg.bots[cfg.n_bots++] = k;
        tok = strtok(NULL, ",");
    }
    if (cfg.n_bots == 0) { usage(); return 1; }
    if (cfg.threads < 1) cfg.threads = 1;
    if (cfg.threads > MAX_THREADS) cfg.threads = MAX_THREADS;

    // Load before threads start; policy_act only reads, so one copy is shared.
    Policy pol;
    if (cfg.policy_path) {
        if (policy_load(&pol, cfg.policy_path) != 0) return 1;
        cfg.policy = &pol;
        fprintf(stderr, "loaded policy %s (%u layers, obs %u -> act %u)\n",
                cfg.policy_path, pol.h.n_layers, pol.h.obs_dim, pol.h.act_dim);
    }
    for (int b = 0; b < cfg.n_bots; b++) {
        if (cfg.bots[b] == BOT_POLICY && !cfg.policy) {
            fprintf(stderr, "bot `policy` requires --policy FILE\n");
            return 1;
        }
    }

    int n_jobs = cfg.n_bots * cfg.tracks;
    if (cfg.threads > n_jobs) cfg.threads = n_jobs;

    if (cfg.dump_prefix && dump_tracks(&cfg, cfg.dump_prefix) != 0) return 1;

    pthread_t threads[MAX_THREADS];
    Job jobs[MAX_THREADS];
    Agg* aggs = (Agg*)calloc(cfg.threads * cfg.n_bots, sizeof(Agg));
    uint64_t* hashes = (uint64_t*)calloc(n_jobs, sizeof(uint64_t));

    for (int t = 0; t < cfg.threads; t++) {
        jobs[t].cfg = &cfg;
        jobs[t].thread_id = t;
        jobs[t].job_start = (int)((long)n_jobs * t / cfg.threads);
        jobs[t].job_end = (int)((long)n_jobs * (t + 1) / cfg.threads);
        jobs[t].agg = &aggs[t * cfg.n_bots];
        jobs[t].hashes = hashes;
        jobs[t].records_written = 0;
        pthread_create(&threads[t], NULL, worker, &jobs[t]);
    }
    long total_records = 0;
    for (int t = 0; t < cfg.threads; t++) {
        pthread_join(threads[t], NULL);
        total_records += jobs[t].records_written;
    }

    Agg* total = (Agg*)calloc(cfg.n_bots, sizeof(Agg));
    for (int t = 0; t < cfg.threads; t++) {
        for (int b = 0; b < cfg.n_bots; b++) agg_merge(&total[b], &aggs[t * cfg.n_bots + b]);
    }

    if (want_hash) {
        // Sorted by job_id so output is byte-identical for any --threads.
        for (int j = 0; j < n_jobs; j++) {
            printf("%s %d %016llx\n", BOT_NAMES[cfg.bots[j / cfg.tracks]],
                   cfg.track_start + (j % cfg.tracks),
                   (unsigned long long)hashes[j]);
        }
        free(aggs); free(hashes);
        return 0;
    }

    FILE* out = stdout;
    if (json_path) {
        out = fopen(json_path, "w");
        if (!out) { fprintf(stderr, "cannot write %s\n", json_path); return 1; }
    }

    // Every rate is a per-episode mean, so divide once here.
    #define AVG(f) (g->f / (g->n > 0 ? g->n : 1))

    if (json_path) {
        fprintf(out, "{\n  \"seed\": %u,\n  \"tracks\": %d,\n  \"track_start\": %d,\n"
                     "  \"episodes\": %d,\n  \"max_steps\": %d,\n  \"epsilon\": %g,\n"
                     "  \"state_version\": %d,\n  \"obs_dim\": %d,\n  \"bots\": {\n",
                cfg.seed, cfg.tracks, cfg.track_start, cfg.episodes, cfg.max_steps,
                cfg.epsilon, RACE_STATE_VERSION, RACE_OBS_SIZE);
        for (int b = 0; b < cfg.n_bots; b++) {
            Agg* g = &total[b];
            double lap1 = g->lap1_count > 0 ? g->steps_to_first_lap / g->lap1_count : -1;
            fprintf(out,
                "    \"%s\": {\n"
                "      \"outcome\":  {\"episodes\": %.0f, \"return\": %.4f, \"length\": %.2f, \"survival_rate\": %.4f},\n"
                "      \"failure\":  {\"crash_rate\": %.4f, \"wall\": %.4f, \"offtrack\": %.4f, \"steps_to_crash\": %.2f},\n"
                "      \"progress\": {\"laps\": %.4f, \"sector_crossings\": %.2f, \"sectors_per_100_steps\": %.3f, \"steps_to_first_lap\": %.2f, \"lap_completion_rate\": %.4f, \"track_coverage\": %.4f},\n"
                "      \"control\":  {\"idle_frac\": %.4f, \"left_frac\": %.4f, \"right_frac\": %.4f, \"reversal_rate\": %.4f},\n"
                "      \"safety\":   {\"mean_dist_centerline\": %.3f, \"max_dist_centerline\": %.3f, \"min_whisker\": %.4f, \"mean_whisker\": %.4f}\n"
                "    }%s\n",
                BOT_NAMES[cfg.bots[b]],
                g->n, AVG(ret), AVG(len), AVG(survived),
                AVG(crashed), AVG(wall), AVG(offtrack), AVG(steps_to_crash),
                AVG(laps), AVG(crossings), AVG(sectors_per_100), lap1,
                g->lap1_count / (g->n > 0 ? g->n : 1), AVG(coverage),
                AVG(idle_frac), AVG(left_frac), AVG(right_frac), AVG(reversal_rate),
                AVG(dist_mean), AVG(dist_max), AVG(min_whisker), AVG(mean_whisker),
                (b + 1 < cfg.n_bots) ? "," : "");
        }
        fprintf(out, "  },\n  \"records_written\": %ld\n}\n", total_records);
        fclose(out);
        fprintf(stderr, "wrote %s\n", json_path);
    } else {
        printf("Race eval   seed=%u  tracks=%d..%d  episodes=%d  max_steps=%d  eps=%g\n",
               cfg.seed, cfg.track_start, cfg.track_start + cfg.tracks - 1,
               cfg.episodes, cfg.max_steps, cfg.epsilon);

        // Five groups, printed as separate blocks. Reading a smoothness number
        // off the same row as an outcome number invites comparing things that
        // are not comparable.
        printf("\n  OUTCOME                     eps    return    length  survive%%\n");
        for (int b = 0; b < cfg.n_bots; b++) { Agg* g = &total[b];
            printf("    %-14s %11.0f %9.3f %9.1f %9.1f\n", BOT_NAMES[cfg.bots[b]],
                   g->n, AVG(ret), AVG(len), 100*AVG(survived)); }

        printf("\n  FAILURE                  crash%%     wall%%  offtrack%%  steps2crash\n");
        for (int b = 0; b < cfg.n_bots; b++) { Agg* g = &total[b];
            printf("    %-14s %10.1f %9.1f %10.1f %12.1f\n", BOT_NAMES[cfg.bots[b]],
                   100*AVG(crashed), 100*AVG(wall), 100*AVG(offtrack), AVG(steps_to_crash)); }

        printf("\n  PROGRESS                   laps  sect/100  lap1_steps   lap%%   coverage\n");
        for (int b = 0; b < cfg.n_bots; b++) { Agg* g = &total[b];
            double lap1 = g->lap1_count > 0 ? g->steps_to_first_lap / g->lap1_count : 0;
            printf("    %-14s %10.3f %9.2f %11s %6.1f %10.3f\n", BOT_NAMES[cfg.bots[b]],
                   AVG(laps), AVG(sectors_per_100),
                   g->lap1_count > 0 ? (snprintf(NULL,0,"%.0f",lap1), ({ static char buf[8][16]; snprintf(buf[b],16,"%.0f",lap1); buf[b]; })) : "-",
                   100*g->lap1_count/(g->n>0?g->n:1), AVG(coverage)); }

        printf("\n  CONTROL                   idle%%     left%%     right%%  reversals\n");
        for (int b = 0; b < cfg.n_bots; b++) { Agg* g = &total[b];
            printf("    %-14s %10.1f %9.1f %10.1f %10.3f\n", BOT_NAMES[cfg.bots[b]],
                   100*AVG(idle_frac), 100*AVG(left_frac), 100*AVG(right_frac), AVG(reversal_rate)); }

        printf("\n  SAFETY                 dist_avg  dist_max  whisk_min  whisk_avg\n");
        for (int b = 0; b < cfg.n_bots; b++) { Agg* g = &total[b];
            printf("    %-14s %10.2f %9.2f %10.3f %10.3f\n", BOT_NAMES[cfg.bots[b]],
                   AVG(dist_mean), AVG(dist_max), AVG(min_whisker), AVG(mean_whisker)); }

        if (total_records) printf("\n  transitions written: %ld\n", total_records);
    }
    #undef AVG

    free(aggs); free(total); free(hashes);
    return 0;
}
