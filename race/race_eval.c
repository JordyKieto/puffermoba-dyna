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

#define MAX_BOTS 8
#define MAX_THREADS 64

// --------------------------------------------------------------- metrics ---
typedef struct {
    double n;
    double ret, len, laps, crossings;
    double crash, wall, offtrack, trunc;
    double dist_mean, dist_max;
    double min_whisker;
    double reversal_rate, straight_frac;
    double coverage;
    double first_crash_step;
} Agg;

static void agg_add(Agg* a, double ret, double len, double laps, double crossings,
                    double crash, double wall, double offtrack, double trunc,
                    double dist_mean, double dist_max, double min_whisker,
                    double reversal_rate, double straight_frac, double coverage,
                    double first_crash_step) {
    a->n += 1;
    a->ret += ret; a->len += len; a->laps += laps; a->crossings += crossings;
    a->crash += crash; a->wall += wall; a->offtrack += offtrack; a->trunc += trunc;
    a->dist_mean += dist_mean; a->dist_max += dist_max;
    a->min_whisker += min_whisker;
    a->reversal_rate += reversal_rate; a->straight_frac += straight_frac;
    a->coverage += coverage; a->first_crash_step += first_crash_step;
}

static void agg_merge(Agg* dst, const Agg* src) {
    dst->n += src->n;
    dst->ret += src->ret; dst->len += src->len; dst->laps += src->laps;
    dst->crossings += src->crossings;
    dst->crash += src->crash; dst->wall += src->wall;
    dst->offtrack += src->offtrack; dst->trunc += src->trunc;
    dst->dist_mean += src->dist_mean; dst->dist_max += src->dist_max;
    dst->min_whisker += src->min_whisker;
    dst->reversal_rate += src->reversal_rate;
    dst->straight_frac += src->straight_frac;
    dst->coverage += src->coverage;
    dst->first_crash_step += src->first_crash_step;
}

// ----------------------------------------------------------------- tracks ---
// Track geometry is episode-constant but the dynamics depend on it: the
// whisker raycast intersects inner_edge/outer_edge, and update_nearest_point
// walks the centerline. It is therefore part of the Markov state, just the
// part that does not change within an episode -- so it is dumped once per
// env_idx rather than once per step. See docs/CONTRACT.md.
#define RACE_TRACK_MAGIC 0x52434b31u /* "RCK1" */

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t n_tracks;
    uint32_t seed;
} TrackFileHeader;

// ------------------------------------------------------------------ config -
typedef struct {
    unsigned int seed;
    int tracks, track_start, episodes, max_steps, threads;
    float epsilon;
    int bots[MAX_BOTS], n_bots;
    const char* dump_prefix;
    const char* record_prefix;
    int record_episodes;
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
            double ret = 0, dist_sum = 0, dist_max = 0, min_whisk = 1.0;
            int steps = 0, reversals = 0, straights = 0, prev_turn = NOOP;
            int first_crash = -1;
            int visited[RACE_MAX_SECTORS] = {0};
            int start_episode = env.episode_id;

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
                if (env.current_sector >= 0 && env.current_sector < RACE_MAX_SECTORS) {
                    visited[env.current_sector] = 1;
                }
                if (a == NOOP) straights++;
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

                if (env.terminated && first_crash < 0) first_crash = steps;
                if (env.terminated || env.truncated) break;
            }

            // c_step already logged and reset; read the outcome off the log
            // delta rather than the (now reset) live fields.
            int crashed = (first_crash >= 0);
            int cov = 0;
            for (int i = 0; i < cfg->num_radial_sectors; i++) cov += visited[i];

            double crossings = ret / (cfg->reward_yellow > 0 ? cfg->reward_yellow : 1.0);
            agg_add(&job->agg[bot_slot], ret, steps,
                    crossings / (double)cfg->num_radial_sectors, crossings,
                    crashed ? 1 : 0,
                    0, 0,             /* cause split filled from log below */
                    crashed ? 0 : 1,
                    steps ? dist_sum / steps : 0, dist_max, min_whisk,
                    steps ? (double)reversals / steps : 0,
                    steps ? (double)straights / steps : 0,
                    (double)cov / cfg->num_radial_sectors,
                    crashed ? first_crash : cfg->max_steps);
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
        "  --record PREFIX  write replays to PREFIX.<bot>.<track>.rr\n"
        "  --record-episodes N   episodes per replay (default 1)\n"
        "  --json PATH      write metrics as JSON (default: stdout table)\n");
}

int main(int argc, char** argv) {
    Config cfg = {
        .seed = 42, .tracks = 32, .track_start = 0, .episodes = 10, .max_steps = 2000,
        .threads = 8, .epsilon = 0.0f, .dump_prefix = NULL,
        .record_prefix = NULL, .record_episodes = 1,
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

    if (json_path) {
        fprintf(out, "{\n  \"seed\": %u,\n  \"tracks\": %d,\n  \"track_start\": %d,\n  \"episodes\": %d,\n"
                     "  \"max_steps\": %d,\n  \"epsilon\": %g,\n"
                     "  \"state_version\": %d,\n  \"obs_dim\": %d,\n  \"bots\": {\n",
                cfg.seed, cfg.tracks, cfg.track_start, cfg.episodes, cfg.max_steps, cfg.epsilon,
                RACE_STATE_VERSION, RACE_OBS_SIZE);
        for (int b = 0; b < cfg.n_bots; b++) {
            Agg* g = &total[b];
            double n = g->n > 0 ? g->n : 1;
            fprintf(out,
                "    \"%s\": {\"episodes\": %.0f, \"return\": %.4f, \"length\": %.2f,\n"
                "      \"laps\": %.4f, \"sector_crossings\": %.2f, \"crash_rate\": %.4f,\n"
                "      \"survival_rate\": %.4f, \"track_coverage\": %.4f,\n"
                "      \"mean_dist_centerline\": %.3f, \"max_dist_centerline\": %.3f,\n"
                "      \"min_whisker\": %.4f, \"steer_reversal_rate\": %.4f,\n"
                "      \"straight_frac\": %.4f, \"steps_to_crash\": %.2f}%s\n",
                BOT_NAMES[cfg.bots[b]], g->n, g->ret / n, g->len / n,
                g->laps / n, g->crossings / n, g->crash / n, g->trunc / n,
                g->coverage / n, g->dist_mean / n, g->dist_max / n,
                g->min_whisker / n, g->reversal_rate / n, g->straight_frac / n,
                g->first_crash_step / n,
                (b + 1 < cfg.n_bots) ? "," : "");
        }
        fprintf(out, "  },\n  \"records_written\": %ld\n}\n", total_records);
        fclose(out);
        fprintf(stderr, "wrote %s\n", json_path);
    } else {
        printf("Race eval  seed=%u tracks=%d episodes=%d max_steps=%d eps=%g\n",
               cfg.seed, cfg.tracks, cfg.episodes, cfg.max_steps, cfg.epsilon);
        printf("%-12s %7s %8s %8s %7s %7s %8s %8s %8s %8s\n",
               "bot", "eps", "return", "length", "laps", "crash", "cover",
               "dist", "revrs", "minwhk");
        for (int b = 0; b < cfg.n_bots; b++) {
            Agg* g = &total[b];
            double n = g->n > 0 ? g->n : 1;
            printf("%-12s %7.0f %8.3f %8.1f %7.3f %7.3f %8.3f %8.2f %8.3f %8.3f\n",
                   BOT_NAMES[cfg.bots[b]], g->n, g->ret / n, g->len / n,
                   g->laps / n, g->crash / n, g->coverage / n,
                   g->dist_mean / n, g->reversal_rate / n, g->min_whisker / n);
        }
        if (total_records) printf("\ntransitions written: %ld\n", total_records);
    }

    free(aggs); free(total); free(hashes);
    return 0;
}
