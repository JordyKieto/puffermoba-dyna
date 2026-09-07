// race_play - the replay client. Reads the same .rr files race_eval writes.
//
//   build/race_play replays/whisker.007.rr                 # watch it
//   build/race_play replays/whisker.007.rr --mp4 out.mp4   # export video
//   build/race_play replays/whisker.007.rr --verify        # headless check
//
// Temporal alignment: one recorded action drives exactly one c_step and
// exactly one rendered frame. c_step already applies frameskip internally, so
// stepping the env extra times per action (which the upstream demo did, to get
// smoother video) would desync the client from the server by a factor of
// frameskip. Frame t on screen is step t on the server, always.
//
// Every run recomputes the trajectory hash and compares it to the one stored
// at record time, so a desync is reported rather than silently rendered.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "race.h"
#include "race_hash.h"
#include "race_replay.h"
#include "race_shard.h"

static void usage(void) {
    fprintf(stderr,
        "usage: race_play REPLAY.rr [options]\n"
        "  --mp4 PATH     encode to mp4 (pipes raw frames to ffmpeg)\n"
        "  --fps N        playback / export frame rate (default 30)\n"
        "  --verify       no window, just re-run and check the hash\n"
        "  --mode7        first-person camera instead of top-down\n"
        "  --loop         repeat playback until the window is closed\n"
        "  --rgb-out DIR  write frame_%%06d.png plus an index-aligned\n"
        "                 states.bin in the same format race_eval --dump uses\n"
        "  --capture-size WxH  output resolution for png/mp4\n"
        "                 (default: the replay's own width x height)\n");
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 1; }
    const char* path = argv[1];
    const char* mp4 = NULL;
    int fps = 30, verify_only = 0, mode7 = 0, loop = 0;
    const char* rgb_out = NULL;
    int out_w = 0, out_h = 0;   // 0 = take the replay's logical size

    for (int i = 2; i < argc; i++) {
        if      (!strcmp(argv[i], "--mp4") && i + 1 < argc) mp4 = argv[++i];
        else if (!strcmp(argv[i], "--fps") && i + 1 < argc) fps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--verify")) verify_only = 1;
        else if (!strcmp(argv[i], "--mode7")) mode7 = 1;
        else if (!strcmp(argv[i], "--loop")) loop = 1;
        else if (!strcmp(argv[i], "--rgb-out") && i + 1 < argc) rgb_out = argv[++i];
        else if (!strcmp(argv[i], "--capture-size") && i + 1 < argc) {
            if (sscanf(argv[++i], "%dx%d", &out_w, &out_h) != 2) {
                fprintf(stderr, "--capture-size wants WxH, e.g. 1080x720\n");
                return 1;
            }
        }
        else { usage(); return 1; }
    }

    ReplayHeader h;
    uint8_t* actions = NULL;
    if (replay_read(path, &h, &actions) != 0) return 1;

    printf("replay %s\n", path);
    printf("  seed=%u env_idx=%d steps=%u frameskip=%d bot=%u\n",
           h.seed, h.env_idx, h.n_steps, h.frameskip, h.bot_kind);

    // Capture resolution is pinned to the replay's logical size, NOT to
    // whatever framebuffer the platform hands back. macOS HiDPI returns a 2x
    // buffer (a 1080x720 env captures at 2160x1440), so capturing raw would
    // make a dataset's resolution depend on which machine rendered it.
    if (out_w <= 0 || out_h <= 0) { out_w = h.width; out_h = h.height; }

    Race env;
    replay_apply(&env, &h);
    env.mode7 = mode7;
    allocate(&env);
    c_reset(&env);

    // ---- headless verification -------------------------------------------
    if (verify_only) {
        uint64_t hash = race_hash_begin(&env);
        for (uint32_t t = 0; t < h.n_steps; t++) {
            env.actions[0] = (double)actions[t];
            c_step(&env);
            hash = race_hash_step(hash, &env, actions[t]);
        }
        int ok = (hash == h.traj_hash);
        printf("  recorded hash %016llx\n", (unsigned long long)h.traj_hash);
        printf("  replayed hash %016llx\n", (unsigned long long)hash);
        printf("  %s\n", ok ? "MATCH - client and server agree" : "MISMATCH - desync");
        free(actions);
        free_allocated(&env);
        return ok ? 0 : 2;
    }

    // ---- rendered playback ------------------------------------------------
    env.client = make_client(&env);
    SetTargetFPS(fps);

    FILE* pipe = NULL;
    int native_w = 0, native_h = 0;

    // RGB / state sync. The state record is captured BEFORE c_step and the
    // frame is captured after the c_render for that same step, so record t and
    // frame_%06d.png at t describe the same instant. Written in the shard
    // format so racewm/shards.py reads RGB captures and headless dumps alike.
    FILE* state_f = NULL;
    ShardHeader shdr;
    uint32_t rgb_n = 0;
    if (rgb_out) {
        char cmd[600];
        snprintf(cmd, sizeof(cmd), "mkdir -p \"%s\"", rgb_out);
        if (system(cmd) != 0) { fprintf(stderr, "cannot create %s\n", rgb_out); return 1; }
        char sp[600];
        snprintf(sp, sizeof(sp), "%s/states.bin", rgb_out);
        state_f = fopen(sp, "wb");
        if (!state_f) { fprintf(stderr, "cannot write %s\n", sp); return 1; }
        shard_header_init(&shdr, &env, h.seed, h.num_points * h.bezier_resolution);
        fwrite(&shdr, sizeof(shdr), 1, state_f);
    }

    uint32_t t = 0;
    uint64_t hash = race_hash_begin(&env);
    int reported = 0;

    while (!WindowShouldClose()) {
        if (t < h.n_steps) {
            Record rec;
            if (state_f) {
                rec.env_idx = h.env_idx;
                rec.episode_id = env.episode_id;
                rec.t = env.tick;
                rec.action = actions[t];
                rec.bot_kind = h.bot_kind;
                rec.pad = 0;
                memcpy(rec.obs, env.observations, sizeof(rec.obs));
            }
            env.actions[0] = (double)actions[t];
            c_step(&env);
            hash = race_hash_step(hash, &env, actions[t]);
            if (state_f) {
                rec.reward = env.rewards[0];
                rec.terminated = (uint8_t)env.terminated;
                rec.truncated = (uint8_t)env.truncated;
                fwrite(&rec, sizeof(rec), 1, state_f);
            }
            t++;
        } else if (loop) {
            c_reset(&env);
            hash = race_hash_begin(&env);
            t = 0;
        } else if (!reported) {
            int ok = (hash == h.traj_hash);
            printf("  playback complete: hash %s (%016llx vs %016llx)\n",
                   ok ? "MATCH" : "MISMATCH",
                   (unsigned long long)hash, (unsigned long long)h.traj_hash);
            reported = 1;
            if (mp4) break;
        }

        c_render(&env);

        if ((mp4 || rgb_out) && rgb_n < t) {
            // LoadImageFromScreen reads back the framebuffer c_render just
            // presented, so the captured frame is exactly the frame shown --
            // no separate offscreen render path to drift out of sync. One
            // capture feeds both the png sequence and the encoder, so they can
            // never disagree.
            Image img = LoadImageFromScreen();
            if (!native_w) {
                native_w = img.width;
                native_h = img.height;
                if (native_w != out_w || native_h != out_h) {
                    printf("  framebuffer %dx%d -> capturing at %dx%d\n",
                           native_w, native_h, out_w, out_h);
                }
            }
            if (img.width != out_w || img.height != out_h) {
                ImageResize(&img, out_w, out_h);
            }

            if (rgb_out) {
                char fp[600];
                snprintf(fp, sizeof(fp), "%s/frame_%06u.png", rgb_out, rgb_n);
                ExportImage(img, fp);
            }
            if (mp4) {
                if (!pipe) {
                    char cmd[1024];
                    snprintf(cmd, sizeof(cmd),
                        "ffmpeg -hide_banner -loglevel error -y -f rawvideo "
                        "-pixel_format rgba -video_size %dx%d -framerate %d -i - "
                        "-c:v libx264 -preset fast -pix_fmt yuv420p \"%s\"",
                        out_w, out_h, fps, mp4);
                    pipe = popen(cmd, "w");
                    if (!pipe) {
                        fprintf(stderr, "cannot start ffmpeg -- is it installed?\n");
                        UnloadImage(img);
                        break;
                    }
                    printf("  encoding %dx%d @ %dfps -> %s\n", out_w, out_h, fps, mp4);
                }
                fwrite(img.data, 4, (size_t)out_w * out_h, pipe);
            }
            UnloadImage(img);
            rgb_n++;
        }
    }

    if (pipe) {
        pclose(pipe);
        printf("  wrote %s\n", mp4);
    }
    if (state_f) {
        shdr.n_records = rgb_n;
        fseek(state_f, 0, SEEK_SET);
        fwrite(&shdr, sizeof(shdr), 1, state_f);
        fclose(state_f);

        // Manifest so a capture is self-describing: resolution, fps, and the
        // replay it came from, including the hash that proves it matched.
        char mp[600];
        snprintf(mp, sizeof(mp), "%s/capture.json", rgb_out);
        FILE* mf = fopen(mp, "w");
        if (mf) {
            fprintf(mf,
                "{\n"
                "  \"replay\": \"%s\",\n"
                "  \"frames\": %u,\n"
                "  \"width\": %d,\n"
                "  \"height\": %d,\n"
                "  \"native_width\": %d,\n"
                "  \"native_height\": %d,\n"
                "  \"fps\": %d,\n"
                "  \"seed\": %u,\n"
                "  \"env_idx\": %d,\n"
                "  \"frameskip\": %d,\n"
                "  \"state_version\": %d,\n"
                "  \"recorded_hash\": \"%016llx\",\n"
                "  \"replayed_hash\": \"%016llx\",\n"
                "  \"hash_match\": %s\n"
                "}\n",
                path, rgb_n, out_w, out_h, native_w, native_h, fps,
                h.seed, h.env_idx, h.frameskip, RACE_STATE_VERSION,
                (unsigned long long)h.traj_hash, (unsigned long long)hash,
                (hash == h.traj_hash) ? "true" : "false");
            fclose(mf);
        }
        printf("  wrote %u frames + states.bin + capture.json to %s\n", rgb_n, rgb_out);
    }
    free(actions);
    close_client(env.client);
    free_allocated(&env);
    return 0;
}
