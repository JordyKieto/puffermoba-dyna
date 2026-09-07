// Interactive Race demo. Hold LEFT SHIFT to drive with the arrow keys,
// otherwise a scripted whisker-following bot drives.
//
// Upstream whisker_racer.c loaded resources/whisker_racer/puffer_whisker_racer_weights.bin
// into a LinearLSTM with obs_size 3. The observation is now the 30-dim Markov
// state (docs/CONTRACT.md), so those weights no longer apply and the demo
// drives itself with the same policy the eval CLI uses as a baseline.

#include <stdio.h>
#include "race.h"
#include "race_bots.h"

int main(int argc, char** argv) {
    unsigned int seed = (argc > 1) ? (unsigned int)atoi(argv[1]) : 42;
    int env_idx = (argc > 2) ? atoi(argv[2]) : 0;

    Race env = {
        .frameskip = 1,
        .width = 1080,
        .height = 720,
        .max_whisker_length = 100,
        .turn_pi_frac = 40,
        .maxv = 5,
        .render = 0,
        .continuous = 0,
        .reward_yellow = 0.2,
        .reward_green = -0.001,
        .track_width = 75,
        .num_radial_sectors = 16,
        .num_points = 16,
        .bezier_resolution = 4,
        .w_ang = 0.777,
        .corner_thresh = 0.5,
        .mode7 = 1, // 640x480 recommended when mode7 is on
        .render_many = 0,
        .method = 1,
        .max_episode_steps = 2000,
        .seed = seed,
        .env_idx = env_idx,
    };

    allocate(&env);
    env.client = make_client(&env);
    c_reset(&env);

    int frame = 0;
    SetTargetFPS(60);
    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            env.actions[0] = NOOP;
            if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env.actions[0] = LEFT;
            if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[0] = RIGHT;
        } else if (frame % 4 == 0) {
            // Frameskip applied outside the env for smoother rendering.
            env.actions[0] = bot_whisker(&env);
        }

        frame = (frame + 1) % 4;
        c_step(&env);
        c_render(&env);
    }

    free_allocated(&env);
    close_client(env.client);
    return 0;
}
