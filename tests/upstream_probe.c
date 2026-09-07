// Validates the audit's determinism claims against UNMODIFIED upstream
// ocean/whisker_racer/whisker_racer.h.
#include <stdio.h>
#include <stdint.h>
#include "whisker_racer.h"

static uint64_t hash_track(WhiskerRacer* e) {
    uint64_t h = 14695981039346656037ULL;
    const uint8_t* p = (const uint8_t*)e->track.centerline;
    for (size_t i = 0; i < sizeof(Vector2) * e->track.total_points; i++) {
        h ^= p[i]; h *= 1099511628211ULL;
    }
    return h;
}

static void cfg(WhiskerRacer* e, unsigned rng, int i) {
    memset(e, 0, sizeof(*e));
    e->width = 1080; e->height = 720; e->track_width = 75;
    e->num_radial_sectors = 180; e->num_points = 16; e->bezier_resolution = 4;
    e->turn_pi_frac = 40; e->maxv = 5; e->w_ang = 0.777;
    e->max_whisker_length = 100; e->reward_yellow = 0.2; e->corner_thresh = 0.5;
    e->frameskip = 4; e->method = 2;
    e->rng = rng;   // from config/whisker_racer.ini: rng = 6
    e->i = i;       // key "i" is absent from every config -> dict_get asserts
}

int main(void) {
    printf("== upstream: track identity across envs ==\n");
    printf("my_init overwrites env->rng with the config constant (6), and 'i'\n"
           "is not in any config, so every env seeds srand(6 + i) identically.\n\n");
    for (int n = 0; n < 4; n++) {
        WhiskerRacer e;
        cfg(&e, 6, 1);            // what config/whisker_racer.ini actually yields
        srand(n);                 // what generic my_vec_init does first
        e.rng = n;                //   "  envs[n].rng = n
        cfg(&e, 6, 1);            // ...then my_init overwrites it again
        allocate(&e);
        printf("  env %d  track_hash=%016llx\n", n, (unsigned long long)hash_track(&e));
        free_allocated(&e);
    }

    printf("\n== upstream: episode starts couple through global rand() ==\n");
    WhiskerRacer a, b;
    cfg(&a, 6, 1); allocate(&a);
    cfg(&b, 6, 1); allocate(&b);
    srand(123);
    c_reset(&a); c_reset(&b);
    printf("  order A,B ->  A start=(%.2f, %.2f)  B start=(%.2f, %.2f)\n", a.px, a.py, b.px, b.py);
    srand(123);
    c_reset(&b); c_reset(&a);
    printf("  order B,A ->  A start=(%.2f, %.2f)  B start=(%.2f, %.2f)\n", a.px, a.py, b.px, b.py);
    printf("\n  Same seed, same envs, different reset order -> different states.\n");
    free_allocated(&a); free_allocated(&b);
    return 0;
}
