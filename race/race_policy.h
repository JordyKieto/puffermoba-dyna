// Feed-forward policy inference in C, so a learned policy evaluates through
// the same headless path as the scripted bots.
//
// The alternative -- driving the env from Python -- would put a ~50us
// interpreter round trip in front of a ~1us env step and make "100 races per
// policy" a minutes-long job instead of a sub-second one. Online eval is the
// arbiter for every imagination result, so it has to stay cheap enough to run
// constantly.
//
// Weights come from racewm/export_policy.py. Input normalization is baked into
// the file: a policy that silently sees unnormalized inputs still produces
// plausible-looking actions, which is the worst kind of bug.

#ifndef RACE_POLICY_H
#define RACE_POLICY_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "race.h"

// A 4-wide dot product. The naive scalar loop made a learned policy 130x
// slower than a scripted bot in race_eval -- 27s vs 0.21s for 3,200 episodes,
// which is 160 GMAC of fully unvectorized work. Online eval is the arbiter for
// every result in this project, so it has to stay cheap enough to run
// constantly.
//
// The accumulation order is fixed, so this is still deterministic: the same
// input gives the same output on every run and every thread. It differs from
// the scalar order in the last bits, which is why the export check compares
// argmax agreement rather than exact logits.
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  #include <arm_neon.h>
  #define RACE_SIMD 1
#elif defined(__AVX__)
  #include <immintrin.h>
  #define RACE_SIMD 2
#else
  #define RACE_SIMD 0
#endif

static inline float race_dot(const float* a, const float* b, int n) {
#if RACE_SIMD == 1
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        acc = vfmaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
    }
    float sum = vaddvq_f32(acc);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
#elif RACE_SIMD == 2
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(a + i),
                                               _mm256_loadu_ps(b + i)));
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, acc);
    float sum = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
#else
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
#endif
}

#define RACE_POLICY_MAGIC 0x52504f31u /* "RPO1" */
#define RACE_POLICY_MAX_LAYERS 12
// Activations live on the caller's stack, so one Policy really is shared
// read-only across threads. They used to be scratch buffers inside the Policy
// struct, which made every worker clobber every other worker's forward pass:
// single-threaded the policy scored 6.4 return, on 8 threads 0.29. Nothing
// crashed and no test failed -- it just quietly produced a broken policy.
#define RACE_POLICY_MAX_WIDTH 2048

enum { RACE_ACT_SILU = 0, RACE_ACT_RELU = 1, RACE_ACT_TANH = 2 };

typedef struct {
    uint32_t magic, version, n_layers, obs_dim, act_dim, activation;
} PolicyHeader;

typedef struct {
    int in, out;
    float* w;   // row-major [out][in]
    float* b;
} PolicyLayer;

typedef struct {
    PolicyHeader h;
    float* mu;
    float* sd;
    PolicyLayer layers[RACE_POLICY_MAX_LAYERS];
    int widest;
} Policy;

static inline int policy_load(Policy* p, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "policy: cannot open %s\n", path); return 1; }
    if (fread(&p->h, sizeof(p->h), 1, f) != 1 || p->h.magic != RACE_POLICY_MAGIC) {
        fprintf(stderr, "policy: %s is not a race policy file\n", path);
        fclose(f); return 1;
    }
    if (p->h.obs_dim != RACE_OBS_SIZE) {
        fprintf(stderr, "policy: %s expects obs_dim %u, this build has %d\n",
                path, p->h.obs_dim, RACE_OBS_SIZE);
        fclose(f); return 1;
    }
    if (p->h.n_layers > RACE_POLICY_MAX_LAYERS) {
        fprintf(stderr, "policy: too many layers (%u)\n", p->h.n_layers);
        fclose(f); return 1;
    }

    p->mu = (float*)malloc(sizeof(float) * p->h.obs_dim);
    p->sd = (float*)malloc(sizeof(float) * p->h.obs_dim);
    if (fread(p->mu, sizeof(float), p->h.obs_dim, f) != p->h.obs_dim ||
        fread(p->sd, sizeof(float), p->h.obs_dim, f) != p->h.obs_dim) {
        fprintf(stderr, "policy: %s truncated in normalization block\n", path);
        fclose(f); return 1;
    }

    p->widest = p->h.obs_dim;
    for (uint32_t i = 0; i < p->h.n_layers; i++) {
        uint32_t dims[2];
        if (fread(dims, sizeof(uint32_t), 2, f) != 2) {
            fprintf(stderr, "policy: %s truncated at layer %u\n", path, i);
            fclose(f); return 1;
        }
        PolicyLayer* L = &p->layers[i];
        L->in = (int)dims[0]; L->out = (int)dims[1];
        L->w = (float*)malloc(sizeof(float) * (size_t)L->in * L->out);
        L->b = (float*)malloc(sizeof(float) * L->out);
        if (fread(L->w, sizeof(float), (size_t)L->in * L->out, f) != (size_t)L->in * L->out ||
            fread(L->b, sizeof(float), L->out, f) != (size_t)L->out) {
            fprintf(stderr, "policy: %s truncated in layer %u weights\n", path, i);
            fclose(f); return 1;
        }
        if (L->out > p->widest) p->widest = L->out;
    }
    fclose(f);

    if (p->widest > RACE_POLICY_MAX_WIDTH) {
        fprintf(stderr, "policy: layer width %d exceeds RACE_POLICY_MAX_WIDTH %d\n",
                p->widest, RACE_POLICY_MAX_WIDTH);
        return 1;
    }
    return 0;
}

static inline void policy_free(Policy* p) {
    for (uint32_t i = 0; i < p->h.n_layers; i++) {
        free(p->layers[i].w); free(p->layers[i].b);
    }
    free(p->mu); free(p->sd);
}

static inline float policy_activate(float x, uint32_t kind) {
    switch (kind) {
        case RACE_ACT_RELU: return x > 0 ? x : 0;
        case RACE_ACT_TANH: return tanhf(x);
        default:            return x / (1.0f + expf(-x));   // SiLU
    }
}

// Greedy argmax over logits. Deterministic, which keeps eval reproducible.
static inline int policy_act(const Policy* p, const float* obs) {
    float sa[RACE_POLICY_MAX_WIDTH];
    float sb[RACE_POLICY_MAX_WIDTH];
    float* a = sa;
    float* b = sb;
    for (uint32_t i = 0; i < p->h.obs_dim; i++) a[i] = (obs[i] - p->mu[i]) / p->sd[i];

    for (uint32_t l = 0; l < p->h.n_layers; l++) {
        const PolicyLayer* L = &p->layers[l];
        for (int o = 0; o < L->out; o++) {
            const float* row = L->w + (size_t)o * L->in;
            float acc = L->b[o] + race_dot(row, a, L->in);
            b[o] = (l + 1 < p->h.n_layers) ? policy_activate(acc, p->h.activation) : acc;
        }
        float* t = a; a = b; b = t;
    }

    int best = 0;
    for (uint32_t i = 1; i < p->h.act_dim; i++) if (a[i] > a[best]) best = (int)i;
    return best;
}

#endif
