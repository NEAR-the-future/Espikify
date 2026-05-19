#include "Connection.h"

#include <math.h>    // fmaf
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>  // calloc, free, rand, exit
#include <string.h>  // memcpy

#ifdef ESP_PLATFORM
  #include "esp_attr.h"  // IRAM_ATTR
#endif

#ifndef RESTRICT
  #if defined(__GNUC__)
    #define RESTRICT __restrict__
  #else
    #define RESTRICT
  #endif
#endif

Connection build_connection(int const pre, int const post) {
  Connection c;
  c.pre = pre;
  c.post = post;
  c.w = calloc(post * pre, sizeof(*c.w));
  return c;
}

void init_connection(Connection *c) {
  for (int i = 0; i < c->post; i++) {
    for (int j = 0; j < c->pre; j++) {
      c->w[i * c->pre + j] = rand() / (float)RAND_MAX;
    }
  }
}

void reset_connection(Connection *c) {}

void load_connection_from_header(Connection *c, ConnectionConf const *conf) {
  if ((c->pre != conf->pre) || (c->post != conf->post)) {
    printf("Connection shape mismatch vs ConnectionConf!\n");
    exit(1);
  }
  memcpy(c->w, conf->w, c->post * c->pre * sizeof(float));
}

void free_connection(Connection *c) {
  free(c->w);
}

void forward_connection(Connection *c, float x[], float const s[]) {
  // Loop over weights and multiply with spikes
  for (int i = 0; i < c->post; i++) {
    for (int j = 0; j < c->pre; j++) {
      if (s[j] > 0.0f) {  // Only add weight if spike is positive
        x[i] += c->w[i * c->pre + j];
      }
    }
  }
}

void forward_connection_real(Connection *c, float x[], float const s[]) {
  // Loop over weights and multiply with real-valued inputs
  for (int i = 0; i < c->post; i++) {
    for (int j = 0; j < c->pre; j++) {
      x[i] += c->w[i * c->pre + j] * s[j];
    }
  }
}

// ---- Fast dot products for common small pre sizes ----
static inline __attribute__((always_inline))
float dot_pre6(const float * RESTRICT w, const float * RESTRICT s) {
  float acc = 0.0f;
  acc = fmaf(w[0], s[0], acc);
  acc = fmaf(w[1], s[1], acc);
  acc = fmaf(w[2], s[2], acc);
  acc = fmaf(w[3], s[3], acc);
  acc = fmaf(w[4], s[4], acc);
  acc = fmaf(w[5], s[5], acc);
  return acc;
}

static inline __attribute__((always_inline))
float dot_pre7(const float * RESTRICT w, const float * RESTRICT s) {
  float acc = dot_pre6(w, s);
  acc = fmaf(w[6], s[6], acc);
  return acc;
}

static inline __attribute__((always_inline))
float dot_pre10(const float * RESTRICT w, const float * RESTRICT s) {
  float acc = 0.0f;
  // 10 = 8 + 2
  acc = fmaf(w[0], s[0], acc);
  acc = fmaf(w[1], s[1], acc);
  acc = fmaf(w[2], s[2], acc);
  acc = fmaf(w[3], s[3], acc);
  acc = fmaf(w[4], s[4], acc);
  acc = fmaf(w[5], s[5], acc);
  acc = fmaf(w[6], s[6], acc);
  acc = fmaf(w[7], s[7], acc);
  acc = fmaf(w[8], s[8], acc);
  acc = fmaf(w[9], s[9], acc);
  return acc;
}

// ---- Large-pre dot: 16-way unroll + 4 accumulators ----
static inline __attribute__((always_inline))
float dot_unroll16_4acc(const float * RESTRICT w, const float * RESTRICT s, int pre) {
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

  int j = 0;
  for (; j <= pre - 16; j += 16) {
    a0 = fmaf(w[j + 0],  s[j + 0],  a0);
    a1 = fmaf(w[j + 1],  s[j + 1],  a1);
    a2 = fmaf(w[j + 2],  s[j + 2],  a2);
    a3 = fmaf(w[j + 3],  s[j + 3],  a3);

    a0 = fmaf(w[j + 4],  s[j + 4],  a0);
    a1 = fmaf(w[j + 5],  s[j + 5],  a1);
    a2 = fmaf(w[j + 6],  s[j + 6],  a2);
    a3 = fmaf(w[j + 7],  s[j + 7],  a3);

    a0 = fmaf(w[j + 8],  s[j + 8],  a0);
    a1 = fmaf(w[j + 9],  s[j + 9],  a1);
    a2 = fmaf(w[j + 10], s[j + 10], a2);
    a3 = fmaf(w[j + 11], s[j + 11], a3);

    a0 = fmaf(w[j + 12], s[j + 12], a0);
    a1 = fmaf(w[j + 13], s[j + 13], a1);
    a2 = fmaf(w[j + 14], s[j + 14], a2);
    a3 = fmaf(w[j + 15], s[j + 15], a3);
  }

  float acc = (a0 + a1) + (a2 + a3);

  for (; j < pre; j++) {
    acc = fmaf(w[j], s[j], acc);
  }
  return acc;
}


#ifdef ESP_PLATFORM
IRAM_ATTR
#endif
__attribute__((optimize("O3")))
void forward_connection_fast(const Connection *c,
                             float * RESTRICT x,
                             const float * RESTRICT s) {
  const int pre  = c->pre;
  const int post = c->post;
  const float * RESTRICT w = c->w;

  for (int i = 0; i < post; i++) {
    const float * RESTRICT w_row = &w[(int32_t)i * pre];

    float acc;
    if (pre == 6) {
      acc = dot_pre6(w_row, s);
    } else if (pre == 7) {
      acc = dot_pre7(w_row, s);
    } else if (pre == 10) {
      acc = dot_pre10(w_row, s);
    } else if (pre >= 32) {
      acc = dot_unroll16_4acc(w_row, s, pre);
    } else {
      float tmp = 0.0f;
      for (int j = 0; j < pre; j++) tmp = fmaf(w_row[j], s[j], tmp);
      acc = tmp;
    }
    x[i] += acc;
  }
}