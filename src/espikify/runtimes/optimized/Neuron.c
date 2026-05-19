#include "Neuron.h"
#include "functional.h"

#include <math.h>    // fmaf
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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

Neuron build_neuron(int const size) {
  // Neuron struct
  Neuron n;

  // Set size
  n.size = size;

  // Set type (default is hard-reset)
  n.type = 1;
  n.x = calloc(size, sizeof(*n.x));
  n.i = calloc(size, sizeof(*n.i));
  n.v = calloc(size, sizeof(*n.v));
  n.th = calloc(size, sizeof(*n.th));
  n.th_base = calloc(size, sizeof(*n.th_base));
  n.add_thresh = calloc(size, sizeof(*n.add_thresh));
  n.th_bound = calloc(size, sizeof(*n.th_bound));
  n.t_s = calloc(size, sizeof(*n.t_s));
  n.s = calloc(size, sizeof(*n.s));
  n.d_i = calloc(size, sizeof(*n.d_i));
  n.d_v = calloc(size, sizeof(*n.d_v));
  // Reset constants
  n.v_rest = 0.0f;

  return n;
}

void init_neuron(Neuron *n) {
  for (int i = 0; i < n->size; i++) {
    n->d_i[i] = 0.8f;
    n->d_v[i] = 0.8f;

    n->x[i] = 0.0f;
    n->i[i] = 0.0f;
    n->v[i] = n->v_rest;

    n->th[i] = 1.0f;
    n->th_base[i] = 1.0f;
    n->t_s[i] = 0.0f;

    n->add_thresh[i] = 0.001f;
    n->th_bound[i] = n->th_base[i] / n->add_thresh[i];

    n->s[i] = 0.0f;
  }
  n->s_count = 0;
}

void reset_neuron(Neuron *n) {
  for (int i = 0; i < n->size; i++) {
    n->x[i] = 0.0f;
    n->i[i] = 0.0f;
    n->v[i] = n->v_rest;
    n->t_s[i] = 0.0f;
    n->s[i] = 0.0f;
  }
  n->s_count = 0;
}

void load_neuron_from_header(Neuron *n, NeuronConf const *conf) {
  if (n->size != conf->size) {
    printf("Neuron shape mismatch vs NeuronConf!\n");
    exit(1);
  }

  for (int i = 0; i < n->size; i++) {
    n->d_i[i] = conf->d_i[i];
    n->d_v[i] = conf->d_v[i];
    n->th[i] = conf->th[i];

    // th_base/add_thresh are application-specific; keep defaults unless you add them to NeuronConf.
    n->th_bound[i] = n->th_base[i] / n->add_thresh[i];
  }

  n->v_rest = conf->v_rest;
  n->type = conf->type;
}

void update_thresholds(Neuron *n) {
  for (int i = 0; i < n->size; i++) {
    n->th[i] = n->th_base[i] + n->add_thresh[i] * n->t_s[i];
  }
}

#ifdef ESP_PLATFORM
IRAM_ATTR
#endif
__attribute__((optimize("O3")))
void forward_neuron(Neuron *n)
{
  const int N = n->size;
  const float vrest = n->v_rest;

  float * RESTRICT v   = n->v;
  float * RESTRICT cur = n->i;
  float * RESTRICT x   = n->x;
  float * RESTRICT s   = n->s;

  const float * RESTRICT dv = n->d_v;
  const float * RESTRICT di = n->d_i;
  const float * RESTRICT th = n->th;

  int i0 = 0;

  for (; i0 <= N - 4; i0 += 4) {
    // ---- lane 0 ----
    float ci0 = fmaf(cur[i0 + 0], di[i0 + 0], x[i0 + 0]);
    cur[i0 + 0] = ci0;  x[i0 + 0] = 0.0f;
    float vi0 = fmaf((v[i0 + 0] - vrest), dv[i0 + 0], ci0);
    float sp0 = (vi0 > th[i0 + 0]) ? 1.0f : 0.0f;
    s[i0 + 0] = sp0;
    vi0 = (sp0 > 0.0f) ? vrest : vi0;
    v[i0 + 0] = vi0;

    // ---- lane 1 ----
    float ci1 = fmaf(cur[i0 + 1], di[i0 + 1], x[i0 + 1]);
    cur[i0 + 1] = ci1;  x[i0 + 1] = 0.0f;
    float vi1 = fmaf((v[i0 + 1] - vrest), dv[i0 + 1], ci1);
    float sp1 = (vi1 > th[i0 + 1]) ? 1.0f : 0.0f;
    s[i0 + 1] = sp1;
    vi1 = (sp1 > 0.0f) ? vrest : vi1;
    v[i0 + 1] = vi1;

    // ---- lane 2 ----
    float ci2 = fmaf(cur[i0 + 2], di[i0 + 2], x[i0 + 2]);
    cur[i0 + 2] = ci2;  x[i0 + 2] = 0.0f;
    float vi2 = fmaf((v[i0 + 2] - vrest), dv[i0 + 2], ci2);
    float sp2 = (vi2 > th[i0 + 2]) ? 1.0f : 0.0f;
    s[i0 + 2] = sp2;
    vi2 = (sp2 > 0.0f) ? vrest : vi2;
    v[i0 + 2] = vi2;

    // ---- lane 3 ----
    float ci3 = fmaf(cur[i0 + 3], di[i0 + 3], x[i0 + 3]);
    cur[i0 + 3] = ci3;  x[i0 + 3] = 0.0f;
    float vi3 = fmaf((v[i0 + 3] - vrest), dv[i0 + 3], ci3);
    float sp3 = (vi3 > th[i0 + 3]) ? 1.0f : 0.0f;
    s[i0 + 3] = sp3;
    vi3 = (sp3 > 0.0f) ? vrest : vi3;
    v[i0 + 3] = vi3;
  }

  // remainder
  for (; i0 < N; i0++) {
    float ci = fmaf(cur[i0], di[i0], x[i0]);
    cur[i0] = ci;
    x[i0] = 0.0f;

    float vi = fmaf((v[i0] - vrest), dv[i0], ci);
    float sp = (vi > th[i0]) ? 1.0f : 0.0f;
    s[i0] = sp;
    vi = (sp > 0.0f) ? vrest : vi;
    v[i0] = vi;
  }
}


void free_neuron(Neuron *n) {
  free(n->d_i);
  free(n->d_v);

  free(n->x);
  free(n->v);
  free(n->i);

  free(n->th);
  free(n->t_s);
  free(n->th_base);
  free(n->add_thresh);
  free(n->th_bound);

  free(n->s);

  n->d_i = n->d_v = NULL;
  n->x = n->v = n->i = NULL;
  n->th = n->t_s = n->th_base = NULL;
  n->add_thresh = n->th_bound = NULL;
  n->s = NULL;
}

void print_neuron(Neuron const *n) {
  printf("Input:\n");
  print_array_1d(n->size, n->x);
  printf("Current:\n");
  print_array_1d(n->size, n->i);
  printf("Voltage:\n");
  print_array_1d(n->size, n->v);
  printf("Threshold:\n");
  print_array_1d(n->size, n->th);
  printf("Threshold bounds:\n");
  print_array_1d(n->size, n->th_bound);
  printf("Spikes:\n");
  print_array_1d_bool(n->size, n->s);
  printf("Decay constants:\n");
  print_array_1d(n->size, n->d_i);
  print_array_1d(n->size, n->d_v);
  printf("Reset voltage: %.4f\n", n->v_rest);
  printf("Spike count: %d\n\n", n->s_count);
}
