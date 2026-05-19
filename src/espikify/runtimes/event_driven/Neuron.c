#include "Neuron.h"
#include "functional.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#if !defined(IRAM_ATTR) && (defined(ESP_PLATFORM) || defined(ARDUINO_ARCH_ESP32))
  #include <esp_attr.h>
#endif

#ifndef IRAM_ATTR
  #define IRAM_ATTR
#endif

#ifndef RESTRICT
  #if defined(__GNUC__)
    #define RESTRICT __restrict__
  #else
    #define RESTRICT
  #endif
#endif

// Build neuron
Neuron build_neuron(int const size) {
  Neuron n;

  n.size = size;
  n.type = 1;

  // State arrays
  n.x = calloc(size, sizeof(*n.x));
  n.i = calloc(size, sizeof(*n.i));
  n.v = calloc(size, sizeof(*n.v));
  n.th = calloc(size, sizeof(*n.th));
  n.th_base = calloc(size, sizeof(*n.th_base));
  n.add_thresh = calloc(size, sizeof(*n.add_thresh));
  n.th_bound = calloc(size, sizeof(*n.th_bound));
  n.t_s = calloc(size, sizeof(*n.t_s));
  n.s = calloc(size, sizeof(*n.s));

  // Event-driven support
  n.spike_idx = calloc(size, sizeof(*n.spike_idx));
  n.spike_k = 0;

  // Decay constants
  n.d_i = calloc(size, sizeof(*n.d_i));
  n.d_v = calloc(size, sizeof(*n.d_v));

  n.v_rest = 0.0f;

  // Legacy spike counter
  n.s_count = 0;

  return n;
}

// Init neuron
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
  n->spike_k = 0;
  n->s_count = 0;
}

// Reset neuron
void reset_neuron(Neuron *n) {
  for (int i = 0; i < n->size; i++) {
    n->x[i] = 0.0f;
    n->i[i] = 0.0f;
    n->v[i] = n->v_rest;
    n->t_s[i] = 0.0f;
    n->s[i] = 0.0f;
  }
  n->spike_k = 0;
  n->s_count = 0;
}

// Load parameters for neuron from header file
void load_neuron_from_header(Neuron *n, NeuronConf const *conf) {
  if (n->size != conf->size) {
    printf("Neuron has a different shape than specified in the NeuronConf!\n");
    exit(1);
  }
  for (int i = 0; i < n->size; i++) {
    n->d_i[i] = conf->d_i[i];
    n->d_v[i] = conf->d_v[i];
    n->th[i] = conf->th[i];
    n->th_bound[i] = n->th_base[i] / n->add_thresh[i];
  }
  n->v_rest = conf->v_rest;
  n->type = conf->type;
}

// update thresholds
void update_thresholds(Neuron *n) {
  for (int i = 0; i < n->size; i++) {
    n->th[i] = n->th_base[i] + n->add_thresh[i] * n->t_s[i];
  }
}

// Forward: updates current/voltage, generates spikes, and populates spike_idx/spike_k
IRAM_ATTR __attribute__((optimize("O3")))
void forward_neuron(Neuron *n) {
  const int N = n->size;

  // Reset per-step counters
  int k = 0;
  n->spike_k = 0;
  n->s_count = 0;

  // Fast path: type 1 only (hard reset)
  if (n->type == 1) {
    const float vrest = n->v_rest;

    float * RESTRICT v   = n->v;
    float * RESTRICT cur = n->i;
    float * RESTRICT x   = n->x;
    float * RESTRICT s   = n->s;

    const float * RESTRICT dv = n->d_v;
    const float * RESTRICT di = n->d_i;
    const float * RESTRICT th = n->th;

    uint16_t * RESTRICT idx = n->spike_idx;

    int i0 = 0;

    // 4-way unroll
    for (; i0 <= N - 4; i0 += 4) {
      // lane 0
      float ci0 = fmaf(cur[i0 + 0], di[i0 + 0], x[i0 + 0]);
      cur[i0 + 0] = ci0; x[i0 + 0] = 0.0f;
      float vi0 = fmaf((v[i0 + 0] - vrest), dv[i0 + 0], ci0);
      int spi0 = (vi0 > th[i0 + 0]);           // 0 or 1
      float sp0 = (float)spi0;
      s[i0 + 0] = sp0;
      vi0 = fmaf(sp0, (vrest - vi0), vi0);     // branchless hard reset
      v[i0 + 0] = vi0;
      idx[k] = (uint16_t)(i0 + 0); k += spi0;  // branchless index push

      // lane 1
      float ci1 = fmaf(cur[i0 + 1], di[i0 + 1], x[i0 + 1]);
      cur[i0 + 1] = ci1; x[i0 + 1] = 0.0f;
      float vi1 = fmaf((v[i0 + 1] - vrest), dv[i0 + 1], ci1);
      int spi1 = (vi1 > th[i0 + 1]);
      float sp1 = (float)spi1;
      s[i0 + 1] = sp1;
      vi1 = fmaf(sp1, (vrest - vi1), vi1);
      v[i0 + 1] = vi1;
      idx[k] = (uint16_t)(i0 + 1); k += spi1;

      // lane 2
      float ci2 = fmaf(cur[i0 + 2], di[i0 + 2], x[i0 + 2]);
      cur[i0 + 2] = ci2; x[i0 + 2] = 0.0f;
      float vi2 = fmaf((v[i0 + 2] - vrest), dv[i0 + 2], ci2);
      int spi2 = (vi2 > th[i0 + 2]);
      float sp2 = (float)spi2;
      s[i0 + 2] = sp2;
      vi2 = fmaf(sp2, (vrest - vi2), vi2);
      v[i0 + 2] = vi2;
      idx[k] = (uint16_t)(i0 + 2); k += spi2;

      // lane 3
      float ci3 = fmaf(cur[i0 + 3], di[i0 + 3], x[i0 + 3]);
      cur[i0 + 3] = ci3; x[i0 + 3] = 0.0f;
      float vi3 = fmaf((v[i0 + 3] - vrest), dv[i0 + 3], ci3);
      int spi3 = (vi3 > th[i0 + 3]);
      float sp3 = (float)spi3;
      s[i0 + 3] = sp3;
      vi3 = fmaf(sp3, (vrest - vi3), vi3);
      v[i0 + 3] = vi3;
      idx[k] = (uint16_t)(i0 + 3); k += spi3;
    }

    // remainder
    for (; i0 < N; i0++) {
      float ci = fmaf(cur[i0], di[i0], x[i0]);
      cur[i0] = ci; x[i0] = 0.0f;

      float vi = fmaf((v[i0] - vrest), dv[i0], ci);
      int spi = (vi > th[i0]);
      float sp = (float)spi;
      s[i0] = sp;

      vi = fmaf(sp, (vrest - vi), vi);
      v[i0] = vi;

      idx[k] = (uint16_t)i0; k += spi;
    }

    n->spike_k = (uint16_t)k;
    n->s_count = k;
    return;
  }

  // Generic fallback (type != 1): original behavior + fill spike_idx/k.
  for (int i = 0; i < N; i++) {
    n->i[i] = n->i[i] * n->d_i[i] + n->x[i];
    n->x[i] = 0.0f;

    n->v[i] = (n->v[i] - n->v_rest) * n->d_v[i] + n->i[i];

    if (n->type == 2) {
      if (n->v[i] < 0.0f) n->v[i] = 0.0f;
    }

    if (n->v[i] > n->th[i]) {
      n->s[i] = 1.0f;
      n->spike_idx[k++] = (uint16_t)i;
      if (n->type == 1) n->v[i] = n->v_rest;
      else if (n->type == 2) n->v[i] = n->v[i] - n->th[i];
    } else {
      n->s[i] = 0.0f;
    }
  }

  n->spike_k = (uint16_t)k;
  n->s_count = k;
}

// Free allocated memory for neuron
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
  free(n->spike_idx);
}

// Print neuron parameters
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
  printf("Reset constant voltage: %.4f\n\n", n->v_rest);
  printf("Spike count (legacy s_count): %d\n", n->s_count);
  printf("Spike count (spike_k): %u\n", (unsigned)n->spike_k);
  printf("\n");
}