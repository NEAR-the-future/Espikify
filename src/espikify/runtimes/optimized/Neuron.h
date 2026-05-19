#ifndef NEURON_H_
#define NEURON_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Neuron {
  int size;
  int type;  // 1: hard-reset, 2: soft-reset

  float *x;       // inputs
  float *i;       // currents
  float *v;       // membrane voltage
  float *th;      // threshold

  // Optional ALIF fields (allocated even if unused)
  float *th_base;
  float *t_s;
  float *s;        // spikes (0/1)
  float *add_thresh;
  float *th_bound;

  float *d_i;      // current decay
  float *d_v;      // voltage decay
  float v_rest;    // reset voltage

  int s_count;     // spike counter (not incremented in forward_neuron)
} Neuron;

typedef struct NeuronConf {
  int const size;
  int const type;
  float const *d_i;
  float const *d_v;
  float const *th;
  float const v_rest;
} NeuronConf;

Neuron build_neuron(int size);
void init_neuron(Neuron *n);
void reset_neuron(Neuron *n);

void load_neuron_from_header(Neuron *n, NeuronConf const *conf);

void forward_neuron(Neuron *n);
void update_thresholds(Neuron *n);

void free_neuron(Neuron *n);
void print_neuron(Neuron const *n);

#ifdef __cplusplus
}
#endif

#endif  // NEURON_H_
