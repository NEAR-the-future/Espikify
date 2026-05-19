#pragma once
#include <stdint.h>

// Struct that defines a layer of neurons
typedef struct Neuron {
  // Neuron layer size
  int size;
  int type;
  // Inputs
  float *x;
  // Currents
  float *i;
  // Cell voltage
  float *v;

  // Cell threshold
  float *th;
  // Cell base threshold  (for ALIF)
  float *th_base;
  // Cell threshold state (for ALIF)
  float *t_s;

  // Cell spikes (0/1 as float)
  float *s;

  // Event-driven support:
  // spike_idx holds indices i where s[i] == 1 for the *current* timestep.
  // spike_k is the number of spikes in spike_idx (0..size).
  uint16_t *spike_idx;
  uint16_t spike_k;

  // Constants (weight) for threshold adaptation
  float *add_thresh;
  // Precomputed bound for t_s
  float *th_bound;
  // Constants for decay of current, voltage
  float *d_i, *d_v;
  // Constants for resetting voltage
  float v_rest;

  // Counter for spikes. For each forward step: s_count == (int)spike_k.
  int s_count;
} Neuron;

// Struct that holds the configuration of a layer of neurons
typedef struct NeuronConf {
  // Neuron layer size
  int const size;
  // Type
  int const type;
  // Constants for decay of voltage
  float const *d_i, *d_v;
  // Constant for threshold
  float const *th;
  // Constants for resetting voltage and threshold
  float const v_rest;
} NeuronConf;

// Build neuron
Neuron build_neuron(int const size);

// Init neuron
void init_neuron(Neuron *n);

// Reset neuron
void reset_neuron(Neuron *n);

// Load parameters for neuron from header file
void load_neuron_from_header(Neuron *n, NeuronConf const *conf);

// Forward
void forward_neuron(Neuron *n);

// update threshold based on threshold state and base threshold
void update_thresholds(Neuron *n);

// Free allocated memory for neuron
void free_neuron(Neuron *n);

// Print neuron parameters
void print_neuron(Neuron const *n);