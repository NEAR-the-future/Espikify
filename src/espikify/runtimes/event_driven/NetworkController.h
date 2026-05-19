#pragma once

#include "Connection.h"
#include "Neuron.h"

// Struct that defines a network of two spiking layers
typedef struct NetworkController {
  int in_size, enc_size, hid_size, hid2_size, out_size;
  int type;
  float *in, *hid2_in;
  float *out;
  float *att_out;  // state estimation network outputs
  float tau_out;
  // Per-step spiking statistics
  int enc_spike_count, hid_spike_count, hid2_spike_count;
  float enc_event_rate, hid_event_rate, hid2_event_rate;
  float enc_sparsity, hid_sparsity, hid2_sparsity;
  // Active presynaptic events used by event-driven connections
  int enchid_active_pre, hidhid_active_pre, attout_active_pre;
  int hid2hid2_active_pre, hid2out_active_pre;
  // Encoding input -> encoding layer
  Connection *inenc;
  // Encoding LIF layer
  Neuron *enc;
  // Connection encoding -> hidden
  Connection *enchid;
  // Recurrent connection hidden -> hidden
  Connection *hidhid;
  // Hidden neurons
  Neuron *hid;
  // Connection hidden -> attitude output
  Connection *attout;
  // Connection -> hidden 2
  Connection *hidhid2;
  // Connection hidden 2 -> hidden 2
  Connection *hid2hid2;
  // Hidden 2 neurons
  Neuron *hid2;
  Connection *hid2out;
} NetworkController;

// Struct that holds the configuration of a two-layer network
typedef struct NetworkControllerConf {
  int const in_size, enc_size, hid_size, hid2_size, out_size;
  int const type;
  // Encoding input -> encoding layer
  ConnectionConf const *inenc;
  // Encoding LIF layer
  NeuronConf const *enc;
  // Connection encoding -> hidden
  ConnectionConf const *enchid;
  // Recurrent connection hidden -> hidden
  ConnectionConf const *hidhid;
  // Hidden neurons
  NeuronConf const *hid;
  // Connection hidden -> attitude output
  ConnectionConf const *attout;
  // Connection -> hidden 2
  ConnectionConf const *hidhid2;
  // Recurrent connection hidden 2 -> hidden 2
  ConnectionConf const *hid2hid2;
  // Hidden 2 neurons
  NeuronConf const *hid2;
  // Connection hidden -> output
  ConnectionConf const *hid2out;
  // Output decay
  const float tau_out;
} NetworkControllerConf;

// Build network
NetworkController build_network(int const in_size, int const enc_size, int const hid_size, int const hid2_size, int const out_size);

// Init network
void init_network(NetworkController *net);

// Reset network
void reset_network(NetworkController *net);

// Load parameters for network from header file
void load_network_from_header(NetworkController *net, NetworkControllerConf const *conf);

// Free allocated memory for network
void free_network(NetworkController *net);

// Print network parameters
void print_network(NetworkController const *net);

// Set the inputs of the encoding layer
void set_network_input(NetworkController *net, float inputs[]);

// Forward network
float* forward_network(NetworkController *net);

// Get attitude network outputs
float* get_attitude_outputs(NetworkController *net);
