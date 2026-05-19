#pragma once

#include "Connection.h"
#include "Neuron.h"

// Struct that defines a network of two spiking layers
typedef struct NetworkController {
  // Input, encoded input, hidden and output layer sizes
  int in_size, enc_size, hid_size, hid2_size, out_size;
  // Type (1: LIF, 2: InputALIF, ...)
  int type;
  float *in, *hid2_in;  // hid2_in: 7 elements for control inputs
  float *out;
  float *att_out;  // Attitude network outputs (pitch, roll, yaw_rate)
  float tau_out;
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
  // Connection [7 inputs] -> hidden 2 (pitch command + gyro + attitude estimates)
  Connection *hidhid2;
  // Connection hidden 2 -> hidden 2
  Connection *hid2hid2;
  // Hidden 2 neurons
  Neuron *hid2;
  Connection *hid2out;
} NetworkController;

// Struct that holds the configuration of a two-layer network
// To be used when loading parameters from a header file
typedef struct NetworkControllerConf {
  // Input, encoded input, hidden and output layer sizes
  int const in_size, enc_size, hid_size, hid2_size, out_size;
  // Type
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
  // Connection [7 inputs] -> hidden 2
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

// Build network: calls build functions for children
NetworkController build_network(int const in_size, int const enc_size, int const hid_size, int const hid2_size, int const out_size);

// Init network: calls init functions for children
void init_network(NetworkController *net);

// Reset network: calls reset functions for children
void reset_network(NetworkController *net);

// Load parameters for network from header file and call load functions for
// children
void load_network_from_header(NetworkController *net, NetworkControllerConf const *conf);

// Free allocated memory for network and call free functions for children
void free_network(NetworkController *net);

// Print network parameters (for debugging purposes)
void print_network(NetworkController const *net);

// Set the inputs of the encoding layer
void set_network_input(NetworkController *net, float inputs[]);

// Forward network and call forward functions for children
// Encoding and decoding inside
float* forward_network(NetworkController *net);

// Get attitude network outputs
float* get_attitude_outputs(NetworkController *net);