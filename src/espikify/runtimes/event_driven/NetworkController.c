#include "NetworkController.h"
#include "Connection.h"
#include "Neuron.h"
#include "functional.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

NetworkController build_network(int const in_size, int const enc_size, int const hid_size, int const hid2_size, int const out_size) {
  NetworkController net;

  net.in_size = in_size;
  net.enc_size = enc_size;
  net.hid_size = hid_size;
  net.hid2_size = hid2_size;
  net.out_size = out_size;

  net.type = 1;
  net.tau_out = 0.9f;

  net.in = calloc(in_size, sizeof(*net.in));
  net.hid2_in = calloc(7, sizeof(*net.hid2_in));
  net.att_out = calloc(3, sizeof(*net.att_out));
  net.out = calloc(out_size, sizeof(*net.out));

  net.inenc = malloc(sizeof(*net.inenc));
  net.enc = malloc(sizeof(*net.enc));
  net.enchid = malloc(sizeof(*net.enchid));
  net.hidhid = malloc(sizeof(*net.hidhid));
  net.hid = malloc(sizeof(*net.hid));
  net.attout = malloc(sizeof(*net.attout));
  net.hidhid2 = malloc(sizeof(*net.hidhid2));
  net.hid2hid2 = malloc(sizeof(*net.hid2hid2));
  net.hid2 = malloc(sizeof(*net.hid2));
  net.hid2out = malloc(sizeof(*net.hid2out));

  *net.inenc = build_connection(in_size, enc_size);
  *net.enc = build_neuron(enc_size);
  *net.enchid = build_connection(enc_size, hid_size);
  *net.hidhid = build_connection(hid_size, hid_size);
  *net.hid = build_neuron(hid_size);
  *net.attout = build_connection(hid_size, 3);

  *net.hidhid2 = build_connection(7, hid2_size);
  *net.hid2hid2 = build_connection(hid2_size, hid2_size);
  *net.hid2 = build_neuron(hid2_size);
  *net.hid2out = build_connection(hid2_size, out_size);


  return net;
}

void init_network(NetworkController *net) {
  init_connection(net->inenc);
  init_neuron(net->enc);
  init_connection(net->enchid);
  init_connection(net->hidhid);
  init_neuron(net->hid);
  init_connection(net->attout);

  init_connection(net->hidhid2);
  init_connection(net->hid2hid2);
  init_neuron(net->hid2);
  init_connection(net->hid2out);

  reset_network(net);
}

void reset_network(NetworkController *net) {
  memset(net->in, 0, net->in_size * sizeof(float));
  memset(net->hid2_in, 0, 7 * sizeof(float));
  memset(net->att_out, 0, 3 * sizeof(float));
  memset(net->out, 0, net->out_size * sizeof(float));

  reset_connection(net->inenc);
  reset_neuron(net->enc);
  reset_connection(net->enchid);
  reset_connection(net->hidhid);
  reset_neuron(net->hid);
  reset_connection(net->attout);

  reset_connection(net->hidhid2);
  reset_connection(net->hid2hid2);
  reset_neuron(net->hid2);
  reset_connection(net->hid2out);
}

void load_network_from_header(NetworkController *net, NetworkControllerConf const *conf) {
  if ((net->in_size != conf->in_size) ||
      (net->enc_size != conf->enc_size) ||
      (net->hid_size != conf->hid_size) ||
      (net->hid2_size != conf->hid2_size) ||
      (net->out_size != conf->out_size)) {
    printf("Network shape mismatch vs NetworkControllerConf!\n");
    exit(1);
  }

  net->type = conf->type;
  net->tau_out = conf->tau_out;

  load_connection_from_header(net->inenc, conf->inenc);
  load_neuron_from_header(net->enc, conf->enc);
  load_connection_from_header(net->enchid, conf->enchid);
  load_connection_from_header(net->hidhid, conf->hidhid);
  load_neuron_from_header(net->hid, conf->hid);
  load_connection_from_header(net->attout, conf->attout);

  load_connection_from_header(net->hidhid2, conf->hidhid2);
  load_connection_from_header(net->hid2hid2, conf->hid2hid2);
  load_neuron_from_header(net->hid2, conf->hid2);
  load_connection_from_header(net->hid2out, conf->hid2out);
}

void set_network_input(NetworkController *net, float inputs[]) {
  for (int i = 0; i < 6; i++) net->in[i] = inputs[i];
  for (int i = 0; i < 7; i++) net->hid2_in[i] = inputs[i + 6];
}

// Forward network (W stored as W^T; spike connections are event-driven)
float* forward_network(NetworkController *net) {
  // ---------- attitude input scaling ----------
  float scaled_att_inputs[6];
  for (int i = 0; i < 6; i++) {
    if (i < 3) scaled_att_inputs[i] = net->in[i] * 0.01f;
    else if (i == 5) scaled_att_inputs[i] = net->in[i] * 0.3f;
    else scaled_att_inputs[i] = net->in[i];
  }

  // ---------- ATTITUDE NETWORK ----------
  // inenc is REAL-VALUED (dense, column-wise)
  forward_connection_fast(net->inenc, net->enc->x, scaled_att_inputs);
  forward_neuron(net->enc);

  // enc -> hid (event-driven)
  forward_connection_spikeidx_bin(net->enchid, net->hid->x,
                                 net->enc->spike_idx, (int)net->enc->spike_k);

  // hid recurrent uses previous hid spikes (event-driven)
  forward_connection_spikeidx_bin(net->hidhid, net->hid->x,
                                 net->hid->spike_idx, (int)net->hid->spike_k);

  forward_neuron(net->hid);

  // hid -> attitude out (event-driven)
  float raw_att_out[3] = {0};
  forward_connection_spikeidx_bin(net->attout, raw_att_out,
                                 net->hid->spike_idx, (int)net->hid->spike_k);

  net->att_out[0] = raw_att_out[0] / 0.05f;
  net->att_out[1] = raw_att_out[1] / 0.05f;
  net->att_out[2] = raw_att_out[2] / 30.0f;

  // ---------- CONTROL NETWORK ----------
  float scaled_control_inputs[7];
  scaled_control_inputs[0] = net->hid2_in[0] * 0.03f;
  scaled_control_inputs[1] = net->hid2_in[1] * 0.03f;
  scaled_control_inputs[2] = net->hid2_in[2] * 0.03f;
  scaled_control_inputs[3] = net->hid2_in[3] * 0.03f;

  scaled_control_inputs[4] = net->att_out[0] * 0.05f;
  scaled_control_inputs[5] = net->att_out[1] * 0.05f;
  scaled_control_inputs[6] = net->att_out[2] * 30.0f;

  float control_in[net->hid2_size];
  for (int i = 0; i < net->hid2_size; i++) control_in[i] = 0.0f;

  // hidhid2 is REAL-VALUED (dense, column-wise)
  forward_connection_fast(net->hidhid2, control_in, scaled_control_inputs);

  // add control_in to hid2->x
  for (int j = 0; j < net->hid2_size; j++) net->hid2->x[j] += control_in[j];

  // hid2 recurrent uses previous hid2 spikes (event-driven)
  forward_connection_spikeidx_bin(net->hid2hid2, net->hid2->x,
                                 net->hid2->spike_idx, (int)net->hid2->spike_k);

  forward_neuron(net->hid2);

  // ---------- OUTPUT ----------
  for (int i = 0; i < net->out_size; i++) net->out[i] = 0.0f;

  float raw_pitch_offset = 0.0f;
  forward_connection_spikeidx_bin(net->hid2out, &raw_pitch_offset,
                                 net->hid2->spike_idx, (int)net->hid2->spike_k);

  net->out[0] = raw_pitch_offset / 0.05f;
  return net->out;
}

float* get_attitude_outputs(NetworkController *net) { return net->att_out; }

void print_network(NetworkController const *net) {
  printf("Attitude outputs (pitch, roll, yaw_rate):\n");
  print_array_1d(3, net->att_out);
  printf("Pitch offset output:\n");
  print_array_1d(net->out_size, net->out);
}

void free_network(NetworkController *net) {
  free_connection(net->inenc);
  free_neuron(net->enc);
  free_connection(net->enchid);
  free_connection(net->hidhid);
  free_neuron(net->hid);
  free_connection(net->attout);

  free_connection(net->hidhid2);
  free_connection(net->hid2hid2);
  free_neuron(net->hid2);
  free_connection(net->hid2out);

  free(net->inenc);
  free(net->enc);
  free(net->enchid);
  free(net->hidhid);
  free(net->hid);
  free(net->attout);
  free(net->hidhid2);
  free(net->hid2hid2);
  free(net->hid2);
  free(net->hid2out);

  free(net->in);
  free(net->hid2_in);
  free(net->att_out);
  free(net->out);
}