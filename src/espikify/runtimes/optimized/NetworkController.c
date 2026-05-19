#include "NetworkController.h"
#include "Connection.h"
#include "Neuron.h"
#include "functional.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Build network: calls build functions for children
NetworkController build_network(int const in_size, int const enc_size, int const hid_size, int const hid2_size, int const out_size) {
  NetworkController net;

  // Set sizes
  net.in_size = in_size;
  net.enc_size = enc_size;
  net.hid_size = hid_size;
  net.hid2_size = hid2_size;
  net.out_size = out_size;

  // Initialize type as LIF
  net.type = 1;

  // Initialize output variables
  net.tau_out = 0.9f;

  // Allocate memory
  net.in = calloc(in_size, sizeof(*net.in));
  net.hid2_in = calloc(7, sizeof(*net.hid2_in));  // 7 control inputs for pitch offset model
  net.att_out = calloc(3, sizeof(*net.att_out));  // 3 attitude outputs: pitch, roll, yaw_rate
  net.out = calloc(out_size, sizeof(*net.out));
  
  // Allocate memory for network components
  net.inenc = malloc(sizeof(*net.inenc));
  net.enc = malloc(sizeof(*net.enc));
  net.enchid = malloc(sizeof(*net.enchid));
  net.hidhid = malloc(sizeof(*net.hidhid));
  net.hid = malloc(sizeof(*net.hid));
  net.attout = malloc(sizeof(*net.attout));  // Attitude output connection
  net.hidhid2 = malloc(sizeof(*net.hidhid2));
  net.hid2hid2 = malloc(sizeof(*net.hid2hid2));
  net.hid2 = malloc(sizeof(*net.hid2));
  net.hid2out = malloc(sizeof(*net.hid2out));

  // Call build functions
  *net.inenc = build_connection(in_size, enc_size);
  *net.enc = build_neuron(enc_size);
  *net.enchid = build_connection(enc_size, hid_size);
  *net.hidhid = build_connection(hid_size, hid_size);
  *net.hid = build_neuron(hid_size);
  *net.attout = build_connection(hid_size, 3);  // 3 attitude outputs
  *net.hidhid2 = build_connection(7, hid2_size);  // 7 inputs for pitch offset model
  *net.hid2hid2 = build_connection(hid2_size, hid2_size);
  *net.hid2 = build_neuron(hid2_size);
  *net.hid2out = build_connection(hid2_size, out_size);

  return net;
}

// Init network: calls init functions for children
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
}

// Reset network: calls reset functions for children
void reset_network(NetworkController *net) {
  for (int i = 0; i < net->out_size; i++) {
    net->out[i] = 0.0f;
  }
  for (int i = 0; i < 3; i++) {
    net->att_out[i] = 0.0f;  // Reset attitude outputs
  }
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

// Load parameters for network from header file
void load_network_from_header(NetworkController *net, NetworkControllerConf const *conf) {
  // Check shapes
  if ((net->in_size != conf->in_size) ||
      (net->enc_size != conf->enc_size) ||
      (net->hid_size != conf->hid_size) ||
      (net->out_size != conf->out_size) ||
      (net->hid2_size != conf->hid2_size)) {
    printf("Network has a different shape than specified in the NetworkConf!\n");
    exit(1);
  }
  
  net->type = conf->type;

  // Load components
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
  
  net->tau_out = conf->tau_out;
}

// Set the inputs of the controller network
void set_network_input(NetworkController *net, float inputs[]) {
    // FWR inputs structure for pitch offset model:
    // [0-5]: IMU inputs for attitude network (gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z)
    // [6-12]: Control inputs for pitch offset network (Target_pitch_y, gyro.x, gyro.y, gyro.z, stateEstimate.pitch, stateEstimate.roll, stateEstimate.yaw.rate.milli)
    
    // Set attitude network inputs (IMU data)
    for (int i = 0; i < 6; i++) {
        net->in[i] = inputs[i];
    }

    // Set control network inputs - all 7 inputs for pitch offset model
    for (int i = 0; i < 7; i++) {
        net->hid2_in[i] = inputs[i + 6];
    }
}

float* forward_network(NetworkController *net) {
    // Apply input scaling for attitude network
    float scaled_att_inputs[6];
    for (int i = 0; i < 6; i++) {
        if (i < 3) scaled_att_inputs[i] = net->in[i] * 0.01f;  // gyro scaling: 0.01
        else if (i == 5) scaled_att_inputs[i] = net->in[i] * 0.3f;  // acc.z scaling: 0.3
        else scaled_att_inputs[i] = net->in[i];  // acc.x, acc.y scaling: 1.0
    }

    // === ATTITUDE NETWORK: IMU → Attitude Estimates ===
    forward_connection_fast(net->inenc, net->enc->x, scaled_att_inputs);
    forward_neuron(net->enc);

    forward_connection_fast(net->enchid, net->hid->x, net->enc->s);
    forward_connection_fast(net->hidhid, net->hid->x, net->hid->s);
    forward_neuron(net->hid);

    // Get raw attitude outputs
    float raw_att_out[3] = {0};
    forward_connection_fast(net->attout, raw_att_out, net->hid->s);
    
    // Apply attitude output scaling
    net->att_out[0] = raw_att_out[0] / 0.05f;      // pitch: divide by 0.05
    net->att_out[1] = raw_att_out[1] / 0.05f;      // roll: divide by 0.05
    net->att_out[2] = raw_att_out[2] / 30.0f;      // yaw_rate: divide by 30.0

    // === CONTROL NETWORK: Attitude Estimates → Pitch Offset ===
    // Apply scaling to control network inputs
    float scaled_control_inputs[7];
    
    // Input scaling
    scaled_control_inputs[0] = net->hid2_in[0] * 0.03f;  // Target_pitch_y
    scaled_control_inputs[1] = net->hid2_in[1] * 0.03f;   // gyro.x
    scaled_control_inputs[2] = net->hid2_in[2] * 0.03f;   // gyro.y
    scaled_control_inputs[3] = net->hid2_in[3] * 0.03f;   // gyro.z
    
    // Attitude estimates with their scaling (indices 4-6 of scaled_control_inputs)
    scaled_control_inputs[4] = net->att_out[0] * 0.05f;   // stateEstimate.pitch
    scaled_control_inputs[5] = net->att_out[1] * 0.05f;   // stateEstimate.roll
    scaled_control_inputs[6] = net->att_out[2] * 30.0f;   // stateEstimate.yaw.rate.milli

    float control_in[net->hid2_size];
    for (int i = 0; i < net->hid2_size; i++) {
        control_in[i] = 0.0f;
    }

    // Forward through hidhid2 connection
    forward_connection_fast(net->hidhid2, control_in, scaled_control_inputs);

    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < net->hid2_size; j++) {
            net->hid2->x[j] = net->hid2->x[j] + control_in[j];
        }
        forward_connection_fast(net->hid2hid2, net->hid2->x, net->hid2->s);
        forward_neuron(net->hid2);
    }

    for (int i = 0; i < net->out_size; i++) {
        net->out[i] = 0.0f;
    }
    
    // Get raw pitch offset output
    float raw_pitch_offset = 0.0f;
    forward_connection_fast(net->hid2out, &raw_pitch_offset, net->hid2->s);
    
    // Apply pitch offset output scaling
    net->out[0] = raw_pitch_offset / 0.05f;  // Pitch offset: divide by 0.05

    return net->out;
}

// Get attitude network outputs
float* get_attitude_outputs(NetworkController *net) {
  return net->att_out;
}

// Print network parameters
void print_network(NetworkController const *net) {
  printf("Attitude outputs (pitch, roll, yaw_rate):\n");
  print_array_1d(3, net->att_out);
  
  printf("Pitch offset output:\n");
  print_array_1d(net->out_size, net->out);
}

// Free allocated memory for network
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