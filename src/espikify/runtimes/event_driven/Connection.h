#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Connection {
  int pre, post;
  float *w;   // ROW-MAJOR W: w[i*pre + j] = W[i,j]
} Connection;

typedef struct ConnectionConf {
  int const pre, post;
  float const *w;   // ROW-MAJOR W
} ConnectionConf;

Connection build_connection(int const pre, int const post);
void init_connection(Connection *c);
void reset_connection(Connection *c);
void free_connection(Connection *c);

// memcpy row-major weights
void load_connection_from_header(Connection *c, ConnectionConf const *conf);

// Dense real-valued forward: x += W*s   (row-major dot)
void forward_connection_fast(const Connection *c, float x[], float const s[]);

// Event-driven binary spikes (row-wise): x += sum_t W[:, idx[t]]
void forward_connection_spikeidx_bin(const Connection *c, float x[],
                                    const uint16_t *idx, int k);

#ifdef __cplusplus
}
#endif