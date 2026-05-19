#ifndef CONNECTION_H_
#define CONNECTION_H_

#ifdef __cplusplus
extern "C" {
#endif

// Connection between two layers (shape: post x pre).
typedef struct Connection {
  int pre;
  int post;
  float *w;  // Flattened weights: w[post * pre], row-major (post rows).
} Connection;

// Configuration container (typically generated in a header).
typedef struct ConnectionConf {
  int const pre;
  int const post;
  float const *w;
} ConnectionConf;

Connection build_connection(int pre, int post);
void init_connection(Connection *c);
void reset_connection(Connection *c);

void load_connection_from_header(Connection *c, ConnectionConf const *conf);
void free_connection(Connection *c);

// Spike-based (binary) input: adds weight when s[j] > 0.
void forward_connection(Connection *c, float x[], float const s[]);

// Real-valued input: x += W * s.
void forward_connection_real(Connection *c, float x[], float const s[]);

// Optimized real-valued path: x += W * s.
void forward_connection_fast(Connection const *c, float x[], float const s[]);

#ifdef __cplusplus
}
#endif

#endif  // CONNECTION_H_
