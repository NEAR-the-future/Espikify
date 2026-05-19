#pragma once

// Print 1D array of floats (as floats)
void print_array_1d(int size, float const *x);

// Print 1D array of floats (as 0/1 integers)
void print_array_1d_bool(int size, float const *x);

// Print 2D array of floats stored as a flattened (row-major) 1D array
void print_array_2d(int rows, int cols, float const *x);

// Read a sequence file
void read_sequence(char const *filename, float **input_container);
