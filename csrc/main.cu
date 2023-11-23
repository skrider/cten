#include "gemm.cuh"
#include "helper_cuda.h"
#include "rand.cuh"
#include "tensor.cuh"
#include "tensor_utils.cuh"
#include "utils.cuh"
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

#define ROW (1 << 12)
#define COL (1 << 12)
#define INNER (1 << 12)

#define WARMUP 2
#define N 10

int main(int argc, char **argv) {
  int seed = atoi(argv[0]);

  Tensor<float, 2> a({ROW, INNER});
  Tensor<float, 2> b({INNER, COL});
  Tensor<float, 2> c({ROW, COL});
  Tensor<float, 2> out({ROW, COL});
  Tensor<float, 2> out1({ROW, COL});
  float alpha = 1.0f;
  float beta = -1.0f;

  randn(a, 0.0, 1.0, 42);
  randn(b, 0.0, 1.0, 43);
  randn(c, 0.0, 1.0, 44);

  for (int i = 0; i < WARMUP; i++) {
    gemm2(out, a, b, c, alpha, beta);
    gemm3(out, a, b, c, alpha, beta);
  }

  for (int i = 0; i < N; i++) {
    gemm2(out, a, b, c, alpha, beta);
    checkCudaErrors(cudaDeviceSynchronize());
  }

  for (int i = 0; i < N; i++) {
    gemm3(out1, a, b, c, alpha, beta);
    checkCudaErrors(cudaDeviceSynchronize());
  }

  // TODO floating point error is significant - resolve with FMA
  cout << "pass: " << allclose(out1, out, (float)5e-3) << endl;

  Tensor<float, 2> d = (out - out1);
  cout << "max: " << d.max() << endl;
  auto i = d.argmax();
  cout << i[0] << ", " << i[1] << endl;

  cout << "min: " << d.min() << endl;
  i = d.argmin();
  cout << i[0] << ", " << i[1] << endl;
}
