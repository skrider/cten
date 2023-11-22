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

#define WARMUP 1
#define N 1

int main(int argc, char **argv) {
  int seed = atoi(argv[0]);

  Tensor<int, 2> a({ROW, INNER});
  Tensor<int, 2> b({INNER, COL});
  Tensor<int, 2> c({ROW, COL});
  Tensor<int, 2> out({ROW, COL});
  Tensor<int, 2> out1({ROW, COL});
  int alpha = 1;
  int beta = 0;

  randu(a, 0, 100, 42);
  randu(b, 0, 100, 43);
  randu(c, 0, 100, 44);

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

  cout << "pass: " << allclose(out1, out, (int)1e-5) << endl;

  Tensor<int, 2> d = (out - out1);
  cout << "max: " << d.max() << endl;
  auto i = d.argmax();
  cout << i[0] << ", " << i[1] << endl;

  cout << "min: " << d.min() << endl;
  i = d.argmin();
  cout << i[0] << ", " << i[1] << endl;
}
