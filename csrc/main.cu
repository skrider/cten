#include "gemm.cuh"
#include "helper_cuda.h"
#include "rand.cuh"
#include "tensor.cuh"
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
  Tensor<float, 2> a({ROW, INNER});
  Tensor<float, 2> b({INNER, COL});
  Tensor<float, 2> c({ROW, COL});
  Tensor<float, 2> out({ROW, COL});
  float alpha = 1.0f;
  float beta = -1.0f;

  randn(a, 0.0, 1.0, 42);
  randn(b, 0.0, 1.0, 42);
  randn(c, 0.0, 1.0, 42);

  for (int i = 0; i < WARMUP; i++) {
    gemm1(out, a, b, c, alpha, beta);
    gemm2(out, a, b, c, alpha, beta);
  }

  for (int i = 0; i < N; i++) {
    gemm1(out, a, b, c, alpha, beta);
    checkCudaErrors(cudaDeviceSynchronize());
  }

  cout << "element:      " << out.get({31, 30}) << endl;

  for (int i = 0; i < N; i++) {
    gemm2(out, a, b, c, alpha, beta);
    checkCudaErrors(cudaDeviceSynchronize());
  }

  cout << "element:      " << out.get({31, 31}) << endl;
}
