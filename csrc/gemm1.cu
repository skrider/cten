#include "gemm.cuh"
#include "helper_cuda.h"
#include "tensor.cuh"

template <typename T>
void gemm1(Tensor<T, 2> out, Tensor<T, 2> a, Tensor<T, 2> b, Tensor<T, 2> c,
           T alpha, T beta) {
  out.fill(0);
  dim3 threadsPerBlock(32, 32);
  dim3 numBlocks(ROUND(c.shape[0], threadsPerBlock.x) / threadsPerBlock.x,
                 ROUND(c.shape[1], threadsPerBlock.y) / threadsPerBlock.y);
  Gemm1<<<numBlocks, threadsPerBlock>>>(out, a, b, c, alpha, beta);
}

// Gemm for a batch size of one at optimization level 1. Basic matrix multiply.
// Each thread computes one element of output.
template <typename T>
__global__ void Gemm1(Tensor<T, 2> out,     // [rows, cols]
                      const Tensor<T, 2> a, // [rows, inner]
                      const Tensor<T, 2> b, // [inner, cols]
                      const Tensor<T, 2> c, // [rows, inner]
                      const T alpha, const T beta) {
  // offset of the out element from beginning of the array
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  int c_i[2] = {row, col};
  out(c_i) = beta * c(c_i);
#pragma unroll
  for (int i = 0; i < a.shape[1]; i++) {
    int a_i[2] = {row, i};
    int b_i[2] = {i, col};
    out(c_i) += alpha * a(a_i) * b(b_i);
  }
}

template void gemm1(Tensor<int, 2> out, Tensor<int, 2> a, Tensor<int, 2> b,
                    Tensor<int, 2> c, int alpha, int beta);

template void gemm1(Tensor<float, 2> out, Tensor<float, 2> a,
                    Tensor<float, 2> b, Tensor<float, 2> c, float alpha,
                    float beta);
