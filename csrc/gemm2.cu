#include "gemm.cuh"
#include "helper_cuda.h"
#include "tensor.cuh"

template <typename T>
void gemm2(Tensor<T, 2> out, Tensor<T, 2> a, Tensor<T, 2> b, Tensor<T, 2> c,
           T alpha, T beta) {
  out.fill(0);
  dim3 threadsPerBlock(32, 32);
  dim3 numBlocks(c.shape[0] / WARP_SIZE, c.shape[1] / WARP_SIZE);
  Gemm2<<<numBlocks, threadsPerBlock>>>(out, a, b, c, alpha, beta);
}

template <typename T, uint BLOCK_SIZE = 32>
__global__ void Gemm2(Tensor<T, 2> out,     // [rows, cols]
                      const Tensor<T, 2> a, // [rows, inner]
                      const Tensor<T, 2> b, // [inner, cols]
                      const Tensor<T, 2> c, // [rows, inner]
                      const T alpha, const T beta) {
  static __shared__ T b_block[BLOCK_SIZE][BLOCK_SIZE];
  static __shared__ T a_t_block[BLOCK_SIZE][BLOCK_SIZE];

  int c_i[2] = {blockDim.y * blockIdx.y + threadIdx.y,
                blockDim.x * blockIdx.x + threadIdx.x};

  T acc = beta * c(c_i);

  for (int ii = 0; ii < a.shape[1]; ii += BLOCK_SIZE) {
    // fetch block of b
    int b_i[2] = {ii + threadIdx.y, blockDim.x * blockIdx.x + threadIdx.x};
    b_block[threadIdx.y][threadIdx.x] = b(b_i);

    // fetch block of a
    int a_i[2] = {blockDim.y * blockIdx.y + threadIdx.y, ii + threadIdx.x};
    a_t_block[threadIdx.x][threadIdx.y] = a(a_i);

    // init accumulator
    __syncthreads();

#pragma unroll
    for (int i = 0; i < BLOCK_SIZE; i++)
      // use i as the first index to fetch entire line at once
      acc += a_t_block[i][threadIdx.y] * b_block[i][threadIdx.x];
  }
  out(c_i) = alpha * acc;
}

template void gemm2(Tensor<int, 2> out, Tensor<int, 2> a, Tensor<int, 2> b,
                    Tensor<int, 2> c, int alpha, int beta);
