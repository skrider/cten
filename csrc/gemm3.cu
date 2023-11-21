#include "gemm.cuh"
#include "helper_cuda.h"
#include "tensor.cuh"

template <typename T, uint BLOCK_SIZE = 32>
__global__ void Gemm3(Tensor<T, 2> out,     // [rows, cols]
                      const Tensor<T, 2> a, // [rows, inner]
                      const Tensor<T, 2> b, // [inner, cols]
                      const Tensor<T, 2> c, // [rows, inner]
                      const T alpha, const T beta) {
  static __shared__ T b_block[BLOCK_SIZE][BLOCK_SIZE];
  static __shared__ T a_t_block[BLOCK_SIZE][BLOCK_SIZE];

  int wid = threadIdx.x / warpSize;
  int lane = threadIdx.x % warpSize;

  int c_i[2] = {BLOCK_SIZE * blockIdx.y + wid, BLOCK_SIZE * blockIdx.x + lane};

  T acc = beta * c(c_i);

  for (int ii = 0; ii < a.shape[1]; ii += BLOCK_SIZE) {
    // fetch block of b
    int b_i[2] = {ii + wid, BLOCK_SIZE * blockIdx.x + lane};
    b_block[wid][lane] = b(b_i);

    // fetch block of a
    int a_i[2] = {BLOCK_SIZE * blockIdx.y + wid, ii + lane};
    a_t_block[lane][wid] = a(a_i);

    // init accumulator
    __syncthreads();

#pragma unroll
    for (int i = 0; i < BLOCK_SIZE; i++)
      // use i as the first index to fetch entire line at once
      acc += a_t_block[i][wid] * b_block[i][lane];
  }
  out(c_i) = alpha * acc;
}

template <typename T>
void gemm3(Tensor<T, 2> out, Tensor<T, 2> a, Tensor<T, 2> b, Tensor<T, 2> c,
           T alpha, T beta) {
  out.fill(0);
  dim3 threadsPerBlock(32 * 32);
  dim3 numBlocks(c.shape[0] / WARP_SIZE, c.shape[1] / WARP_SIZE);
  Gemm3<T, 32><<<numBlocks, threadsPerBlock>>>(out, a, b, c, alpha, beta);
}

template void gemm3(Tensor<int, 2> out, Tensor<int, 2> a, Tensor<int, 2> b,
                    Tensor<int, 2> c, int alpha, int beta);

template void gemm3(Tensor<float, 2> out, Tensor<float, 2> a,
                    Tensor<float, 2> b, Tensor<float, 2> c, float alpha,
                    float beta);
