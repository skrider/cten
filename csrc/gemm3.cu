#include "gemm.cuh"
#include "helper_cuda.h"
#include "tensor.cuh"

template <typename T, uint BLOCK_SIZE = 32, uint NUM_THREADS = 128>
__global__ void Gemm3(Tensor<T, 2> out,     // [rows, cols]
                      const Tensor<T, 2> a, // [rows, inner]
                      const Tensor<T, 2> b, // [inner, cols]
                      const Tensor<T, 2> c, // [rows, inner]
                      const T alpha, const T beta) {
  static __shared__ T b_tile[BLOCK_SIZE][BLOCK_SIZE];
  static __shared__ T a_t_tile[BLOCK_SIZE][BLOCK_SIZE];

  int wid = threadIdx.x / warpSize;
  int lane = threadIdx.x % warpSize;
  constexpr int N_WARPS = NUM_THREADS / WARP_SIZE;

  int tile_row = blockIdx.y * BLOCK_SIZE;
  int tile_col = blockIdx.x * BLOCK_SIZE;

  int c_i[2] = {BLOCK_SIZE * blockIdx.y + wid, BLOCK_SIZE * blockIdx.x + lane};

  // init accumulators
  for (int c_row_offset = 0; c_row_offset < BLOCK_SIZE;
       c_row_offset += N_WARPS) {
    for (int c_col_offset = 0; c_col_offset < BLOCK_SIZE;
         c_col_offset += warpSize) {
      int c_i[2] = {tile_row + c_row_offset + wid,
                    tile_col + c_col_offset + lane};
      for (int i = 0; i < BLOCK_SIZE; i++) {
        // use i as the first index to fetch entire line at once
        out(c_i) = beta * c(c_i);
      }
    }
  }

  for (int tile_inner = 0; tile_inner < a.shape[1]; tile_inner += BLOCK_SIZE) {

    // warp fetch row of b and transpose to shared mem
    for (int b_row_offset = 0; b_row_offset < BLOCK_SIZE;
         b_row_offset += N_WARPS) {
      for (int b_col_offset = 0; b_col_offset < BLOCK_SIZE;
           b_col_offset += warpSize) {
        // important to always have lane in the last dimension
        int b_i[2] = {tile_inner + b_row_offset + wid,
                      tile_col + b_col_offset + lane};
        b_tile[b_row_offset + wid][b_col_offset + lane] = b(b_i);
      }
    }

    // warp fetch row of a into shared mem
    for (int a_row_offset = 0; a_row_offset < BLOCK_SIZE;
         a_row_offset += N_WARPS) {
      for (int a_col_offset = 0; a_col_offset < BLOCK_SIZE;
           a_col_offset += warpSize) {
        int a_i[2] = {tile_row + a_row_offset + wid,
                      tile_inner + a_col_offset + lane};
        a_t_tile[a_col_offset + lane][a_row_offset + wid] = a(a_i);
      }
    }

    // wait until loading is finished
    __syncthreads();

    // compute inner product
    for (int c_row_offset = 0; c_row_offset < BLOCK_SIZE;
         c_row_offset += N_WARPS) {
      for (int c_col_offset = 0; c_col_offset < BLOCK_SIZE;
           c_col_offset += warpSize) {
        int c_i[2] = {tile_row + c_row_offset + wid,
                      tile_col + c_col_offset + lane};
        T acc = 0;
        // use i as the first index to fetch entire line at once
        for (int i = 0; i < BLOCK_SIZE; i++)
          acc +=
              a_t_tile[i][c_row_offset + wid] * b_tile[i][c_col_offset + lane];
        out(c_i) += alpha * acc;
      }
    }
  }
}

template <typename T>
void gemm3(Tensor<T, 2> out, Tensor<T, 2> a, Tensor<T, 2> b, Tensor<T, 2> c,
           T alpha, T beta) {
  out.fill(0);
  dim3 threadsPerBlock(WARP_SIZE * 4);
  dim3 numBlocks(c.shape[0] / WARP_SIZE, c.shape[1] / WARP_SIZE);
  Gemm3<T, 32, WARP_SIZE * 4>
      <<<numBlocks, threadsPerBlock>>>(out, a, b, c, alpha, beta);
}

template void gemm3(Tensor<int, 2> out, Tensor<int, 2> a, Tensor<int, 2> b,
                    Tensor<int, 2> c, int alpha, int beta);

template void gemm3(Tensor<float, 2> out, Tensor<float, 2> a,
                    Tensor<float, 2> b, Tensor<float, 2> c, float alpha,
                    float beta);
