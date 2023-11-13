#include "ops.cuh"
#include "tensor.cuh"
#include "helper_cuda.h"

template <typename scalar_t, uint DIMS>
Tensor<scalar_t, DIMS> arange();

template <typename scalar_t>
Tensor<scalar_t, 2> gemm(
    Tensor<scalar_t, 2> a,
    Tensor<scalar_t, 2> b,
    Tensor<scalar_t, 2> c,
    scalar_t alpha,
    scalar_t beta)
{
    Tensor<scalar_t, 2> out(c.shape);
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(ROUND(c.shape[0], threadsPerBlock.x) / threadsPerBlock.x,
                   ROUND(c.shape[1], threadsPerBlock.y) / threadsPerBlock.y);
    GemmSingle1<<<numBlocks, threadsPerBlock>>>(out, a, b, c, alpha, beta);
    return out;
}

template <typename scalar_t>
void gemm(
    Tensor<scalar_t, 2> out,
    Tensor<scalar_t, 2> a,
    Tensor<scalar_t, 2> b,
    Tensor<scalar_t, 2> c,
    scalar_t alpha,
    scalar_t beta)
{
    out.fill(0);
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(ROUND(c.shape[0], threadsPerBlock.x) / threadsPerBlock.x,
                   ROUND(c.shape[1], threadsPerBlock.y) / threadsPerBlock.y);
    GemmSingle1<<<numBlocks, threadsPerBlock>>>(out, a, b, c, alpha, beta);
}

// Gemm for a batch size of one at optimization level 1. Basic matrix multiply.
// Each thread computes one element of output.
template <typename scalar_t>
__global__ void GemmSingle1(
    Tensor<scalar_t, 2> out,     // [rows, cols]
    const Tensor<scalar_t, 2> a, // [rows, inner]
    const Tensor<scalar_t, 2> b, // [inner, cols]
    const Tensor<scalar_t, 2> c, // [rows, inner]
    const scalar_t alpha,
    const scalar_t beta)
{
    // offset of the out element from beginning of the array
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int c_i[2] = {row, col};
    out(c_i) = beta * c(c_i);
#pragma unroll
    for (int i = 0; i < a.shape[1]; i++)
    {
        int a_i[2] = {row, i};
        int b_i[2] = {i, col};
        out(c_i) += alpha * a(a_i) * b(b_i);
    }
}

template <typename scalar_t>
void gemm2(
    Tensor<scalar_t, 2> out,
    Tensor<scalar_t, 2> a,
    Tensor<scalar_t, 2> b,
    Tensor<scalar_t, 2> c,
    scalar_t alpha,
    scalar_t beta)
{
    out.fill(0);
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(b.shape[1] / WARP_SIZE);
    GemmSingle2<<<numBlocks, threadsPerBlock>>>(out, a, b, c, alpha, beta);
}

/*
Blocked matrix multiplication. Each block is responsible for a 32x32 block of b.
Each block computes a column of Out.
*/
template <typename scalar_t>
__global__ void GemmSingle2(
    Tensor<scalar_t, 2> out,     // [rows, cols]
    const Tensor<scalar_t, 2> a, // [rows, inner]
    const Tensor<scalar_t, 2> b, // [inner, cols]
    const Tensor<scalar_t, 2> c, // [rows, inner]
    const scalar_t alpha,
    const scalar_t beta)
{
    // load b transposed into shared mem
    static __shared__ scalar_t b_t[WARP_SIZE][WARP_SIZE];

    // blocked row-wise iteration over b
    for (int ii = 0; ii < b.shape[0]; ii += WARP_SIZE)
    {
        // fetch block of b transposed into shared memory
        int b_i[2] = {ii + threadIdx.y,
                      blockDim.x * blockIdx.x + threadIdx.x};
        b_t[threadIdx.x][threadIdx.y] = b(b_i);

        // blocked row-wise iteration over a
        for (int jj = 0; jj < a.shape[0]; jj += WARP_SIZE)
        {
            int a_i[2] = {jj + threadIdx.y,
                          ii + threadIdx.x};
            scalar_t a_val = a(a_i);
            // warp owns a sub-row of a, and computes product with each sub-row of b,
            // which is fast because b is transposed in shared memory
#pragma unroll
            for (int i = 0; i < WARP_SIZE; i++)
            {
                scalar_t val = a_val * b_t[i][threadIdx.x];
                // reduce accross warp
                __syncthreads();
                for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
                    val += __shfl_down(val, offset);

                if (threadIdx.x == 0)
                {
                    int c_i[2] = {jj + threadIdx.y, blockDim.x * blockIdx.x + i};
                    out(c_i) += alpha * val;
                    if (ii == 0)
                        out(c_i) += beta * c(c_i);
                }
            }
        }
    }
}

template <typename scalar_t>
void gemm3(
    Tensor<scalar_t, 2> out,
    Tensor<scalar_t, 2> a,
    Tensor<scalar_t, 2> b,
    Tensor<scalar_t, 2> c,
    scalar_t alpha,
    scalar_t beta)
{
    out.fill(0);
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(c.shape[0] / WARP_SIZE, c.shape[1] / WARP_SIZE);
    GemmSingle3<<<numBlocks, threadsPerBlock>>>(out, a, b, c, alpha, beta);
}

template <typename scalar_t, uint BLOCK_SIZE = 32>
__global__ void GemmSingle3(
    Tensor<scalar_t, 2> out,     // [rows, cols]
    const Tensor<scalar_t, 2> a, // [rows, inner]
    const Tensor<scalar_t, 2> b, // [inner, cols]
    const Tensor<scalar_t, 2> c, // [rows, inner]
    const scalar_t alpha,
    const scalar_t beta)
{
    static __shared__ scalar_t b_t_block[BLOCK_SIZE][BLOCK_SIZE];
    static __shared__ scalar_t a_block[BLOCK_SIZE][BLOCK_SIZE];

    int c_i[2] = {blockDim.y * blockIdx.y + threadIdx.y,
                  blockDim.x * blockIdx.x + threadIdx.x};

    scalar_t acc = beta * c(c_i);

    for (int ii = 0; ii < a.shape[1]; ii += BLOCK_SIZE)
    {
        // fetch block of b
        int b_i[2] = {ii + threadIdx.y,
                      blockDim.x * blockIdx.x + threadIdx.x};
        b_t_block[threadIdx.x][threadIdx.y] = b(b_i);

        // fetch block of a
        int a_i[2] = {blockDim.y * blockIdx.y + threadIdx.y,
                      ii + threadIdx.x};
        a_block[threadIdx.x][threadIdx.y] = a(a_i);

        // init accumulator
        __syncthreads();

#pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i++)
            // use i as the first index to fetch entire line at once
            acc += a_block[i][threadIdx.y] * b_t_block[i][threadIdx.x];
    }
    out(c_i) = alpha * acc;
}

template Tensor<int, 2> gemm(
    Tensor<int, 2> a,
    Tensor<int, 2> b,
    Tensor<int, 2> c,
    int alpha,
    int beta);

template void gemm(
    Tensor<int, 2> out,
    Tensor<int, 2> a,
    Tensor<int, 2> b,
    Tensor<int, 2> c,
    int alpha,
    int beta);

template void gemm2(
    Tensor<int, 2> out,
    Tensor<int, 2> a,
    Tensor<int, 2> b,
    Tensor<int, 2> c,
    int alpha,
    int beta);

template void gemm3(
    Tensor<int, 2> out,
    Tensor<int, 2> a,
    Tensor<int, 2> b,
    Tensor<int, 2> c,
    int alpha,
    int beta);
