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
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= c.shape[0] || col >= c.shape[1])
    {
        return;
    }

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

template Tensor<int, 2> gemm(
    Tensor<int, 2> a,
    Tensor<int, 2> b,
    Tensor<int, 2> c,
    int alpha,
    int beta);
