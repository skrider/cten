#include "ops.cuh"
#include "tensor.cuh"
#include "helper_cuda.h"

template <typename scalar_t, unsigned DIMS>
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
    dim3 numBlocks(c.shape[0] / threadsPerBlock.x + 1, c.shape[1] / threadsPerBlock.y + 1);
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
        return;

    out({row, col}) = beta * c({row, col});
#pragma unroll
    for (int i = 0; i < a.shape[1]; i++)
    {
        out({row, col}) += alpha * a({row, i}) * b({i, col});
    }
}

template Tensor<int, 2> gemm(
    Tensor<int, 2> a,
    Tensor<int, 2> b,
    Tensor<int, 2> c,
    int alpha,
    int beta);
