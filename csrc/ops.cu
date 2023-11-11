#include "ops.cuh"
#include "tensor.cuh"
#include "helper_cuda.h"

template <typename scalar_t>
Tensor<scalar_t, 2> gemm(
    Tensor<scalar_t, 2> a,
    Tensor<scalar_t, 2> b,
    Tensor<scalar_t, 2> c,
    scalar_t alpha,
    scalar_t beta)
{
    Tensor<scalar_t, 2> out(c.shape());
}

// Gemm for a batch size of one at optimization level 1. Basic matrix multiply.
// Each thread computes one element of output.
template <typename scalar_t>
__global__ void GemmSingle1(
    scalar_t *out,                  // [rows, cols]
    const scalar_t *__restrict__ a, // [rows, inner]
    const scalar_t *__restrict__ b, // [inner, cols]
    const scalar_t *__restrict__ c, // [rows, cols]
    const scalar_t alpha,
    const scalar_t beta,
    int rows,
    int cols,
    int inner)
{
    // offset of the out element from beginning of the array
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= rows || col >= cols)
        return;

    int offset = row * cols + col;

    out[offset] = beta * c[offset];
#pragma unroll
    for (int i = 0; i < inner; i++)
    {
        out[offset] += a[row * inner + i] * b[i * inner + col];
    }
}
