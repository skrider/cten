#include "tensor.cuh"
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <array>

// template <typename scalar_t, int DIMS, int NEW_DIMS>
// Tensor<scalar_t, NEW_DIMS> Tensor<scalar_t, DIMS>::reshape(std::array<unsigned, NEW_DIMS> new_shape)
// {
//     Tensor<scalar_t, DIMS> ret(new_shape);
//     return ret;
// }

template <typename scalar_t, uint DIMS>
__global__ void Fill(Tensor<scalar_t, DIMS> t, scalar_t value)
{
    t(blockIdx.x * blockDim.x + threadIdx.x) = value;
}

template <typename scalar_t, uint DIMS>
void Tensor<scalar_t, DIMS>::fill(scalar_t value)
{
    dim3 threadsPerBlock(32);
    dim3 numBlocks(phys_size / threadsPerBlock.x);
    Fill<<<numBlocks, threadsPerBlock>>>(*this, value);
}

template class Tensor<int, 2>;
