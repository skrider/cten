#include "helper_cuda.h"
#include "tensor.cuh"
#include <array>
#include <cuda_runtime.h>

// template <typename scalar_t, int DIMS, int NEW_DIMS>
// Tensor<scalar_t, NEW_DIMS> Tensor<scalar_t,
// DIMS>::reshape(std::array<unsigned, NEW_DIMS> new_shape)
// {
//     Tensor<scalar_t, DIMS> ret(new_shape);
//     return ret;
// }

template <typename T, uint D> __global__ void Fill(Tensor<T, D> t, T value) {
  t(blockIdx.x * blockDim.x + threadIdx.x) = value;
}

template <typename T, uint D> void Tensor<T, D>::fill(T value) {
  dim3 threadsPerBlock(32);
  dim3 numBlocks(size / threadsPerBlock.x);
  Fill<<<numBlocks, threadsPerBlock>>>(*this, value);
}

template class Tensor<int, 2>;
