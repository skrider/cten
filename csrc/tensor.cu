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
