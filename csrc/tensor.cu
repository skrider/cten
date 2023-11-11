#include "tensor.cuh"
#include <cuda_runtime.h>
#include "helper_cuda.h"

template <typename scalar_t, int DIMS>
Tensor<scalar_t, DIMS> Tensor<scalar_t, DIMS>::reshape(unsigned[DIMS] new_shape)
{
    Tensor<scalar_t, DIMS> ret(new_shape);
    return ret;
}
