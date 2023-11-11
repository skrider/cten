#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <array>

#ifndef _TENSOR
#define _TENSOR

#define OOB_MSG "error tensor index out of bounds"

#define WARP_SIZE 32
#define WARP_ROUND(size) (((size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE)

template <typename scalar_t, unsigned DIMS>
class Tensor
{
private:
    // shape with rows rounded up to nearest warp dim
    const std::array<unsigned, DIMS> phys_shape;
    const unsigned phys_size;
    const std::array<unsigned, DIMS> stride;
    scalar_t *data;

    __device__ __host__ inline unsigned offset(std::array<int, DIMS> index) const
    {
        unsigned ret = 0;
        for (int i = 0; i < DIMS; i++)
            ret += (index[i] % shape[i]) * stride[i];
        return ret;
    }

    std::array<unsigned, DIMS> getPhysShape(std::array<unsigned, DIMS> shape) const
    {
        std::array<unsigned, DIMS> ret(shape);
        ret[DIMS - 1] = WARP_ROUND(shape[DIMS - 1]);
        return ret;
    }

    std::array<unsigned, DIMS> getStride(std::array<unsigned, DIMS> shape) const
    {
        std::array<unsigned, DIMS> phys_shape = getPhysShape(shape);
        std::array<unsigned, DIMS> ret;
        ret[DIMS - 1] = 1;
        for (int i = DIMS - 1; i > 0; i--)
            ret[i - 1] = ret[i] * phys_shape[i];
        return ret;
    }

    unsigned getSize(std::array<unsigned, DIMS> shape) const
    {
        unsigned ret = 1;
        for (int i = 0; i < DIMS; i++)
            ret *= shape[i];
        return ret;
    }

public:
    const std::array<unsigned, DIMS> shape;
    const unsigned size;

    Tensor(std::array<unsigned, DIMS> _shape)
        : shape(_shape), phys_shape(getPhysShape(_shape)),
          stride(getStride(_shape)), size(getSize(_shape)),
          phys_size(getSize(getPhysShape(_shape)))
    {
        checkCudaErrors(cudaMalloc(&data, size * sizeof(scalar_t)));
        checkCudaErrors(cudaMemset(data, 0, size * sizeof(scalar_t)));
    }

    __device__ scalar_t operator()(std::array<int, DIMS> index) const { return data[offset(index)]; }
    __device__ scalar_t &operator()(std::array<int, DIMS> index) { return data[offset(index)]; }
    __device__ scalar_t operator()(int index) const { return data[index]; }
    __device__ scalar_t &operator()(int index) { return data[index]; }

    scalar_t get(std::array<int, DIMS> index) const
    {
        scalar_t ret;
        checkCudaErrors(cudaMemcpy(&ret, data + offset(index), sizeof(scalar_t), cudaMemcpyDeviceToHost));
        return ret;
    }

    template <int NEW_DIMS>
    Tensor<scalar_t, NEW_DIMS> reshape(std::array<unsigned, NEW_DIMS> new_shape)
    {
        return Tensor<scalar_t, NEW_DIMS>(new_shape);
    }

    void fill(scalar_t value);
};

#endif // _TENSOR
