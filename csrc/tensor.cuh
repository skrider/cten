#include <cuda_runtime.h>

#ifndef _TENSOR
#define _TENSOR

#define OOB_MSG "error tensor index out of bounds"

#define WARP_SIZE 32
#define WARP_ROUND (size)(((size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE)

template <typename scalar_t, int DIMS>
class Tensor
{

private:
    // shape with rows rounded up to nearest warp dim
    const unsigned[DIMS] phys_shape;
    const unsigned phys_size;
    const unsigned[DIMS] stride;
    scalar_t *data;

    unsigned offset(int[DIMS] index) const;
    {
        unsigned ret = 0;
        for (int i = 0; i < DIMS; i++)
            ret += index[i] * _stride[i];
        return ret;
    }

    unsigned[DIMS] getPhysShape(unsigned[DIMS] shape) const
    {
        unsigned[DIMS] ret(shape);
        ret[DIMS - 1] = WARP_ROUND(shape[DIMS - 1]);
        return ret;
    }

    unsigned[DIMS] getStride(unsigned[DIMS] shape) const
    {
        unsigned[DIMS] phys_shape = getPhysShape(shape);
        unsigned[DIMS] ret;
        ret[DIMS - 1] = 1;
        for (int i = DIMS - 1; i > 0; i--)
            ret[i - 1] = ret[i] * phys_shape[i];
        return ret;
    }

    unsigned getSize(unsigned[DIMS] shape) const
    {
        unsigned ret = 1;
        for (int i = DIMS - 1; i > 0; i--)
            ret *= shape[i];
        return ret;
    }

public:
    const unsigned[DIMS] shape;
    const unsigned size;

    Tensor(unsigned[DIMS] _shape)
        : shape(__shape), phys_shape(getPhysShape(_shape)),
          stride(getStride(_shape)), size(getSize(_shape)),
          phys_size(getSize(getPhysShape(_shape)))
    {
        checkCudaErrors(cudaMalloc(&data, _size * sizeof(scalar_t)));
        checkCudaErrors(cudaMemset(data, 0, _size * sizeof(scalar_t)));
    }

    __device__ scalar_t operator()(int[DIMS] index) const
    {
        return data[offset(index)];
    }

    __device__ scalar_t &operator()(int[DIMS] index)
    {
        return &data[offset(index)];
    }

    Tensor<scalar_t, DIMS> reshape(unsigned[DIMS] new_shape);
};

#endif // _TENSOR
