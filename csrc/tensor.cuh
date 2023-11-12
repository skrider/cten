#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <array>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>

#ifndef _TENSOR
#define _TENSOR

#define OOB_MSG "error tensor index out of bounds"

#define WARP_SIZE 32
#define ROUND(a, b) (((a + b - 1) / b) * b)

template <typename scalar_t, unsigned DIMS>
class Tensor
{
private:
    // shape with rows rounded up to nearest warp dim
    const std::array<unsigned, DIMS> phys_shape;
    const unsigned phys_size;
    const std::array<unsigned, DIMS> stride;
    scalar_t *data;

public:
    __device__ inline unsigned doffset(std::array<int, DIMS> index) const
    {
        unsigned ret = 0;
        unsigned _shape[DIMS](shape);
        unsigned _stride[DIMS](stride);
        int _index[DIMS](index);
        for (int i = 0; i < DIMS; i++)
            ret += (_index[i] % _shape[i]) * _stride[i];
        return ret;
    }

    __host__ inline unsigned offset(std::array<int, DIMS> index) const
    {
        unsigned ret = 0;
        for (int i = 0; i < DIMS; i++)
            ret += (index[i] % shape[i]) * stride[i];
        return ret;
    }

private:
    std::array<unsigned, DIMS> getPhysShape(std::array<unsigned, DIMS> shape) const
    {
        std::array<unsigned, DIMS> ret(shape);
        ret[DIMS - 1] = ROUND(shape[DIMS - 1], WARP_SIZE);
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

    __device__ scalar_t operator()(std::array<int, DIMS> index) const { return data[doffset(index)]; }
    __device__ scalar_t &operator()(std::array<int, DIMS> index) { return data[doffset(index)]; }
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

    std::string string() const
    {
        std::vector<scalar_t> buf;
        buf.resize(phys_size);
        std::ostringstream builder;
        checkCudaErrors(cudaMemcpy(buf.data(), data, phys_size * sizeof(scalar_t), cudaMemcpyDeviceToHost));
        if (DIMS == 2)
        {
            for (int i = 0; i < shape[0]; i++)
            {
                for (int j = 0; j < shape[1]; j++)
                    builder << buf[offset({i, j})] << " ";
                builder << std::endl;
            }
        }
        return builder.str();
    }

    void fill(scalar_t value);
};

#endif // _TENSOR
