#include "helper_cuda.h"
#include "utils.cuh"
#include <array>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifndef _TENSOR
#define _TENSOR

#define OOB_MSG "error tensor index out of bounds"

#define WARP_SIZE 32
#define ROUND(a, b) (((a + b - 1) / b) * b)

template <typename scalar_t, uint DIMS> class Tensor {
private:
  // shape with rows rounded up to nearest warp dim
  uint stride[DIMS];
  scalar_t *data;

public:
  __device__ inline uint doffset(int *index) const {
    uint ret = 0;
    for (int i = 0; i < DIMS; i++)
      ret += (index[i] % shape[i]) * stride[i];
    return ret;
  }

  __host__ inline uint offset(std::array<int, DIMS> index) const {
    uint ret = 0;
    for (int i = 0; i < DIMS; i++)
      ret += (index[i] % shape[i]) * stride[i];
    return ret;
  }

  uint shape[DIMS];
  uint size;

  Tensor(std::array<uint, DIMS> _shape) {
    for (int i = 0; i < DIMS; i++) {
      if (_shape[i] % WARP_SIZE != 0)
        throw std::runtime_error("shape must be a multiple of 32");
      shape[i] = _shape[i];
    }

    size = 1;
    for (int i = 0; i < DIMS; i++) {
      size *= shape[i];
    }

    stride[DIMS - 1] = 1;
    for (int i = DIMS - 1; i > 0; i--)
      stride[i - 1] = stride[i] * shape[i];

    checkCudaErrors(cudaMalloc(&data, size * sizeof(scalar_t)));
    checkCudaErrors(cudaMemset(data, 0, size * sizeof(scalar_t)));
  }

  Tensor(uint _shape[DIMS]) : Tensor(packCArr<uint, DIMS>(_shape)) {}

  __device__ scalar_t operator()(int *index) const {
    return data[doffset(index)];
  }
  __device__ scalar_t &operator()(int *index) { return data[doffset(index)]; }
  __device__ scalar_t operator()(int index) const { return data[index]; }
  __device__ scalar_t &operator()(int index) { return data[index]; }

  scalar_t get(std::array<int, DIMS> index) const {
    scalar_t ret;
    checkCudaErrors(cudaMemcpy(&ret, data + offset(index), sizeof(scalar_t),
                               cudaMemcpyDeviceToHost));
    return ret;
  }

  template <int NEW_DIMS>
  Tensor<scalar_t, NEW_DIMS> reshape(std::array<uint, NEW_DIMS> new_shape) {
    return Tensor<scalar_t, NEW_DIMS>(new_shape);
  }

  std::string string() const {
    std::vector<scalar_t> buf;
    buf.resize(size);
    std::ostringstream builder;
    checkCudaErrors(cudaMemcpy(buf.data(), data, size * sizeof(scalar_t),
                               cudaMemcpyDeviceToHost));
    if (DIMS == 2) {
      for (int i = 0; i < shape[0]; i++) {
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
