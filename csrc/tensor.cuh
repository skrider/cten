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

template <uint D> using shape_t = uint[D];
template <uint D> using index_t = int[D];

template <typename T, uint D> class Tensor {
private:
  // shape with rows rounded up to nearest warp dim
  shape_t<D> stride;
  T *data;

  std::array<int, D> indexof(int offset) const {
    std::array<int, D> ret;
    for (int i = 0; i < D; i++) {
      ret[i] = offset / stride[i];
      offset -= ret[i] * stride[i];
    }
    return ret;
  }

public:
  __device__ inline uint doffset(int *index) const {
    uint ret = 0;
    for (int i = 0; i < D; i++)
      ret += (index[i] % shape[i]) * stride[i];
    return ret;
  }

  __host__ inline uint offset(std::array<int, D> index) const {
    uint ret = 0;
    for (int i = 0; i < D; i++)
      ret += (index[i] % shape[i]) * stride[i];
    return ret;
  }

  shape_t<D> shape;
  uint size;

  Tensor(std::array<uint, D> _shape) {
    for (int i = 0; i < D; i++) {
      if (_shape[i] % WARP_SIZE != 0)
        throw std::runtime_error("shape must be a multiple of 32");
      shape[i] = _shape[i];
    }

    size = 1;
    for (int i = 0; i < D; i++) {
      size *= shape[i];
    }

    stride[D - 1] = 1;
    for (int i = D - 1; i > 0; i--)
      stride[i - 1] = stride[i] * shape[i];

    checkCudaErrors(cudaMalloc(&data, size * sizeof(T)));
    checkCudaErrors(cudaMemset(data, 0, size * sizeof(T)));
  }

  Tensor(shape_t<D> _shape) : Tensor(packCArr<uint, D>(_shape)) {}

  __device__ T operator()(int *index) const { return data[doffset(index)]; }
  __device__ T &operator()(int *index) { return data[doffset(index)]; }
  __device__ T operator()(int index) const { return data[index]; }
  __device__ T &operator()(int index) { return data[index]; }

  T get(std::array<int, D> index) const {
    T ret;
    checkCudaErrors(cudaMemcpy(&ret, data + offset(index), sizeof(T),
                               cudaMemcpyDeviceToHost));
    return ret;
  }

  template <int NEW_D> Tensor<T, NEW_D> reshape(shape_t<NEW_D> new_shape) {
    return Tensor<T, NEW_D>(new_shape);
  }

  std::string string() const {
    std::vector<T> buf;
    buf.resize(size);
    std::ostringstream builder;
    checkCudaErrors(
        cudaMemcpy(buf.data(), data, size * sizeof(T), cudaMemcpyDeviceToHost));
    if (D == 2) {
      for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[1]; j++)
          builder << buf[offset({i, j})] << " ";
        builder << '\n';
      }
    }
    return builder.str();
  }

  void fill(T value);
  T max() const;
  T min() const;
  std::array<int, D> argmax() const;
  std::array<int, D> argmin() const;
  Tensor<T, D> operator-(const Tensor<T, D> &other) const;
  Tensor<T, D> operator-(const T scalar) const;
  Tensor<T, D> operator+(const Tensor<T, D> &other) const;
  Tensor<T, D> operator+(const T scalar) const;
  Tensor<T, D> round(const T scalar) const;
};

#endif // _TENSOR
