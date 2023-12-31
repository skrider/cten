#include "tensor.cuh"
#include "utils.cuh"
#include <cuda_runtime.h>

template <typename T, uint D>
__global__ void fill_kernel(Tensor<T, D> t, T value) {
  t(blockIdx.x * blockDim.x + threadIdx.x) = value;
}

template <typename T, uint D> void Tensor<T, D>::fill(T value) {
  dim3 threadsPerBlock(32);
  dim3 numBlocks(size / threadsPerBlock.x);
  fill_kernel<<<numBlocks, threadsPerBlock>>>(*this, value);
}

// MODE = 0 for max, 1 for min
template <typename T, uint D, uint MODE>
__global__ void max_kernel(Tensor<T, D> t, T *max, int *argmax) {
#define CMP_SET(a, b, ai, bi)                                                  \
  if (MODE == 0) {                                                             \
    if (a < b) {                                                               \
      a = b;                                                                   \
      ai = bi;                                                                 \
    }                                                                          \
  } else {                                                                     \
    if (a > b) {                                                               \
      a = b;                                                                   \
      ai = bi;                                                                 \
    }                                                                          \
  }
#define WARP_REDUCE(a, ai)                                                     \
  _Pragma("unroll") for (int offset = WARP_SIZE / 2; offset > 0;               \
                         offset /= 2) {                                        \
    int bi = __shfl_down_sync(0xffffffff, value_index, offset);                \
    T b = __shfl_down_sync(0xffffffff, value, offset);                         \
    CMP_SET(a, b, ai, bi)                                                      \
  }

  __shared__ T warp_max[32];
  __shared__ T warp_argmax[32];

  int lane = threadIdx.x % WARP_SIZE;
  int wid = threadIdx.x / WARP_SIZE;

  // init accumulator
  if (wid == 0) {
    warp_max[lane] = t(lane);
  }

  T value;
  int value_index;

  for (int i = 0; i < t.size; i += blockDim.x) {
    value_index = i + threadIdx.x;
    if (value_index < t.size)
      value = t(value_index);

    // quickly reduce across warps
    WARP_REDUCE(value, value_index)

    if (lane == 0) {
      CMP_SET(warp_max[wid], value, warp_argmax[wid], value_index)
    }
  }

  __syncthreads();

  // reduce across smem
  if (wid == 0) {
    value = warp_max[lane];
    value_index = warp_argmax[lane];

    WARP_REDUCE(value, value_index)

    if (lane == 0) {
      if (max != nullptr)
        *max = value;
      if (argmax != nullptr)
        *argmax = value_index;
    }
  }
}

template <typename T, uint D> T Tensor<T, D>::max() const {
  T *max_value, max_value_host;
  checkCudaErrors(cudaMalloc(&max_value, sizeof(T)));
  max_kernel<T, D, 0><<<1, 1024>>>(*this, max_value, (int *)nullptr);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(&max_value_host, max_value, sizeof(T),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(max_value));
  return max_value_host;
}

template <typename T, uint D> std::array<int, D> Tensor<T, D>::argmax() const {
  int *argmax, argmax_host;
  checkCudaErrors(cudaMalloc(&argmax, sizeof(int)));
  max_kernel<T, D, 0><<<1, 1024>>>(*this, (T *)nullptr, argmax);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(
      cudaMemcpy(&argmax_host, argmax, sizeof(int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(argmax));

  return indexof(argmax_host);
}

template <typename T, uint D> T Tensor<T, D>::min() const {
  T *min_value, min_value_host;
  checkCudaErrors(cudaMalloc(&min_value, sizeof(T)));
  max_kernel<T, D, 1><<<1, 1024>>>(*this, min_value, (int *)nullptr);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(&min_value_host, min_value, sizeof(T),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(min_value));
  return min_value_host;
}

template <typename T, uint D> std::array<int, D> Tensor<T, D>::argmin() const {
  int *argmin, argmin_host;
  checkCudaErrors(cudaMalloc(&argmin, sizeof(int)));
  max_kernel<T, D, 1><<<1, 1024>>>(*this, (T *)nullptr, argmin);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(
      cudaMemcpy(&argmin_host, argmin, sizeof(int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(argmin));

  return indexof(argmin_host);
}

/*
MODE: 0 for addition, 1 for subtraction
*/
template <typename T, uint D, uint MODE>
__global__ void elementwise_kernel(Tensor<T, D> out, const Tensor<T, D> a,
                                   const Tensor<T, D> b) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < a.size)
    if (MODE == 0) {
      out(idx) = a(idx) + b(idx);
    } else if (MODE == 1) {
      out(idx) = a(idx) - b(idx);
    }
}

/*
MODE: 0 for addition, 1 for subtraction, 2 for round to nearest
*/
template <typename T, uint D, uint MODE>
__global__ void scalar_elementwise_kernel(Tensor<T, D> out,
                                          const Tensor<T, D> a, T scalar) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < a.size)
    if (MODE == 0) {
      out(idx) = a(idx) + scalar;
    } else if (MODE == 1) {
      out(idx) = a(idx) - scalar;
    } else if (MODE == 2) {
      out(idx) = roundf(a(idx) / scalar) * scalar;
    }
}

template <typename T, uint D>
Tensor<T, D> Tensor<T, D>::operator-(const Tensor<T, D> &other) const {
  Tensor<T, D> out(packCArr<uint, D>((uint *)shape));
  elementwise_kernel<T, D, 1><<<size / 1024, 1024>>>(out, *this, other);
  return out;
}
template <typename T, uint D>
Tensor<T, D> Tensor<T, D>::operator-(const T scalar) const {
  Tensor<T, D> out(packCArr<uint, D>((uint *)shape));
  scalar_elementwise_kernel<T, D, 1><<<size / 1024, 1024>>>(out, *this, scalar);
  return out;
}

template <typename T, uint D>
Tensor<T, D> Tensor<T, D>::operator+(const Tensor<T, D> &other) const {
  Tensor<T, D> out(packCArr<uint, D>((uint *)shape));
  elementwise_kernel<T, D, 0><<<size / 1024, 1024>>>(out, *this, other);
  return out;
}
template <typename T, uint D>
Tensor<T, D> Tensor<T, D>::operator+(const T scalar) const {
  Tensor<T, D> out(packCArr<uint, D>((uint *)shape));
  scalar_elementwise_kernel<T, D, 0><<<size / 1024, 1024>>>(out, *this, scalar);
  return out;
}
template <typename T, uint D>
Tensor<T, D> Tensor<T, D>::round(const T scalar) const {
  Tensor<T, D> out(packCArr<uint, D>((uint *)shape));
  scalar_elementwise_kernel<T, D, 2><<<size / 1024, 1024>>>(out, *this, scalar);
  return out;
}

template class Tensor<int, 2>;
template class Tensor<float, 2>;
