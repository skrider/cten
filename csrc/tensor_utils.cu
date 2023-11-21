#include "helper_cuda.h"
#include "tensor.cuh"
#include <cuda_runtime.h>

template <typename T, uint D>
__global__ void allclose_kernel(Tensor<T, D> a, Tensor<T, D> b, T eps,
                                bool *res) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < a.size) {
    if (a(idx) - b(idx) > eps) {
      *res = false;
    }
  }
}

template <typename T, uint D>
bool allclose(Tensor<T, D> a, Tensor<T, D> b, T eps) {
  bool *res, ret;

  checkCudaErrors(cudaMallocManaged(&res, sizeof(bool)));

  *res = true;

  dim3 threadsPerBlock(32 * 32);
  // todo add case for 1d matrix
  dim3 numBlocks(a.size / threadsPerBlock.x);

  allclose_kernel<<<numBlocks, threadsPerBlock>>>(a, b, eps, res);
  checkCudaErrors(cudaDeviceSynchronize());

  ret = *res;

  checkCudaErrors(cudaFree(res));
  return ret;
}

template bool allclose(Tensor<int, 2> a, Tensor<int, 2> b, int eps);
template bool allclose(Tensor<float, 2> a, Tensor<float, 2> b, float eps);
