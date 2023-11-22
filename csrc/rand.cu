#include "helper_cuda.h"
#include "rand.cuh"
#include <curand.h>
#include <curand_kernel.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>
#include <mutex>

template <uint D>
Tensor<float, D> randn(shape_t<D> shape, float shift, float scale, long seed) {
  Tensor<float, D> out(shape);
  randn(out, shift, scale, seed);
  return out;
}

template <uint D>
void randn(Tensor<float, D> t, float shift, float scale, long seed) {
  // use 256 threads total as this is maximum supported by curand mtgp32
  dim3 threadsPerBlock(256);
  // use max 200 blocks as mtgp supports generating up to 200 states
  dim3 numBlocks(MIN(200, t.size / (2 * threadsPerBlock.x)));

  mtgp32_kernel_params_t *kernel_params;
  curandStateMtgp32_t *states;
  curandStatus_t s;

  checkCudaErrors(cudaMalloc(&kernel_params, sizeof(mtgp32_kernel_params_t)));
  checkCudaErrors(
      cudaMalloc(&states, sizeof(curandStateMtgp32_t) * numBlocks.x));
  // 256 threads

  s = curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, kernel_params);
  if (s != CURAND_STATUS_SUCCESS) {
    printf("curandMakeMTGP32Constants failed with status %d\n", s);
    exit(1);
  }

  // make one state for each block
  s = curandMakeMTGP32KernelState(states, mtgp32dc_params_fast_11213,
                                  kernel_params, numBlocks.x, seed);
  if (s != CURAND_STATUS_SUCCESS) {
    printf("curandMakeMTGP32KernelState failed with status %d\n", s);
    exit(1);
  }
  randn_kernel<<<numBlocks, threadsPerBlock>>>(t, states, shift, scale);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaFree(kernel_params));
  checkCudaErrors(cudaFree(states));
}

__global__ void randn_kernel(Tensor<float, 2> t, curandStateMtgp32_t *states,
                             float shift, float scale) {
  // load state to shared memory
  __shared__ curandStateMtgp32_t state;
  state = states[blockIdx.x];
  // grid stride over underlying array
  for (int offset = blockIdx.x * blockDim.x * 2; offset < t.size;
       offset += gridDim.x * blockDim.x * 2) {
    int index = offset + threadIdx.x * 2;
    // generate two uniformly random integers using mtgp32 and use the
    // box-muller transform to convert into a 2d gaussian
    float2 r = curand_box_muller(&state);
    // size guarantueed not odd.
    // all threads in the block must participate in the random number
    // generation, however the block bounds may overlap the array end.
    if (index < t.size) {
      t(index) = r.x * scale + shift;
      t(index + 1) = r.y * scale + shift;
    }
  }
}

template void randn(Tensor<float, 2> t, float shift, float scale, long seed);
template Tensor<float, 2> randn(shape_t<2> t, float shift, float scale,
                                long seed);

template <uint D> void randu(Tensor<int, D> t, int a, int b, long seed) {
  // use 256 threads total as this is maximum supported by curand mtgp32
  dim3 threadsPerBlock(256);
  // use max 200 blocks as mtgp supports generating up to 200 states
  dim3 numBlocks(MIN(200, t.size / threadsPerBlock.x));

  mtgp32_kernel_params_t *kernel_params;
  curandStateMtgp32_t *states;
  curandStatus_t s;

  checkCudaErrors(cudaMalloc(&kernel_params, sizeof(mtgp32_kernel_params_t)));
  checkCudaErrors(
      cudaMalloc(&states, sizeof(curandStateMtgp32_t) * numBlocks.x));

  s = curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, kernel_params);
  if (s != CURAND_STATUS_SUCCESS) {
    printf("curandMakeMTGP32Constants failed with status %d\n", s);
    exit(1);
  }

  // make one state for each block
  s = curandMakeMTGP32KernelState(states, mtgp32dc_params_fast_11213,
                                  kernel_params, numBlocks.x, seed);
  if (s != CURAND_STATUS_SUCCESS) {
    printf("curandMakeMTGP32KernelState failed with status %d\n", s);
    exit(1);
  }
  randu_kernel<<<numBlocks, threadsPerBlock>>>(t, states, a, b);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaFree(kernel_params));
  checkCudaErrors(cudaFree(states));
}

__global__ void randu_kernel(Tensor<int, 2> t, curandStateMtgp32_t *states,
                             int a, int b) {
  // load state to shared memory
  __shared__ curandStateMtgp32_t state;
  state = states[blockIdx.x];
  // grid stride over underlying array
  for (int offset = blockIdx.x * blockDim.x; offset < t.size;
       offset += gridDim.x * blockDim.x) {
    int index = offset + threadIdx.x;
    int r = curand(&state) % (b - a) + a;
    if (index < t.size)
      t(index) = r;
  }
}

template void randu(Tensor<int, 2> t, int a, int b, long seed);
