#include "helper_cuda.h"
#include "utils.cuh"
#include <algorithm>
#include <array>
#include <bits/range_access.h>
#include <cuda_runtime.h>
#include <iostream>

void printDeviceProperties() {
  cudaDeviceProp prop;
  int device;

  cudaGetDevice(&device); // get the current device
  cudaGetDeviceProperties(&prop, device);

  std::cout << "Warp size: " << prop.warpSize << std::endl;
}

template <typename T, uint D> std::array<T, D> packCArr(T *c_array) {
  std::array<T, D> std_array;
  for (int i = 0; i < D; i++)
    std_array[i] = c_array[i];
  return std_array;
}

template std::array<uint, 2> packCArr<uint, 2>(uint *c_array);
