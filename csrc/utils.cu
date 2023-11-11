#include "helper_cuda.h"
#include <iostream>
#include <cuda_runtime.h>
#include "utils.cuh"

void printDeviceProperties()
{
    cudaDeviceProp prop;
    int device;

    cudaGetDevice(&device); // get the current device
    cudaGetDeviceProperties(&prop, device);

    std::cout << "Warp size: " << prop.warpSize << std::endl;
}