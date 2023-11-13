#include <iostream>
#include "tensor.cuh"
#include "utils.cuh"
#include "ops.cuh"
#include <chrono>
#include <cuda_runtime.h>
#include "helper_cuda.h"

using namespace std;

#define ROW (1 << 12)
#define COL (1 << 12)
#define INNER (1 << 12)

#define WARMUP 1
#define N 10

int main(int argc, char **argv)
{
    Tensor<int, 2> a({ROW, INNER});
    Tensor<int, 2> b({INNER, COL});
    Tensor<int, 2> c({ROW, COL});
    Tensor<int, 2> out({ROW, COL});
    int alpha = 1;
    int beta = -1;

    a.fill(1);
    b.fill(2);
    c.fill(21);

    for (int i = 0; i < WARMUP; i++)
    {
        gemm(out, a, b, c, alpha, beta);
        gemm2(out, a, b, c, alpha, beta);
        gemm3(out, a, b, c, alpha, beta);
    }

    for (int i = 0; i < N; i++)
    {
        gemm(out, a, b, c, alpha, beta);
        checkCudaErrors(cudaDeviceSynchronize());
    }

    cout << "element:      " << out.get({31, 30}) << endl;

    for (int i = 0; i < N; i++)
    {
        gemm3(out, a, b, c, alpha, beta);
        checkCudaErrors(cudaDeviceSynchronize());
    }

    cout << "element:      " << out.get({31, 31}) << endl;
}
