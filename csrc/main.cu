#include <iostream>
#include "tensor.cuh"
#include "utils.cuh"
#include "ops.cuh"
#include <chrono>
#include <cuda_runtime.h>
#include "helper_cuda.h"

using namespace std;

#define ROW (1 << 10)
#define COL (1 << 10)
#define INNER (1 << 10)

#define WARMUP 4
#define N 10

int main(int argc, char **argv)
{
    Tensor<int, 2> a({ROW, INNER});
    Tensor<int, 2> b({INNER, COL});
    Tensor<int, 2> c({ROW, COL});
    int alpha = 1;
    int beta = -1;

    a.fill(1);
    b.fill(2);
    c.fill(21);

    Tensor<int, 2> out = gemm(a, b, c, alpha, beta);

    for (int i = 0; i < WARMUP; i++)
        gemm2(out, a, b, c, alpha, beta);

    auto start = std::chrono::system_clock::now();

    for (int i = 0; i < N; i++)
    {
        gemm2(out, a, b, c, alpha, beta);
        checkCudaErrors(cudaDeviceSynchronize());
    }

    auto duration = chrono::system_clock::now() - start;
    cout << "average time: " << duration.count() / N << endl;
    cout << "element:      " << out.get({31, 31}) << endl;
}
