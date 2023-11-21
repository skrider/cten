#include "tensor.cuh"

/*
General matrix multiply. Computes the following:
    O = alpha * AB + beta * C
For now, assumes A, B, C are two dimensional. */
template <typename T>
void gemm1(Tensor<T, 2> out, Tensor<T, 2> a, Tensor<T, 2> b, Tensor<T, 2> c,
           T alpha, T beta);

template <typename T>
void gemm2(Tensor<T, 2> out, Tensor<T, 2> a, Tensor<T, 2> b, Tensor<T, 2> c,
           T alpha, T beta);

template <typename T>
void gemm3(Tensor<T, 2> out, Tensor<T, 2> a, Tensor<T, 2> b, Tensor<T, 2> c,
           T alpha, T beta);