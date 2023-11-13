#include "tensor.cuh"

template <typename scalar_t, uint DIMS>
Tensor<scalar_t, DIMS> arange();

/*
General matrix multiply. Computes the following:
    O = alpha * AB + beta * C
For now, assumes A, B, C are two dimensional. */
template <typename scalar_t>
Tensor<scalar_t, 2> gemm(Tensor<scalar_t, 2> a, Tensor<scalar_t, 2> b, Tensor<scalar_t, 2> c, scalar_t alpha, scalar_t beta);

template <typename scalar_t>
void gemm(Tensor<scalar_t, 2> out, Tensor<scalar_t, 2> a, Tensor<scalar_t, 2> b, Tensor<scalar_t, 2> c, scalar_t alpha, scalar_t beta);

template <typename scalar_t>
void gemm2(
    Tensor<scalar_t, 2> out,
    Tensor<scalar_t, 2> a,
    Tensor<scalar_t, 2> b,
    Tensor<scalar_t, 2> c,
    scalar_t alpha,
    scalar_t beta);

template <typename scalar_t>
void gemm3(
    Tensor<scalar_t, 2> out,
    Tensor<scalar_t, 2> a,
    Tensor<scalar_t, 2> b,
    Tensor<scalar_t, 2> c,
    scalar_t alpha,
    scalar_t beta);
