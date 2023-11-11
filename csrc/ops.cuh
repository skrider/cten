#include "tensor.h"
#include "cuda_tensor.h"

/*
General matrix multiply. Computes the following:
    O = alpha * AB + beta * C
For now, assumes A, B, C are two dimensional. */
template <class T>
CudaTensor<T> &gemm(CudaTensor<T> a, CudaTensor<T> b, CudaTensor<T> c, T alpha, T beta);
