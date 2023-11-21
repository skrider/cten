#include "tensor.cuh"

template <typename T, uint D>
bool allclose(Tensor<T, D> a, Tensor<T, D> b, T eps);
