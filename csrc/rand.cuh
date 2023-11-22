#include "tensor.cuh"

template <uint D>
Tensor<float, D> randn(shape_t<D> shape, float shift, float scale, long seed);

template <uint D>
void randn(Tensor<float, D> t, float shift, float scale, long seed);

template <uint D> void randu(Tensor<int, D> t, int a, int b, long seed);
