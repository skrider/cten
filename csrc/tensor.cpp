#include <vector>
#include "tensor.h"
#include <cstdarg>
#include <initializer_list>
#include <functional>
#include <numeric>

template <class T>
unsigned Tensor<T>::size()
{
    return std::accumulate(this->shape.begin(), this->shape.end(), 1, std::multiplies<unsigned>());
}

template <class T>
Tensor<T>::Tensor(std::initializer_list<unsigned> shape) : shape(shape)
{
}

// template <class T>
// Tensor<T> &Tensor<T>::operator=(const Tensor<T> &t) {}
//
// template <class T>
// T &Tensor<T>::operator()(const int index...) {}

template class Tensor<int>;
// template class Tensor<float>;
// template class Tensor<double>;
