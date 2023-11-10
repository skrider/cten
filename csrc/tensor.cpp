#include <vector>
#include "tensor.h"
#include <cstdarg>
#include <initializer_list>
#include <functional>
#include <numeric>

template <class T>
Tensor<T>::Tensor(std::initializer_list<unsigned> shape) : shape(shape)
{
    data.resize(size());
    std::fill(data.begin(), data.end(), 0);
}

template <class T>
unsigned Tensor<T>::size()
{
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<unsigned>());
}

template <class T>
Tensor<T> &Tensor<T>::operator=(const Tensor<T> &t)
{
    if (this != &t)
    {
        data = t.data;
        shape = t.shape;
    }
    return *this;
}

// template <class T>
// Tensor<T> &Tensor<T>::operator=(const Tensor<T> &t) {}
//
// template <class T>
// T &Tensor<T>::operator()(const int index...) {}

template class Tensor<int>;
// template class Tensor<float>;
// template class Tensor<double>;
