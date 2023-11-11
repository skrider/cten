#include <vector>
#include "tensor.h"
#include <initializer_list>
#include <functional>
#include <numeric>
#include <algorithm>

template <class T>
Tensor<T>::Tensor(std::initializer_list<unsigned> shape__) : shape_(shape__)
{
    unsigned acc = 1;
    for (unsigned i : shape_)
    {
        stride_.push_back(acc);
        acc *= i;
    }
    std::reverse(stride_.begin(), stride_.end());
}

template <class T>
unsigned Tensor<T>::size() const
{
    return std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<unsigned>());
}

template <class T>
unsigned Tensor<T>::shape(unsigned i) const
{
    if (i >= shape_.size())
        throw std::runtime_error(OOB_MSG);
    return shape_[i];
}

template <class T>
unsigned Tensor<T>::offset(std::initializer_list<int> index_) const
{
    unsigned acc = 0;
    std::vector<int> index(index_);
    if (index.size() != shape_.size())
        throw std::runtime_error("bad index");
    for (unsigned i = 0; i < shape_.size(); i++)
    {
        acc += index[i] % shape_[i];
    }
    return acc;
}

template <class T>
CpuTensor<T>::CpuTensor(std::initializer_list<unsigned> shape__) : Tensor<T>(shape__)
{
    data.resize(Tensor<T>::size());
    std::fill(data.begin(), data.end(), 0);
}

template <class T>
CpuTensor<T> &CpuTensor<T>::operator=(const CpuTensor<T> &t)
{
    if (this != &t)
    {
        data = t.data;
        Tensor<T>::shape_ = t.shape_;
    }
    return *this;
}

template <class T>
T &CpuTensor<T>::operator()(std::initializer_list<int> index)
{
    return data[Tensor<T>::offset(index)];
}

template <class T>
T CpuTensor<T>::operator()(std::initializer_list<int> index) const
{
    return data[Tensor<T>::offset(index)];
}

// template <class T>
// T &CpuTensor<T>::operator()(const int index...) {}

template class CpuTensor<int>;
// template class Tensor<float>;
// template class Tensor<double>;
