#include <vector>
#include "tensor.h"
#include <vector>
#include <functional>
#include <numeric>
#include <algorithm>

// Tensor helper methods

template <class T>
Tensor<T>::Tensor(std::vector<unsigned> shape__) : shape_(shape__)
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
unsigned Tensor<T>::offset(std::vector<int> index_) const
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

// CPU Tensor

template <class T>
CpuTensor<T>::CpuTensor(std::vector<unsigned> shape__) : Tensor<T>(shape__)
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
void CpuTensor<T>::set(std::vector<int> index, T value)
{
    data[Tensor<T>::offset(index)] = value;
}

template <class T>
T CpuTensor<T>::operator()(std::vector<int> index) const
{
    return data[Tensor<T>::offset(index)];
}

template class CpuTensor<int>;
// template class Tensor<float>;
// template class Tensor<double>;
