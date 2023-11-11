#include <vector>
#include "tensor.h"
#include "cuda_tensor.h"
#include <vector>
#include <functional>
#include <numeric>
#include <cuda_runtime.h>
#include "helper_cuda.h"

template <class T>
CudaTensor<T>::CudaTensor(std::vector<unsigned> shape) : Tensor<T>(shape)
{
    int count = sizeof(T) * Tensor<T>::size();
    checkCudaErrors(cudaMalloc(&data, count));
    checkCudaErrors(cudaMemset(data, 0, count));
}

template <class T>
CudaTensor<T> &CudaTensor<T>::operator=(const CudaTensor<T> &t)
{
    if (this != &t)
    {
        Tensor<T>::shape_ = t.shape_;
        int count = Tensor<T>::size() * sizeof(T);

        checkCudaErrors(cudaFree(data));
        checkCudaErrors(cudaMalloc(&data, count));
        checkCudaErrors(cudaMemcpy(data, t.data, count, cudaMemcpyDeviceToDevice));
    }
    return *this;
}

template <class T>
void CudaTensor<T>::set(std::vector<int> index, T value)
{
    int offset = Tensor<T>::offset(index);
    checkCudaErrors(cudaMemcpy(data + offset, &value, sizeof(T), cudaMemcpyHostToDevice));
}

template <class T>
T CudaTensor<T>::operator()(std::vector<int> index) const
{
    int offset = Tensor<T>::offset(index);
    T value;
    checkCudaErrors(cudaMemcpy(&value, data + offset, sizeof(T), cudaMemcpyDeviceToHost));
    return value;
}

template <class T>
CpuTensor<T> *CudaTensor<T>::cpu() const
{
    CpuTensor<T> *ret = new CpuTensor<T>(Tensor<T>::shape_);
    int count = Tensor<T>::size() * sizeof(T);
    checkCudaErrors(cudaMemcpy(ret->data.data(), data, count, cudaMemcpyDeviceToHost));
    return ret;
}

template class CudaTensor<int>;
