#include <vector>
#include "tensor.h"

template <class T>
class CudaTensor : public Tensor<T>
{
public:
    CudaTensor(std::initializer_list<unsigned> shape);
    CudaTensor(const Tensor<T> &t);
    CudaTensor &operator=(const CudaTensor<T> &t);
    T operator()(std::initializer_list<int> index) const override;

    void set(std::initializer_list<int> index, T value) override;

private:
    T *data;
};
