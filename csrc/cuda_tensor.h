#include <vector>
#include "tensor.h"

template <class T>
class CudaTensor : public Tensor<T>
{
public:
    CudaTensor(std::vector<unsigned> shape);
    CudaTensor &operator=(const CudaTensor<T> &t);
    T operator()(std::vector<int> index) const override;

    void set(std::vector<int> index, T value) override;

    CpuTensor<T> *cpu() const;

    friend class CpuTensor<T>;

protected:
    T *data;
};
