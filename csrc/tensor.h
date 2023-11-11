#include <vector>

#ifndef _TENSOR
#define _TENSOR

#define OOB_MSG "error tensor index out of bounds"

template <class T>
class Tensor
{
public:
    Tensor(std::vector<unsigned> shape);
    unsigned size() const;
    unsigned shape(unsigned i) const;

    virtual T operator()(std::vector<int> index) const = 0;

    virtual void set(std::vector<int> index, T value) = 0;

protected:
    std::vector<unsigned> shape_;
    std::vector<unsigned> stride_;

    unsigned offset(std::vector<int> index) const;
};

template <class T>
class CpuTensor : public Tensor<T>
{
public:
    CpuTensor(std::vector<unsigned> shape);
    CpuTensor &operator=(const CpuTensor<T> &t);
    T operator()(std::vector<int> index) const override;

    void set(std::vector<int> index, T value) override;

protected:
    std::vector<T> data;
};

#endif // _TENSOR
