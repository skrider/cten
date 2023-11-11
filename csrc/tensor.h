#include <vector>

#ifndef _TENSOR
#define _TENSOR

#define OOB_MSG "error tensor index out of bounds"

template <class T>
class Tensor
{
public:
    Tensor(std::initializer_list<unsigned> shape);
    unsigned size() const;
    unsigned shape(unsigned i) const;

    virtual T operator()(std::initializer_list<int> index) const = 0;

    virtual void set(std::initializer_list<int> index, T value) = 0;

protected:
    std::vector<unsigned> shape_;
    std::vector<unsigned> stride_;

    unsigned offset(std::initializer_list<int> index) const;
};

template <class T>
class CpuTensor : public Tensor<T>
{
public:
    CpuTensor(std::initializer_list<unsigned> shape);
    CpuTensor(const Tensor<T> &t);
    CpuTensor &operator=(const CpuTensor<T> &t);
    T operator()(std::initializer_list<int> index) const override;

    void set(std::initializer_list<int> index, T value) override;

private:
    std::vector<T> data;
};

#endif // _TENSOR
