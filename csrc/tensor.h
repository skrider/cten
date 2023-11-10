#include <vector>

template <class T>
class Tensor
{
public:
    Tensor(std::initializer_list<unsigned> shape);
    Tensor(const Tensor &t);
    Tensor &operator=(const Tensor &t);
    T &operator()(std::initializer_list<int> index);
    T operator()(std::initializer_list<int> index) const;
    unsigned size();

private:
    std::vector<T> data;
    std::vector<unsigned> shape;
};
