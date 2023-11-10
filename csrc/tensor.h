#include <vector>

template <class T>
class Tensor
{
public:
    Tensor(std::initializer_list<unsigned> shape);
    // Tensor(const Tensor &t);
    // Tensor &operator=(const Tensor &t);
    // T &operator()(Index... index);
    // T operator()(Index... index) const;
    unsigned size();
    // ~Tensor();

private:
    T *data;
    std::vector<unsigned> shape;
};
