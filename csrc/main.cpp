#include <iostream>
#include "tensor.h"
#include "cuda_tensor.h"

using namespace std;

int main(int argc, char **argv)
{
    CpuTensor<int> a({3, 4, 5});
    CpuTensor<int> b({3, 5, 5});
    cout << "a size: " << a.size() << endl;
    cout << "b size: " << b.size() << endl;
    b = a;
    cout << "b size: " << b.size() << endl;
    a.set({1, 1, 1}, 4);
    cout << "a elem: " << a({1, 2, 1}) << endl;

    // test cuda tensors
    CudaTensor<int> c({100, 1000, 70});
    CudaTensor<int> d({100, 1000, 69});
    cout << "c size: " << c.size() << endl;
    cout << "d size: " << d.size() << endl;
    d.set({10, 10, 10}, 21);
    c = d;
    cout << "c elem: " << c({10, 10, 10}) << endl;
}