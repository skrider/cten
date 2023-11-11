#include <iostream>
#include "tensor.h"

using namespace std;

int main(int argc, char **argv)
{
    CpuTensor<int> a({3, 4, 5});
    CpuTensor<int> b({3, 5, 5});
    cout << "a size: " << a.size() << endl;
    cout << "b size: " << b.size() << endl;
    b = a;
    cout << "b size: " << b.size() << endl;
    a({1, 1, 1}) = 4;
    cout << "a elem: " << a({1, 2, 1}) << endl;
}