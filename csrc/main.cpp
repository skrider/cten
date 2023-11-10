#include <iostream>
#include "tensor.h"

using namespace std;

int main(int argc, char **argv)
{
    Tensor<int> a({3, 4, 5});
    Tensor<int> b({3, 5, 5});
    cout << "a size: " << a.size() << endl;
    cout << "b size: " << b.size() << endl;
    b = a;
    cout << "b size: " << b.size() << endl;
}