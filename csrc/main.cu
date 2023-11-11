#include <iostream>
#include "tensor.cuh"
#include "cuda_utils.cuh"

using namespace std;

int main(int argc, char **argv)
{
    printDeviceProperties();

    Tensor<int, 3> a({300, 400, 500});
    Tensor<int, 3> b;
    cout << "a size: " << a.size << endl;
    b = a;
    cout << "b size: " << b.size << endl;
}