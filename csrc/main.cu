#include <iostream>
#include "tensor.cuh"
#include "utils.cuh"

using namespace std;

int main(int argc, char **argv)
{
    printDeviceProperties();

    Tensor<int, 3> a({300, 400, 500});
    Tensor<int, 1> b({32});
    cout << "a size: " << a.size << endl;
    cout << "b size: " << b.size << endl;
}
