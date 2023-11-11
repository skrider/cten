#include <iostream>
#include "tensor.cuh"
#include "utils.cuh"
#include "ops.cuh"

using namespace std;

int main(int argc, char **argv)
{
    printDeviceProperties();

    Tensor<int, 2> a({32, 32});
    Tensor<int, 2> b({32, 32});
    Tensor<int, 2> c({32, 32});
    int alpha = 4;
    int beta = -1;

    a.fill(1);
    b.fill(2);
    c.fill(21);

    Tensor<int, 2> out = gemm(a, b, c, alpha, beta);

    cout << "some value of a: " << a.get({3, 6}) << endl;
    cout << "some value of out: " << out.get({3, 6}) << endl;
}
