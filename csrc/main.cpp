#include <iostream>
#include "tensor.h"

using namespace std;

int main(int argc, char **argv)
{
    Tensor<int> t({3, 4, 5});
    cout << "size: " << t.size() << endl;
}