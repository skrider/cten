#include <array>

template <typename T, uint D> std::array<T, D> packCArr(T c_array[D]);

void printDeviceProperties();