#include <array>

template <typename T, uint D> std::array<T, D> packCArr(T c_array[D]);

void printDeviceProperties();

#define MIN(a, b) ((a) < (b)) ? (a) : (b)
#define MAX(a, b) ((a) < (b)) ? (b) : (a)