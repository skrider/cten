#include <array>

template <typename T, uint D> std::array<T, D> packCArr(T c_array[D]);

void printDeviceProperties();

#ifndef MIN
#define MIN(a, b) ((a) < (b)) ? (a) : (b)
#endif // MIN

#ifndef MAX
#define MAX(a, b) ((a) < (b)) ? (b) : (a)
#endif // MAX
