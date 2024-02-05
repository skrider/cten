#include "helper_cuda.h"
#include <chrono>
#include <cuda_runtime.h>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cutlass/gemm/kernel/sparse_gemm.h>
#include <iostream>

using namespace cute;

// number of queries
constexpr int M = (1 << 11);
using _M = Int<M>;
// number of keys
constexpr int N = (1 << 11);
using _N = Int<N>;
// head dim
constexpr int K = (1 << 9);
using _K = Int<K>;

// page size
constexpr int PageSize = 16;
constexpr int PageCount = N / PageSize;
static_assert(N % PageSize == 0, "N must be divisible by PageSize");

// page table - some permutation of 0..PageCount
using ElementPT = int32_t;

// query
using ElementQ = cutlass::half_t;
using LayoutQ = Layout<Shape<_M, _K>>;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementQ>::value;

// key
using ElementK = cutlass::half_t;
using LayoutK = Layout<Shape<_N, _K>>;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementK>::value;

int main(int argc, char **argv) {
  Tensor gQ_h = make_tensor<ElementQ>(LayoutQ{});
  Tensor gK_h = make_tensor<ElementK>(LayoutK{});

  Tensor gQ_h_permute = make_tensor<ElementQ>(LayoutQ{});
}

template <typename Element, typename Layout, int PageSize> __global__ void permute_kernel(__restrict__ Element *A, __restrict__ Element *A_permute, __restrict__ ElementPT *pt) {}

template <typename Element, typename Layout, int PageSize> __device__ __forceinline__ void permute_one_page(__restrict__ Element *A, __restrict__ Element *A_permute, ElementPT from, ElementPT to) {
  Tensor gA = make_tensor<Element>(make_gmem_ptr(A), Layout{});
  Tensor gA_permute = make_tensor<Element>(make_gmem_ptr(A_permute), Layout{});

  tensor rA_page = make_fragment_like(gA);

  using CopyAtom =
}
