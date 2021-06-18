#include <thrust/device_vector.h>

// (loops) includes.
#include <loops/generate.hxx>
#include <loops/range.hxx>

using namespace loops;

template <typename T>
using step_range_t = typename range_proxy<T>::step_range_proxy;

template <typename T>
__device__ step_range_t<T> grid_stride_range(T begin, T end) {
  begin += blockDim.x * blockIdx.x + threadIdx.x;
  return range(begin, end).step(gridDim.x * blockDim.x);
}

template <typename type_t>
__global__ void saxpy(int n, type_t a, type_t* x, type_t* y) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    y[i] = a * x[i] + y[i];
  }
}

int main() {
  using type_t = float;
  const int N = 1 << 20;
  const type_t alpha = 2.0f;

  thrust::device_vector<type_t> x(N);
  thrust::device_vector<type_t> y(N);

  generate::random::uniform_distribution(x);
  generate::random::uniform_distribution(y);

  std::size_t threads_per_block = 256;
  std::size_t blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
  saxpy<<<blocks_per_grid, threads_per_block>>>(N, alpha, x.data().get(),
                                                y.data().get());

  std::cout << "x = " << std::endl;
  thrust::copy(x.begin(), (x.size() < 10) ? x.end() : x.begin() + 10,
               std::ostream_iterator<type_t>(std::cout, " "));
  std::cout << std::endl;

  std::cout << "y = " << std::endl;
  thrust::copy(y.begin(), (y.size() < 10) ? y.end() : y.begin() + 10,
               std::ostream_iterator<type_t>(std::cout, " "));
  std::cout << std::endl;
}