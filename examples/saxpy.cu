#include <loops/range.hxx>

#include <thrust/device_vector.h>

using namespace loops;

template <typename T>
using step_range_t = typename range_proxy<T>::step_range_proxy;

template <typename T>
__device__ step_range_t<T> grid_stride_range(T begin, T end) {
  begin += blockDim.x * blockIdx.x + threadIdx.x;
  return range(begin, end).step(gridDim.x * blockDim.x);
}

__global__ void saxpy(int n, float a, float* x, float* y) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    y[i] = a * x[i] + y[i];
  }
}

int main() {
  using type_t = float;
  const int N = 1 << 20;
  const type_t a = 2.0f;

  thrust::device_vector<type_t> x(N);
  thrust::device_vector<type_t> y(N);

  std::size_t threads_per_block = 256;
  std::size_t blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
  saxpy<<<blocks_per_grid, threads_per_block>>>(N, a, x, y);
  cudaDeviceSynchronize();
}