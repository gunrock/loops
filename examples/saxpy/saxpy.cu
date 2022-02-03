/**
 * @file saxpy.cu
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Simple CUDA example of saxpy (Single-Precision A·X Plus Y) using
 * ranged loops.
 * @version 0.1
 * @date 2022-02-02
 *
 * @copyright Copyright (c) 2022
 *
 */
#include <loops/container/vector.hxx>
#include <loops/grid_stride_range.hxx>
#include <loops/util/generate.hxx>

#include <thrust/copy.h>

template <typename type_t>
__global__ void saxpy(int n, type_t a, type_t* x, type_t* y) {
  /// Equivalent to:
  /// i = blockIdx.x * blockDim.x + threadIdx.x; (init)
  /// i < n; (boundary condition)
  /// i += gridDim.x * blockDim.x. (step)
  for (auto i : loops::grid_stride_range(0, n)) {
    y[i] += a * x[i];
  }
}

int main() {
  using type_t = float;
  constexpr int N = 1 << 20;
  constexpr type_t alpha = 2.0f;

  // Create thrust device vectors.
  loops::vector_t<type_t> x(N);
  loops::vector_t<type_t> y(N);

  // Generate random numbers between [0, 1].
  loops::generate::random::uniform_distribution(x.begin(), x.end());
  loops::generate::random::uniform_distribution(y.begin(), y.end());

  // Launch kernel with a given configuration.
  constexpr std::size_t threads_per_block = 256;
  std::size_t blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
  saxpy<<<blocks_per_grid, threads_per_block>>>(N, alpha, x.data().get(),
                                                y.data().get());

  // Print the x and y vectors.
  std::cout << "x = ";
  thrust::copy(x.begin(), (x.size() < 10) ? x.end() : x.begin() + 10,
               std::ostream_iterator<type_t>(std::cout, " "));
  std::cout << std::endl;

  std::cout << "y = ";
  thrust::copy(y.begin(), (y.size() < 10) ? y.end() : y.begin() + 10,
               std::ostream_iterator<type_t>(std::cout, " "));
  std::cout << std::endl;
}