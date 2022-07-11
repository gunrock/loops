/**
 * @file work_oriented.cu
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Sparse Matrix-Vector Multiplication example.
 * @version 0.1
 * @date 2022-02-03
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "helpers.hxx"

using namespace loops;

/**
 * @brief Work oriented SpMV kernel.
 *
 * @tparam threads_per_block Number of threads per block.
 * @tparam index_t Type of column indices.
 * @tparam offset_t Type of row offsets.
 * @tparam type_t Type of values.
 */
template <std::size_t threads_per_block,
          typename index_t,
          typename offset_t,
          typename type_t>
__global__ void __launch_bounds__(threads_per_block, 2)
    work_oriented_spmv(std::size_t rows,
                       std::size_t cols,
                       std::size_t nnz,
                       offset_t* offsets,
                       index_t* indices,
                       const type_t* values,
                       const type_t* x,
                       type_t* y) {
  using setup_t =
      schedule::setup<schedule::algorithms_t::work_oriented, threads_per_block,
                      1, index_t, offset_t, std::size_t, std::size_t>;

  setup_t config(offsets, rows, nnz);
  auto map = config.init();

  /// Accumulate the complete tiles.
  type_t sum = 0;
  for (auto row : config.tiles(map)) {
    for (auto nz : config.atoms(row, map)) {
      sum += values[nz] * x[indices[nz]];
    }
    y[row] = sum;
    sum = 0;
  }

  // Interesting use of syncthreads to ensure all remaining tiles get processed
  // at the same time, possibly causing less thread divergence among the threads
  // in the same warp.
  __syncthreads();

  /// Process remaining tiles.
  for (auto row : config.remainder_tiles(map)) {
    for (auto nz : config.remainder_atoms(map)) {
      sum += values[nz] * x[indices[nz]];
    }
    /// Accumulate the remainder.
    if (sum != 0)
      atomicAdd(&(y[row]), sum);
  }
}

/**
 * @brief Sparse-Matrix Vector Multiplication API.
 *
 * @tparam index_t Type of column indices.
 * @tparam offset_t Type of row offsets.
 * @tparam type_t Type of values.
 * @param parameters CLI parameters.
 * @param csr CSR matrix (GPU).
 * @param x Input vector x (GPU).
 * @param y Output vector y (GPU).
 * @param stream CUDA stream.
 */
template <typename index_t, typename offset_t, typename type_t>
void spmv(parameters_t& parameters,
          csr_t<index_t, offset_t, type_t>& csr,
          vector_t<type_t>& x,
          vector_t<type_t>& y,
          cudaStream_t stream = 0) {
  // Create a schedule.
  constexpr std::size_t block_size = 128;

  /// Set-up kernel launch parameters and run the kernel.

  /// Launch 2 x (SM Count) number of blocks.
  /// Weirdly enough, a really high number here might cause it to fail.
  loops::device::properties_t props;
  std::size_t grid_size = 2 * props.multi_processor_count();

  launch::non_cooperative(
      stream, work_oriented_spmv<block_size, index_t, offset_t, type_t>,
      grid_size, block_size, csr.rows, csr.cols, csr.nnzs,
      csr.offsets.data().get(), csr.indices.data().get(),
      csr.values.data().get(), x.data().get(), y.data().get());

  cudaStreamSynchronize(stream);
}

int main(int argc, char** argv) {
  using index_t = int;
  using offset_t = int;
  using type_t = float;

  // ... I/O parameters, mtx, etc.
  parameters_t parameters(argc, argv);

  csr_t<index_t, offset_t, type_t> csr;
  matrix_market_t<index_t, offset_t, type_t> mtx;
  csr.from_coo(mtx.load(parameters.filename));

  // Input and output vectors.
  vector_t<type_t> x(csr.rows);
  vector_t<type_t> y(csr.rows);

  // Generate random numbers between [0, 1].
  generate::random::uniform_distribution(x.begin(), x.end(), 1, 10);

  // Run the benchmark.
  spmv(parameters, csr, x, y);

  // Validation.
  if (parameters.validate)
    cpu::validate(parameters, csr, x, y);
}