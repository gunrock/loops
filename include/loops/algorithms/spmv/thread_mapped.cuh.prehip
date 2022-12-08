/**
 * @file thread_mapped.cuh
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Sparse Matrix-Vector Multiplication kernels.
 * @version 0.1
 * @date 2022-02-03
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <loops/schedule.hxx>
#include <loops/container/formats.hxx>
#include <loops/container/vector.hxx>
#include <loops/util/launch.hxx>
#include <loops/util/device.hxx>
#include <loops/memory.hxx>
#include <iostream>

namespace loops {
namespace algorithms {
namespace spmv {

template <typename setup_t,
          typename index_t,
          typename offset_t,
          typename type_t>
__global__ void __thread_mapped(setup_t config,
                                const std::size_t rows,
                                const std::size_t cols,
                                const std::size_t nnz,
                                const offset_t* offsets,
                                const index_t* indices,
                                const type_t* values,
                                const type_t* x,
                                type_t* y) {
  /// Equivalent to:
  /// row = blockIdx.x * blockDim.x + threadIdx.x; (init)
  /// row < rows; (boundary condition)
  /// row += gridDim.x * blockDim.x. (step)
  for (auto row : config.tiles()) {
    type_t sum = 0;

    /// Equivalent to:
    /// for (offset_t nz = offset; nz < end; ++nz)
    for (auto nz : config.atoms(row)) {
      sum += values[nz] * x[indices[nz]];
    }

    // Output
    y[row] = sum;
  }
}

/**
 * @brief Sparse-Matrix Vector Multiplication API.
 *
 * @tparam index_t Type of column indices.
 * @tparam offset_t Type of row offsets.
 * @tparam type_t Type of values.
 * @param csr CSR matrix (GPU).
 * @param x Input vector x (GPU).
 * @param y Output vector y (GPU).
 * @param stream CUDA stream.
 */
template <typename index_t, typename offset_t, typename type_t>
void thread_mapped(csr_t<index_t, offset_t, type_t>& csr,
                   vector_t<type_t>& x,
                   vector_t<type_t>& y,
                   cudaStream_t stream = 0) {
  // Create a schedule.
  constexpr std::size_t block_size = 128;

  /// Set-up kernel launch parameters and run the kernel.

  // Create a schedule.
  using setup_t = schedule::setup<schedule::algorithms_t::thread_mapped, 1, 1,
                                  index_t, offset_t>;
  setup_t config(csr.offsets.data().get(), csr.rows, csr.nnzs);

  std::size_t grid_size = (csr.rows + block_size - 1) / block_size;
  launch::non_cooperative(
      stream, __thread_mapped<setup_t, index_t, offset_t, type_t>, grid_size,
      block_size, config, csr.rows, csr.cols, csr.nnzs,
      csr.offsets.data().get(), csr.indices.data().get(),
      csr.values.data().get(), x.data().get(), y.data().get());

  cudaStreamSynchronize(stream);
}

}  // namespace spmv
}  // namespace algorithms
}  // namespace loops