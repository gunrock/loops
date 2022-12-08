#include "hip/hip_runtime.h"
/**
 * @file thread_mapped.cuh
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Sparse Matrix-Matrix Multiplication kernels.
 * @version 0.1
 * @date 2022-02-03
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <loops/stride_ranges.hxx>
#include <loops/schedule.hxx>
#include <loops/container/formats.hxx>
#include <loops/container/vector.hxx>
#include <loops/container/matrix.cuh>
#include <loops/util/launch.hxx>
#include <loops/util/device.hxx>
#include <loops/memory.hxx>
#include <iostream>

namespace loops {
namespace algorithms {
namespace spmm {

template <typename setup_t,
          typename index_t,
          typename offset_t,
          typename type_t>
__global__ void __thread_mapped(setup_t config,
                                const std::size_t a_rows,
                                const std::size_t a_cols,
                                const std::size_t a_nnz,
                                const offset_t* offsets,
                                const index_t* indices,
                                const type_t* values,
                                const matrix_t<type_t> B,
                                matrix_t<type_t> C) {
  for (auto row : config.tiles()) {
    for (auto col :
         custom_stride_range(std::size_t(0), B.cols, std::size_t(1))) {
      type_t sum = 0;
      for (auto nz : config.atoms(row)) {
        sum += values[nz] * B(nz, col);
      }

      // Output
      C(row, col) = sum;
    }
  }
}

/**
 * @brief Sparse-Matrix Matrix Multiplication API.
 *
 * @tparam index_t Type of column indices.
 * @tparam offset_t Type of row offsets.
 * @tparam type_t Type of values.
 * @param csr CSR matrix (GPU).
 * @param n Number of columns in the B-matrix.
 * @param B Input matrix B (GPU).
 * @param C Output matrix C (GPU).
 * @param stream CUDA stream.
 */
template <typename index_t, typename offset_t, typename type_t>
void thread_mapped(csr_t<index_t, offset_t, type_t>& csr,
                   matrix_t<type_t>& B,
                   matrix_t<type_t>& C,
                   hipStream_t stream = 0) {
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
      csr.values.data().get(), B, C);

  hipStreamSynchronize(stream);
}

}  // namespace spmm
}  // namespace algorithms
}  // namespace loops