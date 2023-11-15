/**
 * @file thread_mapped.cuh
 * @author 
 * @brief SpGEMM kernels.
 * @version 0.1
 * @date 2023-10-17
 *
 * @copyright Copyright (c) 2023
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
namespace spgemm {

template <typename setup_t,
          typename index_t,
          typename offset_t,
          typename type_t>
__global__ void __thread_mapped(setup_t config,
                                const std::size_t a_rows,
                                const std::size_t a_cols,
                                const std::size_t a_nnz,
                                const offset_t* a_offsets,
                                const index_t* a_indices,
                                const type_t* a_values,
                                const std::size_t b_rows,
                                const std::size_t b_cols,
                                const std::size_t b_nnz,
                                const offset_t* b_offsets,
                                const index_t* b_indices,
                                const type_t* b_values,
                                matrix_t<type_t> C) {
  for (auto mm : config.tiles()) {
    for (auto nn :
         custom_stride_range(std::size_t(0), b_cols, std::size_t(1))) {
      type_t sum = 0;
      for (auto nz : config.atoms(mm)) {
        auto kk_a = a_indices[nz];
          for (auto nz_b = b_offsets[nn]; nz_b < b_offsets[nn + 1]; ++nz_b) {
            if (kk_a == b_indices[nz_b]) {
              sum += a_values[nz] * b_values[nz_b];
            }
          }
      }

      // Output - C in sparse format (try COO)
      C(mm, nn) = sum;
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
                   csc_t<index_t, offset_t, type_t>& csc,
                   matrix_t<type_t>& C,
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
      csr.values.data().get(), csc.rows, csc.cols, csc.nnzs,
      csc.offsets.data().get(), csc.indices.data().get(),
      csc.values.data().get(), C);

  cudaStreamSynchronize(stream);
}

}  // namespace spgemm
}  // namespace algorithms
}  // namespace loops