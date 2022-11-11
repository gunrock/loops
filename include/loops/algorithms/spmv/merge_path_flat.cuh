/**
 * @file work_oriented.cuh
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Sparse Matrix-Vector Multiplication example.
 * @version 0.1
 * @date 2022-02-03
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <loops/schedule.hxx>
#include <loops/util/math.hxx>
#include <loops/container/formats.hxx>
#include <loops/container/vector.hxx>
#include <loops/util/launch.hxx>
#include <loops/util/device.hxx>
#include <loops/memory.hxx>
#include <iostream>

namespace loops {
namespace algorithms {
namespace spmv {

/**
 * @brief Flat Merge-Path SpMV kernel.
 *
 * @tparam threads_per_block Number of threads per block.
 * @tparam items_per_thread Number of items per thread to process.
 * @tparam index_t Type of column indices.
 * @tparam offset_t Type of row offsets.
 * @tparam type_t Type of values.
 */
template <std::size_t threads_per_block,
          std::size_t items_per_thread,
          typename index_t,
          typename offset_t,
          typename type_t>
__global__ void __launch_bounds__(threads_per_block, 2)
    __merge_path_flat(std::size_t rows,
                      std::size_t cols,
                      std::size_t nnz,
                      offset_t* offsets,
                      index_t* indices,
                      const type_t* values,
                      const type_t* x,
                      type_t* y) {
  using setup_t = schedule::setup<schedule::algorithms_t::merge_path_flat,
                                  threads_per_block, items_per_thread, index_t,
                                  offset_t, std::size_t, std::size_t>;

  /// Allocate temporary storage for the schedule.
  using storage_t = typename setup_t::storage_t;
  __shared__ storage_t temporary_storage;

  /// Construct the schedule.
  setup_t config(temporary_storage, offsets, rows, nnz);
  auto map = config.init();

  if (!config.is_valid_accessor(map))
    return;

  type_t running_total = 0.0f;
  /// Flat Merge-Path loop from 0..items_per_thread.
  for (auto item : config.virtual_idx()) {
    auto nz = config.atom_idx(item, map);
    auto row = config.tile_idx(map);
    type_t nonzero = values[nz] * x[indices[nz]];
    if (config.atoms_counting_it[map.second] <
        temporary_storage.tile_end_offset[map.first]) {
      atomicAdd(&(y[row]), nonzero);
      map.second++;
    } else {
      running_total = 0.0f;
      map.first++;
    }
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
void merge_path_flat(csr_t<index_t, offset_t, type_t>& csr,
                     vector_t<type_t>& x,
                     vector_t<type_t>& y,
                     cudaStream_t stream = 0) {
  // Create a schedule.
  constexpr std::size_t block_size = 128;
  constexpr std::size_t items_per_thread = 3;

  /// Set-up kernel launch parameters and run the kernel.
  int max_dim_x;
  int num_merge_tiles =
      math::ceil_div(csr.rows + csr.nnzs, block_size * items_per_thread);
  cudaDeviceGetAttribute(&max_dim_x, cudaDevAttrMaxGridDimX, 0);

  // TODO: Fix this later.
  dim3 grid_size(num_merge_tiles, 1, 1);
  launch::non_cooperative(stream,
                          __merge_path_flat<block_size, items_per_thread,
                                            index_t, offset_t, type_t>,
                          grid_size, block_size, csr.rows, csr.cols, csr.nnzs,
                          csr.offsets.data().get(), csr.indices.data().get(),
                          csr.values.data().get(), x.data().get(),
                          y.data().get());
  cudaStreamSynchronize(stream);
}

}  // namespace spmv
}  // namespace algorithms
}  // namespace loops