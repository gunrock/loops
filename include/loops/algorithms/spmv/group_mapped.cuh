/**
 * @file group_mapped.cuh
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

template <std::size_t threads_per_block,
          typename index_t,
          typename offset_t,
          typename type_t>
__global__ void __launch_bounds__(threads_per_block, 2)
    __group_mapped(std::size_t rows,
                   std::size_t cols,
                   std::size_t nnz,
                   offset_t* offsets,
                   index_t* indices,
                   const type_t* values,
                   const type_t* x,
                   type_t* y) {
  using setup_t = schedule::setup<schedule::algorithms_t::tile_mapped,
                                  threads_per_block, 32, index_t, offset_t>;

  /// Allocate temporary storage for the schedule.
  using storage_t = typename setup_t::storage_t;
  __shared__ storage_t temporary_storage;

  /// Construct the schedule.
  setup_t config(temporary_storage, offsets, rows, nnz);
  auto p = config.partition();

  for (auto virtual_atom : config.atom_accessor(p)) {
    auto virtual_tile = config.tile_accessor(virtual_atom, p);

    if (!(config.is_valid_accessor(virtual_tile, p)))
      continue;

    auto row = config.tile_id(virtual_tile, p);

    auto nz_idx = config.atom_id(virtual_atom, row, virtual_tile, p);
    atomicAdd(&(y[row]), values[nz_idx] * x[indices[nz_idx]]);
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
void group_mapped(csr_t<index_t, offset_t, type_t>& csr,
                  vector_t<type_t>& x,
                  vector_t<type_t>& y,
                  cudaStream_t stream = 0) {
  // Create a schedule.
  constexpr std::size_t block_size = 128;

  /// Set-up kernel launch parameters and run the kernel.

  /// Traditional kernel launch, this is nice for tile mapped scheduling, which
  /// will allow blocks to be scheduled in and out as needed. And will rely on
  /// NVIDIA's hardware schedule to schedule the blocks efficiently.
  std::size_t grid_size = (csr.rows + block_size - 1) / block_size;
  launch::non_cooperative(
      stream, __group_mapped<block_size, index_t, offset_t, type_t>, grid_size,
      block_size, csr.rows, csr.cols, csr.nnzs, csr.offsets.data().get(),
      csr.indices.data().get(), csr.values.data().get(), x.data().get(),
      y.data().get());

  /// Cooperative kernel launch; requires a fixed number of blocks per grid to
  /// be launched, this number can be determined by using CUDA's occupancy API
  /// to figure out how many blocks will run concurrently at all times per SM.
  /// And then we simply loop over the entire work within the kernel.
  // launch::cooperative(stream, __group_mapped<block_size, index_t, offset_t,
  // type_t>,
  //                     grid_size, block_size, csr.rows, csr.cols, csr.nnzs,
  //                     csr.offsets.data().get(), csr.indices.data().get(),
  //                     csr.values.data().get(), x.data().get(),
  //                     y.data().get());

  cudaStreamSynchronize(stream);
}

}  // namespace spmv
}  // namespace algorithms
}  // namespace loops