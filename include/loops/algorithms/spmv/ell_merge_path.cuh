/**
 * @file ell_merge_path.cuh
 * @author Loops contributors
 * @brief Merge-path SpMV on ELL-formatted matrices.
 *
 * Stresses the layout abstraction harder than thread_mapped: merge_path_flat
 * binary-searches over `layout.tile_end_iter()`, which for ELL is a
 * `thrust::transform_iterator` over a counting iterator (it materializes
 * `(i + 1) * pitch` lazily). If this kernel produces matching results, the
 * layout contract is genuinely format-generic.
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <loops/schedule.hxx>
#include <loops/util/math.hxx>
#include <loops/container/layout.hxx>
#include <loops/container/ell.hxx>
#include <loops/container/vector.hxx>
#include <loops/util/launch.hxx>
#include <loops/util/device.hxx>
#include <loops/util/timer.hxx>
#include <loops/memory.hxx>

#include <cub/block/block_scan.cuh>

namespace loops {
namespace algorithms {
namespace spmv {

template <std::size_t threads_per_block,
          std::size_t items_per_thread,
          typename meta_t,
          typename setup_t,
          typename layout_t,
          typename index_t,
          typename type_t>
__global__ void __launch_bounds__(int(threads_per_block))
    __ell_merge_path(meta_t meta,
                     layout_t lay,
                     const index_t* indices,
                     const type_t* values,
                     const type_t* x,
                     type_t* y) {
  using storage_t = typename setup_t::storage_t;
  __shared__ storage_t temporary_storage;

  setup_t config(meta, temporary_storage, lay);
  auto map = config.init();

  if (!config.is_valid_accessor(map))
    return;

#pragma unroll
  for (auto item : config.virtual_idx()) {
    auto nz = config.atom_idx(item, map);
    auto row = config.tile_idx(map);
    const index_t col = indices[nz];
    type_t nonzero = (col >= 0) ? (values[nz] * x[col]) : type_t(0);
    if (config.atoms_counting_it[map.y] <
        temporary_storage.tile_end_offset[map.x]) {
      atomicAdd(&(y[row]), nonzero);
      map.y++;
    } else {
      map.x++;
    }
  }
}

/**
 * @brief Merge-path SpMV for ELL inputs.
 */
template <typename index_t, typename type_t>
util::timer_t ell_merge_path(ell_t<index_t, type_t>& ell,
                             vector_t<type_t>& x,
                             vector_t<type_t>& y,
                             cudaStream_t stream = 0) {
  using tile_id_t = index_t;
  using atom_id_t = index_t;
  using ell_layout_t = layout::ell<tile_id_t, atom_id_t>;

  constexpr std::size_t block_size = sizeof(type_t) > 4 ? 64 : 128;
  constexpr std::size_t items_per_thread = sizeof(type_t) > 4 ? 3 : 5;

  using preprocessor_t =
      schedule::merge_path::preprocess_t<block_size, items_per_thread,
                                         tile_id_t, atom_id_t, std::size_t,
                                         std::size_t, ell_layout_t>;
  using setup_t =
      schedule::setup<schedule::algorithms_t::merge_path_flat, block_size,
                      items_per_thread, tile_id_t, atom_id_t, std::size_t,
                      std::size_t, ell_layout_t>;

  ell_layout_t lay(static_cast<tile_id_t>(ell.rows),
                   static_cast<atom_id_t>(ell.pitch));
  preprocessor_t meta(lay, stream);

  int max_dim_x;
  int num_merge_tiles = math::ceil_div(ell.rows + ell.rows * ell.pitch,
                                       block_size * items_per_thread);
  int device_ordinal = device::get();
  cudaDeviceGetAttribute(&max_dim_x, cudaDevAttrMaxGridDimX, device_ordinal);

  util::timer_t timer;
  timer.start();

  int within_bounds = min(num_merge_tiles, max_dim_x);
  int overflow = math::ceil_div(num_merge_tiles, max_dim_x);
  dim3 grid_size(within_bounds, overflow, 1);

  // y is assumed zero-initialized (vector_t<type_t>(n) default-constructs
  // each element to 0). The kernel uses atomicAdd and relies on this.

  launch::non_cooperative(
      stream,
      __ell_merge_path<block_size, items_per_thread, preprocessor_t, setup_t,
                       ell_layout_t, index_t, type_t>,
      grid_size, block_size, meta, lay, ell.indices.data().get(),
      ell.values.data().get(), x.data().get(), y.data().get());
  cudaStreamSynchronize(stream);
  timer.stop();

  return timer;
}

}  // namespace spmv
}  // namespace algorithms
}  // namespace loops
