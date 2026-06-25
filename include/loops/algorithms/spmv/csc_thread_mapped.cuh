/**
 * @file csc_thread_mapped.cuh
 * @author Loops contributors
 * @brief Atomic-add SpMV on CSC-formatted matrices.
 * @version 0.1
 * @date 2026-05-06
 *
 * One thread per column, walking the column's nonzeros and atomic-adding
 * @c values[a] @c * @c x[col] into @c y[row_indices[a]] . CSC's column-major
 * storage means a single tile (= column) hits multiple rows of @c y , so
 * atomics are unavoidable for thread-level parallelism. (A more
 * sophisticated kernel could group_mapped or merge_path_flat over the
 * same layout - the schedule contract is satisfied identically.)
 *
 * The output vector @c y must be zero-initialized by the caller.
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <loops/schedule.hxx>
#include <loops/util/math.hxx>
#include <loops/container/layout.hxx>
#include <loops/container/csc.hxx>
#include <loops/container/vector.hxx>
#include <loops/util/launch.hxx>
#include <loops/util/device.hxx>
#include <loops/util/timer.hxx>
#include <loops/memory.hxx>

namespace loops {
namespace algorithms {
namespace spmv {

template <typename setup_t, typename index_t, typename type_t>
__global__ void __csc_thread_mapped(setup_t config,
                                    const index_t* row_indices,
                                    const type_t* values,
                                    const type_t* x,
                                    type_t* y) {
  for (auto col : config.tiles()) {
    const type_t x_col = x[col];
    for (auto atom : config.atoms(col)) {
      atomicAdd(&y[row_indices[atom]], values[atom] * x_col);
    }
  }
}

/**
 * @brief Thread-mapped SpMV for CSC inputs (one thread per column).
 *
 * @tparam index_t  Type of the row indices.
 * @tparam offset_t Type of the column offsets.
 * @tparam type_t   Type of the non-zero values.
 */
template <typename index_t, typename offset_t, typename type_t>
util::timer_t csc_thread_mapped(csc_t<index_t, offset_t, type_t>& csc,
                                vector_t<type_t>& x,
                                vector_t<type_t>& y,
                                xpu::stream_t stream = 0) {
  using tile_id_t = index_t;
  using atom_id_t = offset_t;
  using csc_layout_t = layout::csc<tile_id_t, atom_id_t>;

  using setup_t =
      schedule::setup<schedule::algorithms_t::thread_mapped, 1, 1, tile_id_t,
                      atom_id_t, std::size_t, std::size_t, csc_layout_t>;

  csc_layout_t lay(csc.offsets.data().get(), static_cast<tile_id_t>(csc.cols),
                   static_cast<atom_id_t>(csc.nnzs));
  setup_t config(lay);

  constexpr std::size_t block_size = 128;
  std::size_t grid_size =
      math::ceil_div(static_cast<std::size_t>(csc.cols), block_size);

  util::timer_t timer;
  timer.start();
  launch::non_cooperative(stream, __csc_thread_mapped<setup_t, index_t, type_t>,
                          grid_size, block_size, config,
                          csc.indices.data().get(), csc.values.data().get(),
                          x.data().get(), y.data().get());
  xpu::stream_synchronize(stream);
  timer.stop();
  return timer;
}

}  // namespace spmv
}  // namespace algorithms
}  // namespace loops
