/**
 * @file coo_thread_mapped.cuh
 * @author Loops contributors
 * @brief Atomic-add SpMV on COO-formatted matrices.
 * @version 0.1
 * @date 2026-05-06
 *
 * Drives the standard @c thread_mapped schedule over @c layout::coo .
 * Because the COO layout exposes one tile per nonzero, each thread
 * handles exactly one NZ and atomic-adds its contribution into the
 * appropriate row of @c y . This is the canonical "scalar COO SpMV"
 * pattern.
 *
 * The output vector @c y must be zero-initialized by the caller (which
 * @c loops::vector_t<type_t>(n) already guarantees).
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <loops/schedule.hxx>
#include <loops/util/math.hxx>
#include <loops/container/layout.hxx>
#include <loops/container/coo.hxx>
#include <loops/container/vector.hxx>
#include <loops/util/launch.hxx>
#include <loops/util/device.hxx>
#include <loops/util/timer.hxx>
#include <loops/memory.hxx>

namespace loops {
namespace algorithms {
namespace spmv {

template <typename setup_t, typename index_t, typename type_t>
__global__ void __coo_thread_mapped(setup_t config,
                                    const index_t* row_indices,
                                    const index_t* col_indices,
                                    const type_t* values,
                                    const type_t* x,
                                    type_t* y) {
  for (auto t : config.tiles()) {
    for (auto atom : config.atoms(t)) {
      const index_t row = row_indices[atom];
      const index_t col = col_indices[atom];
      atomicAdd(&y[row], values[atom] * x[col]);
    }
  }
}

/**
 * @brief Thread-mapped SpMV for COO inputs (one thread per nonzero).
 *
 * @tparam index_t Type of the row/column indices.
 * @tparam type_t  Type of the non-zero values.
 */
template <typename index_t, typename type_t>
util::timer_t coo_thread_mapped(coo_t<index_t, type_t>& coo,
                                vector_t<type_t>& x,
                                vector_t<type_t>& y,
                                cudaStream_t stream = 0) {
  using tile_id_t = index_t;
  using atom_id_t = index_t;
  using coo_layout_t = layout::coo<tile_id_t, atom_id_t>;

  using setup_t =
      schedule::setup<schedule::algorithms_t::thread_mapped, 1, 1, tile_id_t,
                      atom_id_t, std::size_t, std::size_t, coo_layout_t>;

  coo_layout_t lay(static_cast<atom_id_t>(coo.nnzs));
  setup_t config(lay);

  constexpr std::size_t block_size = 128;
  std::size_t grid_size =
      math::ceil_div(static_cast<std::size_t>(coo.nnzs), block_size);

  util::timer_t timer;
  timer.start();
  launch::non_cooperative(
      stream, __coo_thread_mapped<setup_t, index_t, type_t>, grid_size,
      block_size, config, coo.row_indices.data().get(),
      coo.col_indices.data().get(), coo.values.data().get(), x.data().get(),
      y.data().get());
  cudaStreamSynchronize(stream);
  timer.stop();
  return timer;
}

}  // namespace spmv
}  // namespace algorithms
}  // namespace loops
