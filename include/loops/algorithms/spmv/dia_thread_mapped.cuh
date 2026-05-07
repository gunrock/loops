/**
 * @file dia_thread_mapped.cuh
 * @author Loops contributors
 * @brief DIA SpMV using the thread_mapped schedule.
 * @version 0.1
 * @date 2026-05-06
 *
 * One thread per row. Each thread sweeps every stored diagonal and
 * accumulates into a single register, then writes the result to @c y[r] .
 * Out-of-range column indices are clamped to a valid value and the
 * corresponding diagonal entry will be a padded zero, so the multiply
 * contributes nothing.
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <loops/schedule.hxx>
#include <loops/util/math.hxx>
#include <loops/container/layout.hxx>
#include <loops/container/dia.hxx>
#include <loops/container/vector.hxx>
#include <loops/util/launch.hxx>
#include <loops/util/device.hxx>
#include <loops/util/timer.hxx>
#include <loops/memory.hxx>

#include <cstddef>

namespace loops {
namespace algorithms {
namespace spmv {

template <typename setup_t, typename index_t, typename type_t>
__global__ void __dia_thread_mapped(setup_t config,
                                    std::size_t cols,
                                    std::size_t stride,
                                    std::size_t num_diagonals,
                                    const index_t* diag_offsets,
                                    const type_t* values,
                                    const type_t* x,
                                    type_t* y) {
  for (auto r : config.tiles()) {
    type_t acc = type_t{0};
    for (auto a : config.atoms(r)) {
      const std::size_t d = a - r * num_diagonals;
      const auto off = diag_offsets[d];
      const long c = static_cast<long>(r) + static_cast<long>(off);
      if (c >= 0 && c < static_cast<long>(cols)) {
        acc += values[d * stride + r] * x[c];
      }
    }
    y[r] = acc;
  }
}

/**
 * @brief Thread-mapped SpMV for DIA inputs (one thread per row).
 *
 * @tparam index_t  Type of the diagonal-offsets (must be signed!).
 * @tparam offset_t Type used by the layout for atom-ids.
 * @tparam type_t   Type of the non-zero values.
 */
template <typename index_t, typename offset_t, typename type_t>
util::timer_t dia_thread_mapped(dia_t<index_t, offset_t, type_t>& dia,
                                vector_t<type_t>& x,
                                vector_t<type_t>& y,
                                cudaStream_t stream = 0) {
  using tile_id_t = std::size_t;
  using atom_id_t = std::size_t;
  using dia_layout_t = layout::dia<tile_id_t, atom_id_t>;

  using setup_t =
      schedule::setup<schedule::algorithms_t::thread_mapped, 1, 1, tile_id_t,
                      atom_id_t, std::size_t, std::size_t, dia_layout_t>;

  dia_layout_t lay(static_cast<tile_id_t>(dia.rows),
                   static_cast<atom_id_t>(dia.num_diagonals));
  setup_t config(lay);

  constexpr std::size_t block_size = 128;
  std::size_t grid_size =
      math::ceil_div(static_cast<std::size_t>(dia.rows), block_size);

  util::timer_t timer;
  timer.start();
  launch::non_cooperative(stream, __dia_thread_mapped<setup_t, index_t, type_t>,
                          grid_size, block_size, config, dia.cols, dia.stride,
                          dia.num_diagonals, dia.diag_offsets.data().get(),
                          dia.values.data().get(), x.data().get(),
                          y.data().get());
  cudaStreamSynchronize(stream);
  timer.stop();
  return timer;
}

}  // namespace spmv
}  // namespace algorithms
}  // namespace loops
