/**
 * @file bcsr_thread_mapped.cuh
 * @author Loops contributors
 * @brief Block CSR SpMV using the thread_mapped schedule.
 * @version 0.1
 * @date 2026-05-06
 *
 * One thread per block-row. Each thread iterates over its stored
 * @c R-by-C dense blocks (driven by @c layout::bcsr ) and accumulates
 * @c R partial sums in registers, then writes them out to @c y . No
 * atomics are needed because each block-row owns @c R consecutive output
 * rows; the writes are local to the thread.
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <loops/schedule.hxx>
#include <loops/util/math.hxx>
#include <loops/container/layout.hxx>
#include <loops/container/bcsr.hxx>
#include <loops/container/vector.hxx>
#include <loops/util/launch.hxx>
#include <loops/util/device.hxx>
#include <loops/util/timer.hxx>
#include <loops/memory.hxx>

#include <cstddef>

namespace loops {
namespace algorithms {
namespace spmv {

template <std::size_t R,
          std::size_t C,
          typename setup_t,
          typename index_t,
          typename type_t>
__global__ void __bcsr_thread_mapped(setup_t config,
                                     std::size_t rows,
                                     const index_t* block_col_indices,
                                     const type_t* values,
                                     const type_t* x,
                                     type_t* y) {
  for (auto br : config.tiles()) {
    type_t acc[R];
#pragma unroll
    for (std::size_t i = 0; i < R; ++i) acc[i] = type_t{0};

    for (auto b : config.atoms(br)) {
      const std::size_t bc = static_cast<std::size_t>(block_col_indices[b]);
      const type_t* block = values + b * R * C;
#pragma unroll
      for (std::size_t i = 0; i < R; ++i) {
#pragma unroll
        for (std::size_t j = 0; j < C; ++j) {
          acc[i] += block[i * C + j] * x[bc * C + j];
        }
      }
    }

    const std::size_t row_base = static_cast<std::size_t>(br) * R;
#pragma unroll
    for (std::size_t i = 0; i < R; ++i) {
      const std::size_t row = row_base + i;
      if (row < rows) {
        y[row] = acc[i];
      }
    }
  }
}

/**
 * @brief Thread-mapped SpMV for BCSR inputs (one thread per block-row).
 *
 * The block dimensions @c R and @c C are inferred from the @c bcsr_t
 * template, so the user just calls @c bcsr_thread_mapped(bcsr, x, y) .
 *
 * @tparam R        Block height (rows per dense block).
 * @tparam C        Block width  (cols per dense block).
 * @tparam index_t  Type of the block-column-indices.
 * @tparam offset_t Type of the block-offsets.
 * @tparam type_t   Type of the non-zero values.
 */
template <std::size_t R,
          std::size_t C,
          typename index_t,
          typename offset_t,
          typename type_t>
util::timer_t bcsr_thread_mapped(bcsr_t<R, C, index_t, offset_t, type_t>& bcsr,
                                 vector_t<type_t>& x,
                                 vector_t<type_t>& y,
                                 cudaStream_t stream = 0) {
  using tile_id_t = index_t;
  using atom_id_t = offset_t;
  using bcsr_layout_t = layout::bcsr<tile_id_t, atom_id_t>;

  using setup_t =
      schedule::setup<schedule::algorithms_t::thread_mapped, 1, 1, tile_id_t,
                      atom_id_t, std::size_t, std::size_t, bcsr_layout_t>;

  bcsr_layout_t lay(bcsr.block_offsets.data().get(),
                    static_cast<tile_id_t>(bcsr.num_block_rows),
                    static_cast<atom_id_t>(bcsr.num_blocks));
  setup_t config(lay);

  constexpr std::size_t block_size = 128;
  std::size_t grid_size =
      math::ceil_div(static_cast<std::size_t>(bcsr.num_block_rows), block_size);

  util::timer_t timer;
  timer.start();
  launch::non_cooperative(
      stream,
      __bcsr_thread_mapped<R, C, setup_t, index_t, type_t>, grid_size,
      block_size, config, bcsr.rows, bcsr.block_col_indices.data().get(),
      bcsr.values.data().get(), x.data().get(), y.data().get());
  cudaStreamSynchronize(stream);
  timer.stop();
  return timer;
}

}  // namespace spmv
}  // namespace algorithms
}  // namespace loops
