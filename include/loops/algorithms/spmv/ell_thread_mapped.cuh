/**
 * @file ell_thread_mapped.cuh
 * @author Loops contributors
 * @brief Thread-mapped SpMV on ELL-formatted matrices.
 *
 * Demonstrates that the existing thread-mapped schedule (originally written
 * for CSR) works unchanged on a non-CSR format once it is described by a
 * layout view satisfying the contract in `loops/container/layout.hxx`.
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <loops/schedule.hxx>
#include <loops/container/layout.hxx>
#include <loops/container/ell.hxx>
#include <loops/container/vector.hxx>
#include <loops/util/launch.hxx>
#include <loops/util/device.hxx>
#include <loops/memory.hxx>

namespace loops {
namespace algorithms {
namespace spmv {

template <typename setup_t, typename index_t, typename type_t>
__global__ void __ell_thread_mapped(setup_t config,
                                    const index_t* indices,
                                    const type_t* values,
                                    const type_t* x,
                                    type_t* y) {
  for (auto row : config.tiles()) {
    type_t sum = 0;
    for (auto atom : config.atoms(row)) {
      const index_t col = indices[atom];
      if (col >= 0)  // skip ELL padding
        sum += values[atom] * x[col];
    }
    y[row] = sum;
  }
}

/**
 * @brief Thread-mapped SpMV for ELL inputs.
 *
 * @param ell ELL matrix on the device.
 * @param x   Input dense vector (size cols).
 * @param y   Output dense vector (size rows).
 */
template <typename index_t, typename type_t>
void ell_thread_mapped(ell_t<index_t, type_t>& ell,
                       vector_t<type_t>& x,
                       vector_t<type_t>& y,
                       cudaStream_t stream = 0) {
  using tile_id_t = index_t;
  using atom_id_t = index_t;
  using ell_layout_t = layout::ell<tile_id_t, atom_id_t>;
  using setup_t = schedule::setup<schedule::algorithms_t::thread_mapped, 1, 1,
                                  tile_id_t, atom_id_t,
                                  std::size_t, std::size_t,
                                  ell_layout_t>;

  ell_layout_t lay(static_cast<tile_id_t>(ell.rows),
                   static_cast<atom_id_t>(ell.pitch));
  setup_t config(lay);

  constexpr std::size_t block_size = 128;
  std::size_t grid_size = (ell.rows + block_size - 1) / block_size;
  launch::non_cooperative(
      stream, __ell_thread_mapped<setup_t, index_t, type_t>, grid_size,
      block_size, config, ell.indices.data().get(), ell.values.data().get(),
      x.data().get(), y.data().get());

  cudaStreamSynchronize(stream);
}

}  // namespace spmv
}  // namespace algorithms
}  // namespace loops
