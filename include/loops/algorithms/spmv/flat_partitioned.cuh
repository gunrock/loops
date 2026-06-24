/**
 * @file flat_partitioned.cuh
 * @author Loops contributors
 * @brief SpMV driven by a flat-uniform-occupancy partitioned layout.
 * @version 0.1
 * @date 2026-05-06
 *
 * Demonstrates that a *partitioner* (in this case
 * @c layout::flat_uniform_occupancy<K, csr_layout> ) is just another
 * layout view as far as the schedule is concerned. The schedule code is
 * the standard @c thread_mapped specialization; the only difference from
 * a vanilla CSR run is:
 *
 * - The "tile" the schedule sees is a fixed-size group of @c K atoms,
 *   *not* a CSR row. This means a single tile can span multiple rows.
 * - Output addressing is therefore atomic-add. The kernel recovers the
 *   true CSR row of each atom via @c partitioner.base().tile_of(atom) ,
 *   which is an O(log num_rows) binary search over the offsets array.
 *
 * The choice of @c K trades load-imbalance against atomic contention:
 * small @c K hands more, smaller tiles to the scheduler (better balance,
 * more atomics); large @c K reduces atomic traffic but lets long rows
 * dominate. A reasonable default is @c K=8 for sparse-balanced matrices.
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <loops/schedule.hxx>
#include <loops/util/math.hxx>
#include <loops/container/layout.hxx>
#include <loops/container/csr.hxx>
#include <loops/container/vector.hxx>
#include <loops/util/launch.hxx>
#include <loops/util/device.hxx>
#include <loops/algorithms/spmv/launch_box.hxx>
#include <loops/util/timer.hxx>
#include <loops/memory.hxx>

namespace loops {
namespace algorithms {
namespace spmv {

template <typename setup_t, typename index_t, typename type_t>
__global__ void __flat_partitioned(setup_t config,
                                   const index_t* indices,
                                   const type_t* values,
                                   const type_t* x,
                                   type_t* y) {
  const auto& part = config.layout();
  const auto& base = part.base();
  for (auto t : config.tiles()) {
    for (auto atom : config.atoms(t)) {
      const auto row = base.tile_of(atom);
      atomicAdd(&y[row], values[atom] * x[indices[atom]]);
    }
  }
}

/**
 * @brief SpMV via a flat-uniform-occupancy partitioner over CSR.
 *
 * @tparam K        Atoms per tile (compile-time constant).
 * @tparam index_t  Column-index type.
 * @tparam offset_t Row-offset type.
 * @tparam type_t   Value type.
 */
template <std::size_t K = 8,
          typename index_t,
          typename offset_t,
          typename type_t>
util::timer_t flat_partitioned(csr_t<index_t, offset_t, type_t>& csr,
                               vector_t<type_t>& x,
                               vector_t<type_t>& y,
                               cudaStream_t stream = 0) {
  using tile_id_t = index_t;
  using atom_id_t = offset_t;
  using base_layout_t = layout::csr<tile_id_t, atom_id_t>;
  using lay_t = layout::flat_uniform_occupancy<K, base_layout_t>;

  using setup_t =
      schedule::setup<schedule::algorithms_t::thread_mapped, 1, 1, tile_id_t,
                      atom_id_t, std::size_t, std::size_t, lay_t>;

  base_layout_t base(csr.offsets.data().get(), static_cast<tile_id_t>(csr.rows),
                     static_cast<atom_id_t>(csr.nnzs));
  lay_t partitioned(base);
  setup_t config(partitioned);

  constexpr std::size_t block_size = launch_t<type_t>::block_size;
  std::size_t grid_size = math::ceil_div(
      static_cast<std::size_t>(partitioned.num_tiles()), block_size);

  // y is assumed zero-initialized (vector_t<type_t>(n) default-constructs
  // each element to 0). The kernel atomic-adds into y[row].
  util::timer_t timer;
  timer.start();
  launch::non_cooperative(stream, __flat_partitioned<setup_t, index_t, type_t>,
                          grid_size, block_size, config,
                          csr.indices.data().get(), csr.values.data().get(),
                          x.data().get(), y.data().get());
  cudaStreamSynchronize(stream);
  timer.stop();
  return timer;
}

}  // namespace spmv
}  // namespace algorithms
}  // namespace loops
