/**
 * @file schedule.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Header file for the schedule class.
 * @version 0.1
 * @date 2022-02-04
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <cstddef>

#include <loops/container/layout.hxx>
#include <loops/util/xpu.hxx>

namespace loops {
namespace schedule {

/**
 * @brief Load balancing algorithms.
 *
 */
enum algorithms_t {
  merge_path_flat,  /// < Merge-path flat scheduling algorithm.
  work_oriented,    /// < Work oriented scheduling algorithm.
  thread_mapped,    /// < Thread mapped scheduling algorithm.
  group_mapped,     /// < Group mapped scheduling algorithm.
  bucketing,        /// < Bucketing scheduling algorithm.
};

/**
 * @brief Schedule's setup interface.
 *
 * Schedules consume the workload through a layout view (see
 * `loops/container/layout.hxx`). The 5+ method layout contract is what makes
 * the schedules format-generic: the same setup specialization handles CSR,
 * COO, ELL, or any user-defined layout that satisfies the contract.
 *
 * The `layout_type` template parameter defaults to
 * `layout::csr<tiles_t, atoms_t>` for backward compatibility with the
 * original CSR-only API; pass a different layout view to swap formats.
 *
 * @tparam scheme            The scheduling algorithm.
 * @tparam threads_per_block Number of threads per block.
 * @tparam threads_per_tile  Number of threads per tile.
 * @tparam tiles_t           Tile-id type (e.g., row id).
 * @tparam atoms_t           Atom-id type (e.g., flat nnz position).
 * @tparam tile_size_t       Type of the tile size (default: std::size_t).
 * @tparam atom_size_t       Type of the atom size (default: std::size_t).
 * @tparam layout_type       Layout view (default: layout::csr).
 */
template <algorithms_t scheme,
          std::size_t threads_per_block,
          std::size_t threads_per_tile,
          typename tiles_t,
          typename atoms_t,
          typename tile_size_t = std::size_t,
          typename atom_size_t = std::size_t,
          typename layout_type = layout::csr<tiles_t, atoms_t>>
class setup;

}  // namespace schedule
}  // namespace loops

#include <loops/schedule/thread_mapped.hxx>
// group_mapped relies on NVIDIA cooperative-groups (experimental
// block_tile_memory, exclusive_scan); no HIP equivalent yet, so it is
// CUDA-only. HIP builds skip it (and the matching SpMV example).
#if !defined(LOOPS_BACKEND_HIP)
#include <loops/schedule/group_mapped.hxx>
#endif
#include <loops/schedule/work_oriented.hxx>
#include <loops/schedule/merge_path_flat.hxx>
