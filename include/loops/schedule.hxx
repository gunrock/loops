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

template <algorithms_t scheme, typename atoms_t, typename atom_size_t>
class atom_traits;

template <algorithms_t scheme, typename tiles_t, typename tile_size_t>
class tile_traits;

/**
 * @brief Schedule's setup interface.
 *
 * @tparam scheme The scheduling algorithm.
 * @tparam threads_per_block Number of threads per block.
 * @tparam threads_per_tile Number of threads per tile.
 * @tparam tiles_t Type of the tiles.
 * @tparam atoms_t Type of the atoms.
 * @tparam tile_size_t Type of the tile size (default: std::size_t).
 * @tparam atom_size_t Type of the atom size (default: std::size_t).
 */
template <algorithms_t scheme,
          std::size_t threads_per_block,
          std::size_t threads_per_tile,
          typename tiles_t,
          typename atoms_t,
          typename tile_size_t = std::size_t,
          typename atom_size_t = std::size_t>
class setup;

}  // namespace schedule
}  // namespace loops

#include <loops/schedule/thread_mapped.hxx>
#include <loops/schedule/group_mapped.hxx>
#include <loops/schedule/work_oriented.hxx>
#include <loops/schedule/merge_path_flat.hxx>