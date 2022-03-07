/**
 * @file schedule.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
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
enum algroithms_t {
  work_oriented,  /// < Work oriented scheduling algorithm.
  thread_mapped,  /// < Thread mapped scheduling algorithm.
  tile_mapped,    /// < Tile mapped scheduling algorithm.
  bucketing,      /// < Bucketing scheduling algorithm.
};

template <algroithms_t scheme, typename atoms_t, typename atom_size_t>
class atom_traits;

template <algroithms_t scheme, typename tiles_t, typename tile_size_t>
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
template <algroithms_t scheme,
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
#include <loops/schedule/tile_mapped.hxx>
#include <loops/schedule/work_oriented.hxx>