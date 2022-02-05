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
  block_mapped,   /// < Block mapped scheduling algorithm.
  bucketing,      /// < Bucketing scheduling algorithm.
};

template <algroithms_t scheme, typename atoms_t, typename atom_size_t>
class atom_traits;

template <algroithms_t scheme, typename tiles_t, typename tile_size_t>
class tile_traits;

/**
 * @brief
 *
 * @tparam scheme
 * @tparam threads_per_block
 * @tparam tiles_t
 * @tparam atoms_t
 * @tparam tile_size_t
 * @tparam atom_size_t
 */
template <algroithms_t scheme,
          typename tiles_t,
          typename atoms_t,
          typename tile_size_t = std::size_t,
          typename atom_size_t = std::size_t>
class setup;

}  // namespace schedule
}  // namespace loops

#include <loops/schedule/thread_mapped.hxx>
#include <loops/schedule/block_mapped.hxx>