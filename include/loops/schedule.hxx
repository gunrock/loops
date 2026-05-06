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

/**
 * @brief Traits describing the global atom range.
 *
 * The atoms_iterator_t/size pair describes the underlying flat atom storage
 * (e.g., the column-indices array for a sparse matrix). This default
 * implementation is shared across every scheduling algorithm; specializations
 * may be added later if a particular schedule needs algorithm-specific state.
 *
 * @tparam scheme The scheduling algorithm (currently unused, reserved for
 *                future per-algorithm overrides).
 * @tparam atoms_type Type of the atoms (e.g., column index type).
 * @tparam atom_size_type Type used to count atoms.
 */
template <algorithms_t scheme, typename atoms_type, typename atom_size_type>
class atom_traits {
 public:
  using atoms_t = atoms_type;
  using atoms_iterator_t = atoms_t*;
  using atom_size_t = atom_size_type;

  __host__ __device__ atom_traits() : size_(0), atoms_(nullptr) {}
  __host__ __device__ atom_traits(atom_size_t size)
      : size_(size), atoms_(nullptr) {}
  __host__ __device__ atom_traits(atom_size_t size, atoms_iterator_t atoms)
      : size_(size), atoms_(atoms) {}

  __host__ __device__ atom_size_t size() const { return size_; }
  __host__ __device__ atoms_iterator_t begin() { return atoms_; }
  __host__ __device__ atoms_iterator_t end() { return atoms_ + size_; }

 private:
  atom_size_t size_;
  atoms_iterator_t atoms_;
};

/**
 * @brief Traits describing the tile range.
 *
 * The tiles_iterator_t/size pair describes the per-tile metadata used by the
 * schedule. For CSR-shaped inputs the iterator points at the row-offset array
 * and the schedules use the prefix-sum semantics directly. This default
 * implementation is shared across every scheduling algorithm.
 *
 * @tparam scheme The scheduling algorithm (currently unused, reserved for
 *                future per-algorithm overrides).
 * @tparam tiles_type Type of the tiles (e.g., row offset type).
 * @tparam tile_size_type Type used to count tiles.
 */
template <algorithms_t scheme, typename tiles_type, typename tile_size_type>
class tile_traits {
 public:
  using tiles_t = tiles_type;
  using tiles_iterator_t = tiles_t*;
  using tile_size_t = tile_size_type;

  __host__ __device__ tile_traits() : size_(0), tiles_(nullptr) {}
  __host__ __device__ tile_traits(tile_size_t size, tiles_iterator_t tiles)
      : size_(size), tiles_(tiles) {}

  __host__ __device__ tile_size_t size() const { return size_; }
  __host__ __device__ tiles_iterator_t begin() { return tiles_; }
  __host__ __device__ tiles_iterator_t end() { return tiles_ + size_; }

 private:
  tile_size_t size_;
  tiles_iterator_t tiles_;
};

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
