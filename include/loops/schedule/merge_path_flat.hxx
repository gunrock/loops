/**
 * @file merge_path_flat.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Merge-Path Flat scheduling algorithm (map even-share of work to
 * threads.)
 * @version 0.1
 * @date 2022-03-07
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <loops/stride_ranges.hxx>
#include <loops/util/math.hxx>

#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/pair.h>
#include <thrust/execution_policy.h>

namespace loops {
namespace schedule {

/**
 * @brief Traits for Atom.
 *
 * @todo Implement an atom iterator, right now it is based on CSR only. Can be
 * abstracted very simply by allowing UDF iterators.
 *
 * @tparam atoms_type Type of the atoms.
 * @tparam atom_size_type Type of the atom size.
 */
template <typename atoms_type, typename atom_size_type>
class atom_traits<algorithms_t::merge_path_flat, atoms_type, atom_size_type> {
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
  __host__ __device__ atoms_iterator_t begin() { return atoms_; };
  __host__ __device__ atoms_iterator_t end() { return atoms_ + size_; };

 private:
  atom_size_t size_;
  atoms_iterator_t atoms_;
};

/**
 * @brief Traits for Tile.
 *
 * @todo Implement an tile iterator, right now it is based on CSR only. Can be
 * abstracted very simply by allowing UDF iterators.
 *
 * @tparam tiles_type Type of the tiles.
 * @tparam tile_size_type Type of the tile size (default: std::size_t).
 */
template <typename tiles_type, typename tile_size_type>
class tile_traits<algorithms_t::merge_path_flat, tiles_type, tile_size_type> {
 public:
  using tiles_t = tiles_type;
  using tiles_iterator_t = tiles_t*;
  using tile_size_t = tile_size_type;

  __host__ __device__ tile_traits() : size_(0), tiles_(nullptr) {}
  __host__ __device__ tile_traits(tile_size_t size, tiles_iterator_t tiles)
      : size_(size), tiles_(tiles) {}

  __host__ __device__ tile_size_t size() const { return size_; }
  __host__ __device__ tiles_iterator_t begin() { return tiles_; };
  __host__ __device__ tiles_iterator_t end() { return tiles_ + size_; };

 private:
  tile_size_t size_;
  tiles_iterator_t tiles_;
};

struct coordinate_t {
  int first;
  int second;
};

/**
 * @brief Work-oriented schedule's setup interface.
 *
 * @tparam THREADS_PER_BLOCK Threads per block.
 * @tparam ITEMS_PER_THREAD Number of Items per thread to process.
 * @tparam tiles_type Type of the tiles.
 * @tparam atoms_type Type of the atoms.
 * @tparam tile_size_type Type of the tile size.
 * @tparam atom_size_type Type of the atom size.
 */
template <std::size_t THREADS_PER_BLOCK,
          std::size_t ITEMS_PER_THREAD,
          typename tiles_type,
          typename atoms_type,
          typename tile_size_type,
          typename atom_size_type>
class setup<algorithms_t::merge_path_flat,
            THREADS_PER_BLOCK,
            ITEMS_PER_THREAD,
            tiles_type,
            atoms_type,
            tile_size_type,
            atom_size_type> : public tile_traits<algorithms_t::merge_path_flat,
                                                 tiles_type,
                                                 tile_size_type>,
                              public atom_traits<algorithms_t::merge_path_flat,
                                                 atoms_type,
                                                 atom_size_type> {
 public:
  using tiles_t = tiles_type;          /// Tile Type
  using atoms_t = atoms_type;          /// Atom Type
  using tiles_iterator_t = tiles_t*;   /// Tile Iterator Type
  using atoms_iterator_t = tiles_t*;   /// Atom Iterator Type
  using tile_size_t = tile_size_type;  /// Tile Size Type
  using atom_size_t = atom_size_type;  /// Atom Size Type

  using tile_traits_t =
      tile_traits<algorithms_t::merge_path_flat, tiles_type, tile_size_type>;
  using atom_traits_t =
      atom_traits<algorithms_t::merge_path_flat, atoms_type, atom_size_type>;

  enum : unsigned int {
    threads_per_block = THREADS_PER_BLOCK,
    items_per_thread = ITEMS_PER_THREAD,
    items_per_tile = threads_per_block * items_per_thread,
  };

  /// Shared memory type required by this thread block.
  struct storage_t {
    coordinate_t tile_coords[2];
    tiles_t tile_end_offset[items_per_thread + items_per_tile + 1];
  };

  storage_t& buffer;

  thrust::counting_iterator<atoms_t> atoms_counting_it;
  thrust::counting_iterator<tiles_t> tiles_counting_it;

  /**
   * @brief Construct a setup object for load balance schedule.
   *
   * @param tiles Tiles iterator.
   * @param num_tiles Number of tiles.
   * @param num_atoms Number of atoms.
   */
  __device__ __forceinline__ setup(storage_t& _buffer,
                                   tiles_iterator_t _tiles,
                                   tile_size_t _num_tiles,
                                   atom_size_t _num_atoms)
      : tile_traits_t(_num_tiles, _tiles),
        atom_traits_t(_num_atoms),
        buffer(_buffer),
        total_work(_num_tiles + _num_atoms),  // num_merge_items
        merge_tile_size(items_per_tile),      // merge_tile_size
        num_merge_tiles(
            math::ceil_div(total_work, merge_tile_size))  // num_merge_tiles
  {}

  __device__ __forceinline__ auto init() {
    /// Calculate the diagonals.
    atom_size_t tid = (blockIdx.x * gridDim.y) + blockIdx.y;

    if (tid >= num_merge_tiles)
      return coordinate_t{-1, -1};

    /// Two threads per-block perform the search to find the diagonal for a
    /// block. This limits the search each thread has to do to per-block
    /// diagonals.
    if (threadIdx.x < 2) {
      // Search our starting coordinates
      atom_size_t diagonal = (tid + threadIdx.x) * items_per_tile;
      thrust::counting_iterator<atoms_t> atoms_indices(0);

      /// Search across the diagonals to find coordinates to process.
      auto st = search(diagonal, (tile_traits_t::begin() + 1), atoms_indices,
                       tile_traits_t::size(), atom_traits_t::size());

      buffer.tile_coords[threadIdx.x] = st;
    }
    __syncthreads();

    auto tile_start_coord = buffer.tile_coords[0];
    auto tile_end_coord = buffer.tile_coords[1];

    tile_size_t tile_num_tiles = tile_end_coord.first - tile_start_coord.first;
    atom_size_t tile_num_nonzeros =
        tile_end_coord.second - tile_start_coord.second;

    tiles_iterator_t end_offsets = tile_traits_t::begin() + 1;

    /// Gather the row end-offsets for the merge tile into shared memory.
    for (int item = threadIdx.x;  // first thread of the block.
         item < tile_num_tiles + items_per_thread;  // thread's work.
         item += threads_per_block)                 // stride by block dim.
    {
      const int offset = min(static_cast<int>(tile_start_coord.first + item),
                             static_cast<int>(tile_traits_t::size() - 1));
      buffer.tile_end_offset[item] = end_offsets[offset];
    }

    // Set these iterators for use later.
    tiles_counting_it =
        thrust::counting_iterator<tiles_t>(tile_start_coord.first);
    atoms_counting_it =
        thrust::counting_iterator<atoms_t>(tile_start_coord.second);

    __syncthreads();

    // Search for the thread's starting coordinate within the merge tile
    thrust::counting_iterator<atoms_t> tile_atoms_indices(
        tile_start_coord.second);

    /// Search across the diagonals to find coordinates to process.
    auto thread_start_coord = search(
        atom_size_t(threadIdx.x * items_per_thread), buffer.tile_end_offset,
        tile_atoms_indices, tile_num_tiles, tile_num_nonzeros);

    __syncthreads();  // Perf-sync

    return thread_start_coord;
  }

  __device__ __forceinline__ bool is_valid_accessor(coordinate_t& coord) const {
    return (coord.first >= 0) && (coord.second >= 0);
  }

  __device__ __forceinline__ step_range_t<int> virtual_idx() const {
    return custom_stride_range(int(0), int(items_per_thread), tiles_t(1));
  }

  __device__ __forceinline__ atoms_t atom_idx(int vid, coordinate_t& coord) {
    return min(atoms_counting_it[coord.second], int(atom_traits_t::size()) - 1);
  }

  __device__ __forceinline__ tiles_t tile_idx(coordinate_t& coord) const {
    return tiles_counting_it[coord.first];
  }

 private:
  /**
   * @brief Thrust based 2D binary-search for merge-path algorithm.
   *
   * @param diagonal Diagonal of the search.
   * @param a First iterator.
   * @param b Second iterator.
   * @param a_len Length of the first iterator.
   * @param b_len Length of the second iterator.
   * @return A coordinate.
   */
  template <typename offset_t, typename xit_t, typename yit_t>
  __device__ __forceinline__ auto search(const offset_t& diagonal,
                                         const xit_t a,
                                         const yit_t b,
                                         const offset_t& a_len,
                                         const offset_t& b_len) {
    /// Diagonal search range (in x-coordinate space)
    /// Note that the subtraction can result into a negative number, in which
    /// case the max would result as 0. But if we use offset_t here, and it is
    /// an unsigned type, we would get strange behavior, possible an unwanted
    /// sign conversion that we do not want.
    int x_min = max(int(diagonal) - int(b_len), int(0));
    int x_max = min(int(diagonal), int(a_len));

    auto it = thrust::lower_bound(
        thrust::seq,                                 // Sequential impl
        thrust::counting_iterator<offset_t>(x_min),  // Start iterator @x_min
        thrust::counting_iterator<offset_t>(x_max),  // End iterator @x_max
        diagonal,                                    // ...
        [=] __device__(const offset_t& idx, const offset_t& diagonal) {
          return a[idx] <= b[diagonal - idx - 1];
        });

    return coordinate_t{int(min(*it, a_len)), int(diagonal - *it)};

    // while (x_min < x_max) {
    //   int split_pivot = (x_min + x_max) >> 1;
    //   if (a[split_pivot] <= b[diagonal - split_pivot - 1]) {
    //     // Move candidate split range up A, down B
    //     x_min = split_pivot + 1;
    //   } else {
    //     // Move candidate split range up B, down A
    //     x_max = split_pivot;
    //   }
    // }

    // return coordinate_t{min(x_min, int(a_len)), (int(diagonal) - x_min)};
  }

  std::size_t total_work;
  std::size_t merge_tile_size;
  std::size_t num_merge_tiles;
};

}  // namespace schedule
}  // namespace loops