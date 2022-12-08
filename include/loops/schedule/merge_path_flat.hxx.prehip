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
#include <loops/util/search.hxx>
#include <loops/util/launch.hxx>
#include <loops/util/device.hxx>

#include <loops/container/coordinate.hxx>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/pair.h>

namespace loops {
namespace schedule {

using coord_t = coordinate_t<unsigned int>;

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

namespace merge_path {
/**
 * @brief CUB's implementation, generalized. Identifies merge path starting
 * coordinates for each tile.
 *
 * @tparam THREADS_PER_BLOCK
 * @tparam ITEMS_PER_THREAD
 * @tparam tiles_t
 * @tparam tile_size_t
 * @tparam atom_size_t
 * @param tiles
 * @param num_tiles
 * @param num_atoms
 * @param d_tile_coordinates
 * @return __global__
 */
template <std::size_t THREADS_PER_BLOCK,
          std::size_t ITEMS_PER_THREAD,
          typename tiles_t,
          typename atoms_t,
          typename tile_size_t,
          typename atom_size_t>
__global__ void generate_search_coordinates(tiles_t* tiles,
                                            tile_size_t num_tiles,
                                            atom_size_t num_atoms,
                                            std::size_t num_merge_tiles,
                                            coord_t* d_tile_coordinates) {
  enum : unsigned int {
    items_per_tile = THREADS_PER_BLOCK * ITEMS_PER_THREAD,
  };

  // Find the starting coordinate for all tiles (plus the end coordinate of
  // the last one) as a separate step.
  int tile_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tile_idx < num_merge_tiles + 1) {
    thrust::counting_iterator<atoms_t> atoms_indices(0);
    tile_size_t diagonal = (tile_idx * items_per_tile);
    coord_t tile_coordinate;

    // Search the merge path (block-wide.)
    tile_coordinate = search::_binary_search(diagonal, tiles + 1, atoms_indices,
                                             num_tiles, num_atoms);

    // Output starting offset
    d_tile_coordinates[tile_idx] = tile_coordinate;
  }
}

/**
 * @brief Work-oriented schedule's preprocess interface.
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
class preprocess_t {
 public:
  using tiles_t = tiles_type;          /// Tile Type
  using atoms_t = atoms_type;          /// Atom Type
  using tiles_iterator_t = tiles_t*;   /// Tile Iterator Type
  using atoms_iterator_t = atoms_t*;   /// Atom Iterator Type
  using tile_size_t = tile_size_type;  /// Tile Size Type
  using atom_size_t = atom_size_type;  /// Atom Size Type

  /**
   * @brief Construct a preprocess object for load balance schedule.
   *
   * @param tiles Tiles iterator.
   * @param num_tiles Number of tiles.
   * @param num_atoms Number of atoms.
   */
  preprocess_t(tiles_iterator_t _tiles,
               tile_size_t _num_tiles,
               atom_size_t _num_atoms,
               cudaStream_t stream = 0)
      : total_work(_num_tiles + _num_atoms),
        num_merge_tiles(
            math::ceil_div(total_work, THREADS_PER_BLOCK * ITEMS_PER_THREAD)),
        d_tile_coordinates(nullptr) {
    int sm_count;
    int device_ordinal = device::get();
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                           device_ordinal);

    constexpr std::size_t block_size = THREADS_PER_BLOCK;
    dim3 grid_size = math::ceil_div(num_merge_tiles + 1, block_size);

    // Use separate search kernel if we have enough tiles to saturate the
    // device. This will pre-calculated the bounds per block, which can
    // then be used inside our actual load-balanced kernel to compute faster.
    if (grid_size.x >= sm_count) {
      tile_coordinates.resize(num_merge_tiles + 1);
      d_tile_coordinates = tile_coordinates.data().get();

      if (d_tile_coordinates == nullptr)
        error::throw_if_exception(true, "Tile Coordinates failed allocation.");

      auto kernel =
          generate_search_coordinates<THREADS_PER_BLOCK, ITEMS_PER_THREAD,
                                      tiles_t, atoms_t, tile_size_t,
                                      atom_size_t>;
      launch::non_cooperative(stream, kernel, grid_size, block_size, _tiles,
                              _num_tiles, _num_atoms, num_merge_tiles,
                              d_tile_coordinates);
    }
  }

  /**
   * @brief Special copy constructor; we never copy the device_vector as it is
   * not supported within device code.
   *
   * @param rhs
   */
  __device__ __host__ preprocess_t(preprocess_t const& rhs) {
    total_work = rhs.total_work;
    num_merge_tiles = rhs.num_merge_tiles;
    d_tile_coordinates = tile_coordinates.data().get();
#ifndef __CUDA_ARH__
    tile_coordinates = rhs.tile_coordinates;
#endif
  }

  __device__ __host__ auto data() const { return d_tile_coordinates; }

 private:
  std::size_t total_work;
  std::size_t num_merge_tiles;
  coord_t* d_tile_coordinates;
  thrust::device_vector<coord_t> tile_coordinates;
};

}  // namespace merge_path

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
  using atoms_iterator_t = atoms_t*;   /// Atom Iterator Type
  using tile_size_t = tile_size_type;  /// Tile Size Type
  using atom_size_t = atom_size_type;  /// Atom Size Type
  using meta_t = merge_path::preprocess_t<THREADS_PER_BLOCK,
                                          ITEMS_PER_THREAD,
                                          tiles_type,
                                          atoms_type,
                                          tile_size_type,
                                          atom_size_type>;

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
    coord_t tile_coords[2];
    tiles_t tile_end_offset[items_per_thread + items_per_tile + 1];
  };

  storage_t& buffer;
  meta_t& meta;

  thrust::counting_iterator<atoms_t> atoms_counting_it;
  thrust::counting_iterator<tiles_t> tiles_counting_it;

  /**
   * @brief Construct a setup object for load balance schedule.
   *
   * @param tiles Tiles iterator.
   * @param num_tiles Number of tiles.
   * @param num_atoms Number of atoms.
   */
  __device__ __forceinline__ setup(meta_t& _meta,
                                   storage_t& _buffer,
                                   tiles_iterator_t _tiles,
                                   tile_size_t _num_tiles,
                                   atom_size_t _num_atoms)
      : tile_traits_t(_num_tiles, _tiles),
        atom_traits_t(_num_atoms),
        meta(_meta),
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
      return coord_t{std::numeric_limits<unsigned int>::max(),
                     std::numeric_limits<unsigned int>::max()};

    /// Two threads per-block perform the search to find the diagonal for a
    /// block. This limits the search each thread has to do to per-block
    /// diagonals.
    if (threadIdx.x < 2) {
      if (meta.data() == nullptr) {
        // Search our starting coordinates
        atom_size_t diagonal = (tid + threadIdx.x) * items_per_tile;
        thrust::counting_iterator<atoms_t> atoms_indices(0);

        /// Search across the diagonals to find coordinates to process.
        coord_t st = search::_binary_search(
            diagonal, (tile_traits_t::begin() + 1), atoms_indices,
            tile_traits_t::size(), atom_traits_t::size());

        buffer.tile_coords[threadIdx.x] = st;
      } else {
        buffer.tile_coords[threadIdx.x] = meta.data()[tid + threadIdx.x];
      }
    }
    __syncthreads();

    auto tile_start_coord = buffer.tile_coords[0];
    auto tile_end_coord = buffer.tile_coords[1];

    tile_num_tiles = tile_end_coord.x - tile_start_coord.x;
    tile_num_atoms = tile_end_coord.y - tile_start_coord.y;

    tiles_iterator_t end_offsets = tile_traits_t::begin() + 1;

    /// Gather the row end-offsets for the merge tile into shared memory.
    for (int item = threadIdx.x;  // first thread of the block.
         item < tile_num_tiles + items_per_thread;  // thread's work.
         item += threads_per_block)                 // stride by block dim.
    {
      const int offset = min(static_cast<int>(tile_start_coord.x + item),
                             static_cast<int>(tile_traits_t::size() - 1));
      buffer.tile_end_offset[item] = end_offsets[offset];
    }

    // Set these iterators for use later.
    tiles_counting_it = thrust::counting_iterator<tiles_t>(tile_start_coord.x);
    atoms_counting_it = thrust::counting_iterator<atoms_t>(tile_start_coord.y);

    __syncthreads();

    // Search for the thread's starting coordinate within the merge tile
    thrust::counting_iterator<atoms_t> tile_atoms_indices(tile_start_coord.y);

    /// Search across the diagonals to find coordinates to process.
    coord_t thread_start_coord = search::_binary_search(
        atom_size_t(threadIdx.x * items_per_thread), buffer.tile_end_offset,
        tile_atoms_indices, tile_num_tiles, tile_num_atoms);

    __syncthreads();  // Perf-sync

    return thread_start_coord;
  }

  __device__ __forceinline__ bool is_valid_accessor(coord_t& coord) const {
    return (coord.x != std::numeric_limits<unsigned int>::max()) &&
           (coord.y != std::numeric_limits<unsigned int>::max());
  }

  /**
   * @brief Range from 0 to ITEMS_PER_THREAD.
   *
   * @return step_range_t<int> returns the range.
   */
  __device__ __forceinline__ step_range_t<int> virtual_idx() const {
    return custom_stride_range(int(0), int(items_per_thread), tiles_t(1));
  }

  /**
   * @brief Returns the atoms index, notably it does not increment it.
   *
   * @param vid ITEM to be processed.
   * @param coord load-balanced map.
   * @return atoms_t Return the atom index.
   */
  __device__ __forceinline__ atoms_t atom_idx(int vid, coord_t& coord) {
    return min(atoms_counting_it[coord.y], int(atom_traits_t::size()) - 1);
  }

  /**
   * @brief Returns the tile index, notably it does not increment it.
   *
   * @param coord load-balanced map.
   * @return tiles_t Return the tile index.
   */
  __device__ __forceinline__ tiles_t tile_idx(coord_t& coord) const {
    return tiles_counting_it[coord.x];
  }

  __device__ __forceinline__ tile_size_t num_tiles() const {
    return tile_num_tiles;
  }

  __device__ __forceinline__ atom_size_t num_atoms() const {
    return tile_num_atoms;
  }

 private:
  std::size_t total_work;
  std::size_t merge_tile_size;
  std::size_t num_merge_tiles;
  tile_size_t tile_num_tiles;
  atom_size_t tile_num_atoms;
};

}  // namespace schedule
}  // namespace loops