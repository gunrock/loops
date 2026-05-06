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
#include <loops/container/layout.hxx>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/pair.h>

namespace loops {
namespace schedule {

using coord_t = coordinate_t<unsigned int>;

namespace merge_path {
/**
 * @brief Per-block merge-path search kernel.
 *
 * Layout-generic version of CUB's diagonal search: for each merge-path tile,
 * find the starting (tile_id, atom_id) coordinate by binary-searching the
 * layout's `tile_end_iter` against a counting iterator over atoms. The result
 * is materialized into `d_tile_coordinates` so the main kernel can skip the
 * search at runtime.
 */
template <std::size_t THREADS_PER_BLOCK,
          std::size_t ITEMS_PER_THREAD,
          typename layout_t,
          typename tile_size_t,
          typename atom_size_t>
__global__ void generate_search_coordinates(layout_t layout,
                                            tile_size_t num_tiles,
                                            atom_size_t num_atoms,
                                            std::size_t num_merge_tiles,
                                            coord_t* d_tile_coordinates) {
  using atoms_t = typename layout_t::atom_id_t;

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
    tile_coordinate = search::_binary_search(diagonal, layout.tile_end_iter(),
                                             atoms_indices, num_tiles,
                                             num_atoms);

    // Output starting offset
    d_tile_coordinates[tile_idx] = tile_coordinate;
  }
}

/**
 * @brief Merge-path preprocess (host-side helper).
 *
 * Computes per-block starting coordinates ahead of time so the main kernel
 * can skip its diagonal search. Layout-generic: any layout view satisfying
 * the contract in `loops/container/layout.hxx` works.
 *
 * @tparam THREADS_PER_BLOCK Threads per block.
 * @tparam ITEMS_PER_THREAD  Number of items per thread to process.
 * @tparam tiles_type        Tile-id type (used for back-compat ctor).
 * @tparam atoms_type        Atom-id type (used for back-compat ctor).
 * @tparam tile_size_type    Counter type for tiles.
 * @tparam atom_size_type    Counter type for atoms.
 */
template <std::size_t THREADS_PER_BLOCK,
          std::size_t ITEMS_PER_THREAD,
          typename tiles_type,
          typename atoms_type,
          typename tile_size_type,
          typename atom_size_type,
          typename layout_type = layout::csr<tiles_type, atoms_type>>
class preprocess_t {
 public:
  using tiles_t = tiles_type;
  using atoms_t = atoms_type;
  using tiles_iterator_t = tiles_t*;
  using atoms_iterator_t = atoms_t*;
  using tile_size_t = tile_size_type;
  using atom_size_t = atom_size_type;
  using layout_t = layout_type;

  /// Construct from a CSR-shaped offsets pointer (back-compat shortcut;
  /// only valid when @c layout_type is @c layout::csr ).
  preprocess_t(tiles_iterator_t _tiles,
               tile_size_t _num_tiles,
               atom_size_t _num_atoms,
               cudaStream_t stream = 0)
      : preprocess_t(layout_t(_tiles, _num_tiles, _num_atoms), stream) {}

  /// Construct directly from a layout view (any layout type).
  preprocess_t(layout_t _layout, cudaStream_t stream = 0)
      : total_work(_layout.num_tiles() + _layout.num_atoms()),
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
    // device. This will pre-calculate the bounds per block, which can
    // then be used inside our actual load-balanced kernel to compute faster.
    if (grid_size.x >= sm_count) {
      tile_coordinates.resize(num_merge_tiles + 1);
      d_tile_coordinates = tile_coordinates.data().get();

      if (d_tile_coordinates == nullptr)
        error::throw_if_exception(true, "Tile Coordinates failed allocation.");

      auto kernel =
          generate_search_coordinates<THREADS_PER_BLOCK, ITEMS_PER_THREAD,
                                      layout_t, tile_size_t, atom_size_t>;
      launch::non_cooperative(stream, kernel, grid_size, block_size, _layout,
                              _layout.num_tiles(), _layout.num_atoms(),
                              num_merge_tiles, d_tile_coordinates);
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
          typename atom_size_type,
          typename layout_type>
class setup<algorithms_t::merge_path_flat,
            THREADS_PER_BLOCK,
            ITEMS_PER_THREAD,
            tiles_type,
            atoms_type,
            tile_size_type,
            atom_size_type,
            layout_type> {
 public:
  using tiles_t = tiles_type;
  using atoms_t = atoms_type;
  using tiles_iterator_t = tiles_t*;
  using atoms_iterator_t = atoms_t*;
  using tile_size_t = tile_size_type;
  using atom_size_t = atom_size_type;
  using layout_t = layout_type;
  using meta_t = merge_path::preprocess_t<THREADS_PER_BLOCK,
                                          ITEMS_PER_THREAD,
                                          tiles_type,
                                          atoms_type,
                                          tile_size_type,
                                          atom_size_type,
                                          layout_type>;

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
   * @brief Construct a setup from a CSR-shaped offsets pointer.
   *
   * @param _meta      Pre-computed merge-path metadata.
   * @param _buffer    Scratch storage in shared memory.
   * @param _tiles     Pointer to the tile-end-offset array (size num_tiles+1).
   * @param _num_tiles Number of tiles.
   * @param _num_atoms Total number of atoms.
   */
  __device__ __forceinline__ setup(meta_t& _meta,
                                   storage_t& _buffer,
                                   tiles_iterator_t _tiles,
                                   tile_size_t _num_tiles,
                                   atom_size_t _num_atoms)
      : layout_(_tiles, _num_tiles, _num_atoms),
        meta(_meta),
        buffer(_buffer),
        total_work(_num_tiles + _num_atoms),
        merge_tile_size(items_per_tile),
        num_merge_tiles(math::ceil_div(total_work, merge_tile_size)) {}

  /// Construct directly from a layout view.
  __device__ __forceinline__ setup(meta_t& _meta,
                                   storage_t& _buffer,
                                   layout_t _layout)
      : layout_(_layout),
        meta(_meta),
        buffer(_buffer),
        total_work(_layout.num_tiles() + _layout.num_atoms()),
        merge_tile_size(items_per_tile),
        num_merge_tiles(math::ceil_div(total_work, merge_tile_size)) {}

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
        /// Explicit casts unify offset_t for template deduction (layout
        /// reports counts in its native int-ish type; the search wants
        /// atom_size_t).
        coord_t st = search::_binary_search(
            diagonal, layout_.tile_end_iter(), atoms_indices,
            static_cast<atom_size_t>(layout_.num_tiles()),
            static_cast<atom_size_t>(layout_.num_atoms()));

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

    auto end_offsets = layout_.tile_end_iter();

    /// Gather the row end-offsets for the merge tile into shared memory.
    for (int item = threadIdx.x;  // first thread of the block.
         item < tile_num_tiles + items_per_thread;  // thread's work.
         item += threads_per_block)                 // stride by block dim.
    {
      const int offset = min(static_cast<int>(tile_start_coord.x + item),
                             static_cast<int>(layout_.num_tiles() - 1));
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
    return min(atoms_counting_it[coord.y],
               static_cast<int>(layout_.num_atoms()) - 1);
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

  /// Direct read access to the underlying layout (advanced use).
  __host__ __device__ const layout_t& layout() const { return layout_; }

 private:
  layout_t layout_;
  std::size_t total_work;
  std::size_t merge_tile_size;
  std::size_t num_merge_tiles;
  tile_size_t tile_num_tiles;
  atom_size_t tile_num_atoms;
};

}  // namespace schedule
}  // namespace loops