/**
 * @file thread_mapped.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Thread-mapped schedule (one thread per tile).
 * @version 0.2
 * @date 2026-05-05
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <loops/stride_ranges.hxx>
#include <loops/schedule.hxx>
#include <loops/container/layout.hxx>

namespace loops {
namespace schedule {

/**
 * @brief Thread-mapped schedule's setup interface.
 *
 * Each thread is responsible for one tile and walks the atoms of that tile
 * sequentially. The schedule consumes the workload through a layout view
 * (default: `layout::csr`) so it can be reused across CSR / COO / ELL /
 * user-defined sparse formats without modification.
 *
 * @tparam tiles_type      Tile-id storage type (e.g., row-id).
 * @tparam atoms_type      Atom-id storage type (e.g., flat nnz position).
 * @tparam tile_size_type  Counter type for tiles.
 * @tparam atom_size_type  Counter type for atoms.
 * @tparam layout_type     Layout view (default: layout::csr).
 */
template <typename tiles_type,
          typename atoms_type,
          typename tile_size_type,
          typename atom_size_type,
          typename layout_type>
class setup<algorithms_t::thread_mapped,
            1,
            1,
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

  /// Layout view over the workload. Defaults to CSR but is supplied as a
  /// template parameter; users may pass any layout that satisfies the
  /// contract in `loops/container/layout.hxx` (CSR, ELL, custom, ...).
  using layout_t = layout_type;

  /// Default constructor produces an empty schedule.
  __host__ __device__ setup() : layout_() {}

  /**
   * @brief Construct a setup from a CSR-shaped offsets pointer.
   *
   * Equivalent to constructing a `layout::csr` view internally.
   *
   * @param tiles      Pointer to the tile-end-offset array (size num_tiles+1).
   * @param num_tiles  Number of tiles.
   * @param num_atoms  Total number of atoms.
   */
  __host__ __device__ setup(tiles_t* tiles,
                            tile_size_t num_tiles,
                            atom_size_t num_atoms)
      : layout_(tiles, num_tiles, num_atoms) {}

  /**
   * @brief Construct a setup directly from a layout view.
   *
   * Lets callers supply a non-CSR layout (e.g., a user-defined one) without
   * routing through the CSR-flavored constructor above.
   */
  __host__ __device__ explicit setup(layout_t layout) : layout_(layout) {}

  /**
   * @brief Range of tiles assigned to this thread (grid-stride).
   *
   * Usage:
   * \code{.cpp}
   * for (auto t : config.tiles()) {
   *   // process tile t
   * }
   * \endcode
   */
  __device__ step_range_t<tile_size_t> tiles() const {
    return grid_stride_range(tile_size_t(0),
                             static_cast<tile_size_t>(layout_.num_tiles()));
  }

  /**
   * @brief Range of atoms inside the given tile.
   *
   * @param tile Tile id whose atoms to iterate.
   * @return Range over [tile_begin(tile), tile_end(tile)).
   */
  __device__ auto atoms(const tile_size_t& tile) {
    return loops::range(layout_.tile_begin(tile), layout_.tile_end(tile));
  }

  /**
   * @brief Range of atoms inside the given tile, with a custom start.
   *
   * Used by SpMM-style kernels that resume iteration mid-tile.
   *
   * @param tile           Tile id whose atoms to iterate.
   * @param count_entries  Functor returning the resume offset for `tile`.
   */
  template <typename iterator_t>
  __device__ auto atoms(const tile_size_t& tile, iterator_t count_entries) {
    return loops::range(count_entries(tile), layout_.tile_end(tile));
  }

  /// Direct read access to the underlying layout (advanced use).
  __host__ __device__ const layout_t& layout() const { return layout_; }

 private:
  layout_t layout_;
};

}  // namespace schedule
}  // namespace loops
