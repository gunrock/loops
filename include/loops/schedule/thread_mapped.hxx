/**
 * @file thread_mapped.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2022-02-04
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <loops/stride_ranges.hxx>
#include <loops/schedule.hxx>

namespace loops {
namespace schedule {

/**
 * @brief Thread-mapped schedule's setup interface.
 *
 * @tparam tiles_type Type of the tiles.
 * @tparam atoms_type Type of the atoms.
 * @tparam tile_size_type Type of the tile size.
 * @tparam atom_size_type Type of the atom size.
 */
template <typename tiles_type,
          typename atoms_type,
          typename tile_size_type,
          typename atom_size_type>
class setup<algorithms_t::thread_mapped,
            1,
            1,
            tiles_type,
            atoms_type,
            tile_size_type,
            atom_size_type> : public tile_traits<algorithms_t::thread_mapped,
                                                 tiles_type,
                                                 tile_size_type>,
                              public atom_traits<algorithms_t::thread_mapped,
                                                 atoms_type,
                                                 atom_size_type> {
 public:
  using tiles_t = tiles_type;          /// Tile Type
  using atoms_t = atoms_type;          /// Atom Type
  using tiles_iterator_t = tiles_t*;   /// Tile Iterator Type
  using atoms_iterator_t = atoms_t*;   /// Atom Iterator Type
  using tile_size_t = tile_size_type;  /// Tile Size Type
  using atom_size_t = atom_size_type;  /// Atom Size Type

  using tile_traits_t =
      tile_traits<algorithms_t::thread_mapped, tiles_type, tile_size_type>;
  using atom_traits_t =
      atom_traits<algorithms_t::thread_mapped, atoms_type, atom_size_type>;

  /**
   * @brief Default constructor.
   *
   */
  __host__ __device__ setup() : tile_traits_t(), atom_traits_t() {}

  /**
   * @brief Construct a setup object for load balance schedule.
   *
   * @param tiles Tiles iterator.
   * @param num_tiles Number of tiles.
   * @param num_atoms Number of atoms.
   */
  __host__ __device__ setup(tiles_t* tiles,
                            tile_size_t num_tiles,
                            atom_size_t num_atoms)
      : tile_traits_t(num_tiles, tiles), atom_traits_t(num_atoms) {}

  /**
   * @brief Creates a range of tiles to process within a for loop.
   *
   * @example The following code snippet shows how to use this function.
   * \code{.cpp}
   * for (auto t : config.tiles()) {
   *  // Process tile t.
   * }
   * \endcode
   *
   * @return grid_stride_range<tile_size_t> Range of tiles to process.
   */
  __device__ step_range_t<tile_size_t> tiles() const {
    return grid_stride_range(tile_size_t(0), tile_traits_t::size());
  }

  /**
   * @brief Creates a range of atoms to process within a for loop.
   *
   * @param tid Tile ID for which to create the atom range for.
   * @return Range of atoms to process.
   */
  __device__ auto atoms(const tile_size_t& tile) {
    return loops::range(tile_traits_t::begin()[tile],
                        tile_traits_t::begin()[tile + 1]);
  }

  template <typename iterator_t>
  __device__ auto atoms(const tile_size_t& tile, iterator_t count_entries) {
    return loops::range(count_entries(tile), tile_traits_t::begin()[tile + 1]);
  }
};

}  // namespace schedule
}  // namespace loops
