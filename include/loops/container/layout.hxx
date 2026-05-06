/**
 * @file layout.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Layout views consumed by the schedule machinery.
 *
 * A "tile-atom layout" describes how the irregular workload is partitioned
 * into tiles, where each tile owns some number of atoms. Schedules
 * (thread_mapped, group_mapped, work_oriented, merge_path_flat) talk to
 * layouts only through this contract; they never touch format-specific
 * storage. Adding support for a new format (COO, ELL, BCSR, ...) means
 * writing a new layout view that exposes the same methods.
 *
 * The minimal contract a layout view must provide is:
 *
 *   using tile_id_t = ...;             // index type for tiles
 *   using atom_id_t = ...;             // index type for atoms
 *   tile_id_t num_tiles() const;       // total number of tiles
 *   atom_id_t num_atoms() const;       // total number of atoms
 *   atom_id_t tile_begin(tile_id_t t); // first atom id in tile t
 *   atom_id_t tile_end  (tile_id_t t); // one-past-last atom id in tile t
 *   atom_id_t tile_size (tile_id_t t); // tile_end(t) - tile_begin(t)
 *   <random-access iterator>
 *     tile_end_iter()      const;      // i[k] == tile_end(k), for merge-path
 *
 * Methods may be __host__ __device__ as appropriate. Layout views are passed
 * by value into kernels, so they should be POD-like and small.
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <cstddef>

namespace loops {
namespace layout {

/**
 * @brief CSR-shaped tile-atom layout view.
 *
 * Tiles correspond to rows; the atoms of tile t are the contiguous range
 * [offsets[t], offsets[t + 1]). Equivalently, offsets is a non-decreasing
 * prefix-sum array of length num_tiles + 1 over per-tile atom counts.
 *
 * This is the layout used by the existing CSR-based examples; it can be
 * constructed from a `csr_t` container's offsets pointer + sizes.
 *
 * @tparam TileId Index type for tiles (e.g., row id).
 * @tparam AtomId Index type for atoms (e.g., flat nnz position).
 */
template <typename TileId, typename AtomId>
struct csr {
  using tile_id_t = TileId;
  using atom_id_t = AtomId;
  using tile_end_iterator_t = AtomId const*;

  AtomId const* offsets_;  /// length num_tiles + 1, monotonically non-decreasing
  TileId n_tiles_;
  AtomId n_atoms_;

  __host__ __device__ csr() : offsets_(nullptr), n_tiles_(0), n_atoms_(0) {}

  __host__ __device__ csr(AtomId const* offsets,
                          TileId num_tiles,
                          AtomId num_atoms)
      : offsets_(offsets), n_tiles_(num_tiles), n_atoms_(num_atoms) {}

  __host__ __device__ TileId num_tiles() const { return n_tiles_; }
  __host__ __device__ AtomId num_atoms() const { return n_atoms_; }

  __host__ __device__ AtomId tile_begin(TileId t) const { return offsets_[t]; }
  __host__ __device__ AtomId tile_end(TileId t) const { return offsets_[t + 1]; }
  __host__ __device__ AtomId tile_size(TileId t) const {
    return offsets_[t + 1] - offsets_[t];
  }

  /// Random-access iterator i where i[k] == tile_end(k). For CSR this is
  /// just `offsets + 1`; merge-path-style schedules binary-search over it.
  __host__ __device__ tile_end_iterator_t tile_end_iter() const {
    return offsets_ + 1;
  }
};

}  // namespace layout
}  // namespace loops
