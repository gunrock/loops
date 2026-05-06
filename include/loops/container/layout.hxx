/**
 * @file layout.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Layout views consumed by the schedule machinery.
 *
 * A "tile-atom layout" describes how an irregular workload is partitioned
 * into tiles, where each tile owns some number of atoms. The schedules
 * (thread_mapped, group_mapped, work_oriented, merge_path_flat) talk to
 * layouts only through this contract; they never touch format-specific
 * storage. Adding support for a new format (COO, ELL, BCSR, DIA, ...) is
 * therefore a matter of writing a new layout view that satisfies the
 * contract below.
 *
 * --------------------------- The Layout Contract ---------------------------
 *
 *   struct my_layout {
 *     using tile_id_t = ...;                    // tile index type
 *     using atom_id_t = ...;                    // atom index type
 *     using tile_end_iterator_t = ...;          // random-access iterator
 *                                               // returning atom_id_t
 *
 *     __host__ __device__ tile_id_t num_tiles() const;
 *     __host__ __device__ atom_id_t num_atoms() const;
 *
 *     __host__ __device__ atom_id_t tile_begin(tile_id_t t) const;
 *     __host__ __device__ atom_id_t tile_end  (tile_id_t t) const;
 *     __host__ __device__ atom_id_t tile_size (tile_id_t t) const;
 *
 *     // For merge-path schedules. Must satisfy: it[k] == tile_end(k) for
 *     // any k in [0, num_tiles). May be a raw pointer, a thrust
 *     // transform_iterator, or any other random-access iterator.
 *     __host__ __device__ tile_end_iterator_t tile_end_iter() const;
 *   };
 *
 * Properties the schedules assume:
 *
 *   * tile_begin(0) == 0
 *   * tile_end(num_tiles() - 1) == num_atoms()
 *   * tile_end is monotonically non-decreasing in t (merge-path search relies
 *     on this; it binary-searches over tile_end_iter()).
 *
 * Layout views are passed *by value* into __global__ kernels, so they
 * should be POD-like and small (a few words is typical: pointers, sizes,
 * maybe a stride). Avoid owning resources; treat the layout as a
 * non-owning view over user-managed storage.
 *
 * For an end-to-end example of writing a custom layout, see
 * `examples/spmv/custom_layout.cu`.
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <cstddef>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

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

/**
 * @brief ELL-shaped tile-atom layout view (uniform tile size).
 *
 * Tiles correspond to rows; every tile holds exactly `pitch` atoms (some of
 * which may be padding, signaled by a sentinel column index in the storage
 * array). No offsets array is needed:
 *
 *   tile_begin(t)   = t * pitch
 *   tile_end(t)     = (t + 1) * pitch
 *   tile_size(t)    = pitch
 *   tile_end_iter() = transform_iterator: i -> (i + 1) * pitch
 *
 * The atom payload (column ids and values) lives in user-managed arrays of
 * length `num_tiles * pitch`, indexed directly by `atom_id`. Whether the
 * arrays are stored row-major or column-major is the user's choice; the
 * schedule treats `atom_id` as opaque.
 *
 * @tparam TileId Tile-id type (e.g., row id).
 * @tparam AtomId Atom-id type (flat index into the per-row buckets).
 */
template <typename TileId, typename AtomId>
struct ell {
 private:
  /// Functor used to materialize tile_end values lazily.
  struct tile_end_fn {
    AtomId pitch;
    __host__ __device__ AtomId operator()(TileId i) const {
      return static_cast<AtomId>(i + 1) * pitch;
    }
  };

 public:
  using tile_id_t = TileId;
  using atom_id_t = AtomId;
  using tile_end_iterator_t = thrust::transform_iterator<
      tile_end_fn,
      thrust::counting_iterator<TileId>,
      AtomId>;

  TileId n_tiles_;
  AtomId pitch_;  /// atoms per tile (uniform); = max-non-zeros-per-row in SpMV

  __host__ __device__ ell() : n_tiles_(0), pitch_(0) {}

  __host__ __device__ ell(TileId num_tiles, AtomId pitch)
      : n_tiles_(num_tiles), pitch_(pitch) {}

  __host__ __device__ TileId num_tiles() const { return n_tiles_; }
  __host__ __device__ AtomId num_atoms() const {
    return static_cast<AtomId>(n_tiles_) * pitch_;
  }

  __host__ __device__ AtomId tile_begin(TileId t) const {
    return static_cast<AtomId>(t) * pitch_;
  }
  __host__ __device__ AtomId tile_end(TileId t) const {
    return static_cast<AtomId>(t + 1) * pitch_;
  }
  __host__ __device__ AtomId tile_size(TileId /*t*/) const { return pitch_; }

  /// Random-access iterator i where i[k] == tile_end(k).
  __host__ __device__ tile_end_iterator_t tile_end_iter() const {
    return thrust::make_transform_iterator(
        thrust::counting_iterator<TileId>(0), tile_end_fn{pitch_});
  }
};

}  // namespace layout
}  // namespace loops
