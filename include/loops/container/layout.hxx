/**
 * @file layout.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Layout views consumed by the schedule machinery.
 * @version 0.2
 * @date 2026-05-05
 *
 * A "tile-atom layout" describes how an irregular workload is partitioned
 * into tiles, where each tile owns some number of atoms. The schedules
 * (@c thread_mapped , @c group_mapped , @c work_oriented ,
 * @c merge_path_flat ) talk to layouts only through the contract below;
 * they never touch format-specific storage. Adding support for a new
 * format (COO, ELL, BCSR, DIA, ...) is therefore a matter of writing a new
 * layout view that satisfies the contract.
 *
 * @section layout_contract The Layout Contract
 *
 * @code{.cpp}
 * struct my_layout {
 *   using tile_id_t = ...;             // tile index type
 *   using atom_id_t = ...;             // atom index type
 *   using tile_end_iterator_t = ...;   // random-access iterator
 *                                      // returning atom_id_t
 *
 *   __host__ __device__ tile_id_t num_tiles() const;
 *   __host__ __device__ atom_id_t num_atoms() const;
 *
 *   __host__ __device__ atom_id_t tile_begin(tile_id_t t) const;
 *   __host__ __device__ atom_id_t tile_end  (tile_id_t t) const;
 *   __host__ __device__ atom_id_t tile_size (tile_id_t t) const;
 *
 *   // For merge-path schedules. Must satisfy: it[k] == tile_end(k) for
 *   // any k in [0, num_tiles). May be a raw pointer, a thrust
 *   // transform_iterator, or any other random-access iterator.
 *   __host__ __device__ tile_end_iterator_t tile_end_iter() const;
 *
 *   // Optional: inverse of tile_begin/end. Returns the tile-id that owns
 *   // the given atom (i.e., t s.t. tile_begin(t) <= a < tile_end(t)).
 *   // Required only for kernels that need per-atom output addressing
 *   // (e.g., SpMV kernels driving partitioned layouts that cross row
 *   // boundaries). Implementations should be O(1) or O(log num_tiles).
 *   __host__ __device__ tile_id_t tile_of(atom_id_t a) const;
 * };
 * @endcode
 *
 * @par Invariants assumed by the schedules
 * - @c tile_begin(0) @c == @c 0
 * - @c tile_end(num_tiles()-1) @c == @c num_atoms()
 * - @c tile_end is monotonically non-decreasing in @c t (merge-path search
 *   relies on this; it binary-searches over @c tile_end_iter() ).
 *
 * @note Layout views are passed *by value* into @c __global__ kernels, so
 * they should be POD-like and small (a few words is typical: pointers,
 * sizes, maybe a stride). Avoid owning resources; treat the layout as a
 * non-owning view over user-managed storage.
 *
 * @see examples/spmv/custom_layout.cu for an end-to-end example of writing
 *      a custom layout.
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
 * @tparam tile_id_type Index type for tiles (e.g., row id).
 * @tparam atom_id_type Index type for atoms (e.g., flat nnz position).
 */
template <typename tile_id_type, typename atom_id_type>
struct csr {
  using tile_id_t = tile_id_type;
  using atom_id_t = atom_id_type;
  using tile_end_iterator_t = atom_id_t const*;

  atom_id_t const*
      offsets_;  /// length num_tiles + 1, monotonically non-decreasing.
  tile_id_t n_tiles_;
  atom_id_t n_atoms_;

  __host__ __device__ csr() : offsets_(nullptr), n_tiles_(0), n_atoms_(0) {}

  __host__ __device__ csr(atom_id_t const* offsets,
                          tile_id_t num_tiles,
                          atom_id_t num_atoms)
      : offsets_(offsets), n_tiles_(num_tiles), n_atoms_(num_atoms) {}

  __host__ __device__ tile_id_t num_tiles() const { return n_tiles_; }
  __host__ __device__ atom_id_t num_atoms() const { return n_atoms_; }

  __host__ __device__ atom_id_t tile_begin(tile_id_t t) const {
    return offsets_[t];
  }
  __host__ __device__ atom_id_t tile_end(tile_id_t t) const {
    return offsets_[t + 1];
  }
  __host__ __device__ atom_id_t tile_size(tile_id_t t) const {
    return offsets_[t + 1] - offsets_[t];
  }

  /**
   * @brief Random-access iterator @c i where @c i[k] @c == @c tile_end(k).
   *
   * For CSR this is just @c offsets+1 ; merge-path-style schedules
   * binary-search over it to find tile boundaries.
   */
  __host__ __device__ tile_end_iterator_t tile_end_iter() const {
    return offsets_ + 1;
  }

  /**
   * @brief Tile-id (row) that owns atom @c a.
   *
   * Hand-rolled @c upper_bound on the offsets array, returning @c t such
   * that @c offsets_[t] @c <= @c a @c < @c offsets_[t+1] . Runs in
   * O(log num_tiles); empty rows are skipped over correctly because
   * @c upper_bound returns the *first* index with a strictly greater
   * value, and we subtract one to land on the owning tile.
   */
  __host__ __device__ tile_id_t tile_of(atom_id_t a) const {
    tile_id_t lo = 0;
    tile_id_t hi = n_tiles_;
    while (lo < hi) {
      tile_id_t mid = lo + ((hi - lo) >> 1);
      if (offsets_[mid + 1] <= a)
        lo = mid + 1;
      else
        hi = mid;
    }
    return lo;
  }
};

/**
 * @brief DIA-shaped tile-atom layout view (tile is a row, atom is a diagonal
 * cell).
 *
 * Diagonal (DIA) format stores @c num_diagonals dense diagonals; from a
 * scheduling perspective this is identical to an ELL with @c pitch ==
 * @c num_diagonals : every row has exactly @c num_diagonals atoms
 * (some of which may be padding zeros if @c (r, r + diag_offsets[d])
 * is out of the matrix). We could literally @c using @c dia @c =
 * @c ell , but a distinct type makes the kernel-side semantics
 * (atom-id maps to diagonal index, not ELL bucket) explicit.
 *
 * @tparam tile_id_type Tile-id type (row id).
 * @tparam atom_id_type Atom-id type (flat (row, diagonal-index) pair).
 */
template <typename tile_id_type, typename atom_id_type>
struct dia {
  using tile_id_t = tile_id_type;
  using atom_id_t = atom_id_type;

 private:
  struct tile_end_fn {
    atom_id_t pitch;  // == num_diagonals
    __host__ __device__ atom_id_t operator()(tile_id_t i) const {
      return static_cast<atom_id_t>(i + 1) * pitch;
    }
  };

 public:
  using tile_end_iterator_t =
      thrust::transform_iterator<tile_end_fn,
                                 thrust::counting_iterator<tile_id_t>,
                                 atom_id_t>;

  tile_id_t n_tiles_;  /// == num_rows
  atom_id_t pitch_;    /// == num_diagonals

  __host__ __device__ dia() : n_tiles_(0), pitch_(0) {}

  __host__ __device__ dia(tile_id_t num_rows, atom_id_t num_diags)
      : n_tiles_(num_rows), pitch_(num_diags) {}

  __host__ __device__ tile_id_t num_tiles() const { return n_tiles_; }
  __host__ __device__ atom_id_t num_atoms() const {
    return static_cast<atom_id_t>(n_tiles_) * pitch_;
  }

  __host__ __device__ atom_id_t tile_begin(tile_id_t t) const {
    return static_cast<atom_id_t>(t) * pitch_;
  }
  __host__ __device__ atom_id_t tile_end(tile_id_t t) const {
    return static_cast<atom_id_t>(t + 1) * pitch_;
  }
  __host__ __device__ atom_id_t tile_size(tile_id_t /*t*/) const {
    return pitch_;
  }

  __host__ __device__ tile_end_iterator_t tile_end_iter() const {
    return thrust::make_transform_iterator(
        thrust::counting_iterator<tile_id_t>(0), tile_end_fn{pitch_});
  }

  /// O(1): row that owns flat (row, diag-index) atom @c a.
  __host__ __device__ tile_id_t tile_of(atom_id_t a) const {
    return static_cast<tile_id_t>(a / pitch_);
  }
};

/**
 * @brief BCSR-shaped tile-atom layout view (tile is a block-row).
 *
 * Block Compressed Sparse Row (BCSR) compresses the matrix at the level
 * of @c R-by-C dense blocks. The layout's offsets array indexes
 * *block-rows*, and an atom is a *block id* (not a scalar nonzero):
 *
 *   - num_tiles == num_block_rows == ceil(rows / R)
 *   - num_atoms == num_blocks (total stored R-by-C blocks)
 *   - tile_size(br) == number of stored blocks in block-row br
 *
 * The R/C block dimensions live in the @c bcsr_t container and the
 * kernel template; the layout itself is dimension-agnostic, just like
 * @c layout::csr , so the same schedules work unchanged. The kernel,
 * given an atom (block id), reaches into @c values[atom * R * C + ...]
 * and @c block_col_indices[atom] to do its R-by-C dense update.
 *
 * @tparam tile_id_type Index type for tiles (block-row id).
 * @tparam atom_id_type Index type for atoms (block id).
 */
template <typename tile_id_type, typename atom_id_type>
struct bcsr {
  using tile_id_t = tile_id_type;
  using atom_id_t = atom_id_type;
  using tile_end_iterator_t = atom_id_t const*;

  atom_id_t const* offsets_;  /// length num_tiles + 1.
  tile_id_t n_tiles_;
  atom_id_t n_atoms_;

  __host__ __device__ bcsr() : offsets_(nullptr), n_tiles_(0), n_atoms_(0) {}

  __host__ __device__ bcsr(atom_id_t const* offsets,
                           tile_id_t num_tiles,
                           atom_id_t num_atoms)
      : offsets_(offsets), n_tiles_(num_tiles), n_atoms_(num_atoms) {}

  __host__ __device__ tile_id_t num_tiles() const { return n_tiles_; }
  __host__ __device__ atom_id_t num_atoms() const { return n_atoms_; }

  __host__ __device__ atom_id_t tile_begin(tile_id_t t) const {
    return offsets_[t];
  }
  __host__ __device__ atom_id_t tile_end(tile_id_t t) const {
    return offsets_[t + 1];
  }
  __host__ __device__ atom_id_t tile_size(tile_id_t t) const {
    return offsets_[t + 1] - offsets_[t];
  }

  __host__ __device__ tile_end_iterator_t tile_end_iter() const {
    return offsets_ + 1;
  }

  __host__ __device__ tile_id_t tile_of(atom_id_t a) const {
    tile_id_t lo = 0;
    tile_id_t hi = n_tiles_;
    while (lo < hi) {
      tile_id_t mid = lo + ((hi - lo) >> 1);
      if (offsets_[mid + 1] <= a)
        lo = mid + 1;
      else
        hi = mid;
    }
    return lo;
  }
};

/**
 * @brief CSC-shaped tile-atom layout view (tile is a *column*).
 *
 * Structurally identical to @c layout::csr - the offsets array indexes
 * tiles, every tile has @c offsets_[t+1] - offsets_[t] atoms, and tiles
 * are contiguous in atom-id space. The *interpretation* is what differs:
 *
 *   - csr: tile = row, atom_id indexes col_indices/values arrays
 *   - csc: tile = col, atom_id indexes row_indices/values arrays
 *
 * The schedule code is oblivious to which interpretation is active; only
 * the kernel cares (it dereferences the right index array and chooses
 * whether the per-tile output is row-stationary, requiring no atomics,
 * or column-stationary, where each atom in a tile writes a different
 * row of @c y and an @c atomicAdd is needed).
 *
 * Carrying a distinct type for CSC (instead of an alias for CSR) makes
 * SpMV kernels self-documenting: a kernel templated on
 * @c layout::csc<...> declares "I expect column-major data and will
 * atomic-add by row". A kernel templated on @c layout::csr<...> declares
 * "I expect row-major data and will accumulate locally per tile".
 *
 * @tparam tile_id_type Index type for tiles (column id).
 * @tparam atom_id_type Index type for atoms (flat nnz position).
 */
template <typename tile_id_type, typename atom_id_type>
struct csc {
  using tile_id_t = tile_id_type;
  using atom_id_t = atom_id_type;
  using tile_end_iterator_t = atom_id_t const*;

  atom_id_t const*
      offsets_;  /// length num_tiles + 1, monotonically non-decreasing.
  tile_id_t n_tiles_;
  atom_id_t n_atoms_;

  __host__ __device__ csc() : offsets_(nullptr), n_tiles_(0), n_atoms_(0) {}

  __host__ __device__ csc(atom_id_t const* offsets,
                          tile_id_t num_tiles,
                          atom_id_t num_atoms)
      : offsets_(offsets), n_tiles_(num_tiles), n_atoms_(num_atoms) {}

  __host__ __device__ tile_id_t num_tiles() const { return n_tiles_; }
  __host__ __device__ atom_id_t num_atoms() const { return n_atoms_; }

  __host__ __device__ atom_id_t tile_begin(tile_id_t t) const {
    return offsets_[t];
  }
  __host__ __device__ atom_id_t tile_end(tile_id_t t) const {
    return offsets_[t + 1];
  }
  __host__ __device__ atom_id_t tile_size(tile_id_t t) const {
    return offsets_[t + 1] - offsets_[t];
  }

  __host__ __device__ tile_end_iterator_t tile_end_iter() const {
    return offsets_ + 1;
  }

  __host__ __device__ tile_id_t tile_of(atom_id_t a) const {
    tile_id_t lo = 0;
    tile_id_t hi = n_tiles_;
    while (lo < hi) {
      tile_id_t mid = lo + ((hi - lo) >> 1);
      if (offsets_[mid + 1] <= a)
        lo = mid + 1;
      else
        hi = mid;
    }
    return lo;
  }
};

/**
 * @brief COO-shaped tile-atom layout view (one tile per nonzero).
 *
 * Coordinate (COO) format stores each nonzero as an independent
 * @c (row, col, value) triple with no implicit grouping. The format-native
 * mapping into the layout contract is therefore:
 *
 *   - @c num_tiles == @c num_atoms == @c nnz
 *   - @c tile_size(t) == 1 for every tile
 *   - @c tile_begin(t) == t , @c tile_end(t) == t+1
 *
 * This is a *degenerate* but valid layout: it's perfect for kernels that
 * want one thread per nonzero with atomic-add output (the canonical COO
 * SpMV pattern), and it lets us drive any of the existing schedules over
 * COO data without first converting to CSR.
 *
 * If you have a *sorted* COO and want tile=row instead of tile=NZ, build
 * an offsets array and use @c layout::csr - that's structurally what CSR
 * is for. If you want every K nonzeros to form a tile (across rows),
 * wrap this layout in @c layout::flat_uniform_occupancy<K, layout::coo>.
 *
 * @tparam tile_id_type Tile-id type (one per nonzero, so equals nnz).
 * @tparam atom_id_type Atom-id type (one per nonzero, so equals nnz).
 */
template <typename tile_id_type, typename atom_id_type>
struct coo {
  using tile_id_t = tile_id_type;
  using atom_id_t = atom_id_type;
  using tile_end_iterator_t = thrust::counting_iterator<atom_id_t>;

  atom_id_t n_nzs_;  /// total non-zeros == num_tiles == num_atoms.

  __host__ __device__ coo() : n_nzs_(0) {}

  __host__ __device__ explicit coo(atom_id_t nnz) : n_nzs_(nnz) {}

  __host__ __device__ tile_id_t num_tiles() const {
    return static_cast<tile_id_t>(n_nzs_);
  }
  __host__ __device__ atom_id_t num_atoms() const { return n_nzs_; }

  __host__ __device__ atom_id_t tile_begin(tile_id_t t) const {
    return static_cast<atom_id_t>(t);
  }
  __host__ __device__ atom_id_t tile_end(tile_id_t t) const {
    return static_cast<atom_id_t>(t) + 1;
  }
  __host__ __device__ atom_id_t tile_size(tile_id_t /*t*/) const {
    return atom_id_t{1};
  }

  /// @c counting_iterator(1) gives @c i[k] == @c k+1 == @c tile_end(k).
  __host__ __device__ tile_end_iterator_t tile_end_iter() const {
    return thrust::counting_iterator<atom_id_t>(1);
  }

  /// O(1): @c tile_of(a) is just @c a, since every nonzero is its own tile.
  __host__ __device__ tile_id_t tile_of(atom_id_t a) const {
    return static_cast<tile_id_t>(a);
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
 * @tparam tile_id_type Tile-id type (e.g., row id).
 * @tparam atom_id_type Atom-id type (flat index into the per-row buckets).
 */
template <typename tile_id_type, typename atom_id_type>
struct ell {
 private:
  /// Functor used to materialize tile_end values lazily.
  struct tile_end_fn {
    atom_id_type pitch;
    __host__ __device__ atom_id_type operator()(tile_id_type i) const {
      return static_cast<atom_id_type>(i + 1) * pitch;
    }
  };

 public:
  using tile_id_t = tile_id_type;
  using atom_id_t = atom_id_type;
  using tile_end_iterator_t =
      thrust::transform_iterator<tile_end_fn,
                                 thrust::counting_iterator<tile_id_t>,
                                 atom_id_t>;

  tile_id_t n_tiles_;
  atom_id_t
      pitch_;  /// atoms per tile (uniform); = max-non-zeros-per-row in SpMV.

  __host__ __device__ ell() : n_tiles_(0), pitch_(0) {}

  __host__ __device__ ell(tile_id_t num_tiles, atom_id_t pitch)
      : n_tiles_(num_tiles), pitch_(pitch) {}

  __host__ __device__ tile_id_t num_tiles() const { return n_tiles_; }
  __host__ __device__ atom_id_t num_atoms() const {
    return static_cast<atom_id_t>(n_tiles_) * pitch_;
  }

  __host__ __device__ atom_id_t tile_begin(tile_id_t t) const {
    return static_cast<atom_id_t>(t) * pitch_;
  }
  __host__ __device__ atom_id_t tile_end(tile_id_t t) const {
    return static_cast<atom_id_t>(t + 1) * pitch_;
  }
  __host__ __device__ atom_id_t tile_size(tile_id_t /*t*/) const {
    return pitch_;
  }

  /// Random-access iterator @c i where @c i[k] @c == @c tile_end(k).
  __host__ __device__ tile_end_iterator_t tile_end_iter() const {
    return thrust::make_transform_iterator(
        thrust::counting_iterator<tile_id_t>(0), tile_end_fn{pitch_});
  }

  /// Tile-id (row) that owns atom @c a. O(1) for ELL.
  __host__ __device__ tile_id_t tile_of(atom_id_t a) const {
    return static_cast<tile_id_t>(a / pitch_);
  }
};

}  // namespace layout
}  // namespace loops

#include <loops/container/partitioning.hxx>
