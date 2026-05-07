/**
 * @file partitioning.hxx
 * @author Loops contributors
 * @brief Layout adaptors that re-partition atoms into new tile groupings.
 * @version 0.1
 * @date 2026-05-06
 *
 * The schedules consume workloads through the layout contract in
 * @c loops/container/layout.hxx (see @ref layout_contract). A
 * **partitioner** is a layout view that *adapts* an underlying base layout
 * into a new tile grouping while still satisfying the same contract, so it
 * can be plugged into any schedule without changing the schedule itself.
 *
 * @par Why partitioners are layouts
 * The partitioner does not represent a new sparse format; it represents a
 * different *grouping* of the atoms exposed by some other layout. Since
 * the schedules only ever ask "how many tiles? what atoms in tile t?", the
 * partitioner answers those questions on top of any base layout. The
 * scheduling code (thread_mapped, group_mapped, work_oriented,
 * merge_path_flat) is unchanged.
 *
 * @par Available partitioners
 * - @c flat_uniform_occupancy<K, base_layout_type> : stateless,
 *   pure-math. Tiles are size-K windows into the *flat* atom enumeration of
 *   the base layout; the last tile may be smaller. Tiles freely cross the
 *   base layout's natural tile boundaries (e.g., CSR rows), so kernels
 *   that need per-atom output addressing (such as SpMV's @c y[row]+= )
 *   should query @c partitioner.base().tile_of(atom) and atomic-add.
 *
 * @par How to plug a partitioner into a schedule
 * @code{.cpp}
 * using base_t  = loops::layout::csr<int, int>;
 * using lay_t   = loops::layout::flat_uniform_occupancy<8, base_t>;
 * using setup_t = loops::schedule::setup<
 *     loops::schedule::algorithms_t::thread_mapped,
 *     128, 1, int, int, std::size_t, std::size_t, lay_t>;
 *
 * base_t base(csr.offsets.data().get(), csr.rows, csr.nnzs);
 * lay_t  partitioned(base);
 * setup_t config(partitioned);
 * @endcode
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace loops {
namespace layout {

/**
 * @brief Flatten + chunk a base layout into K-atom tiles.
 *
 * Given a base layout that exposes @c num_atoms() atoms in some natural
 * ordering, this adaptor re-bins them into @c ceil(num_atoms / K) tiles
 * of size @c K (the last tile may be smaller if @c num_atoms is not
 * divisible by @c K). Tile boundaries are pure math and require no
 * precompute — this is the cheapest possible partitioner.
 *
 * Tile @c t covers atoms @c [t*K, min((t+1)*K, num_atoms)) in the base
 * atom enumeration. The base layout's *natural* tile partition (e.g., CSR
 * rows) is preserved on @c partitioned.base() for kernels that need it.
 *
 * @tparam K              Number of atoms per tile (compile-time constant).
 * @tparam base_layout_type Underlying layout (e.g., @c layout::csr ).
 */
template <std::size_t K, typename base_layout_type>
struct flat_uniform_occupancy {
  static_assert(K > 0, "flat_uniform_occupancy: K must be positive.");

 private:
  /// Functor used to materialize tile_end values lazily.
  struct tile_end_fn {
    typename base_layout_type::atom_id_t k;
    typename base_layout_type::atom_id_t total;
    __host__ __device__ typename base_layout_type::atom_id_t operator()(
        typename base_layout_type::tile_id_t i) const {
      auto end = static_cast<typename base_layout_type::atom_id_t>(i + 1) * k;
      return end < total ? end : total;
    }
  };

 public:
  using base_layout_t = base_layout_type;
  using tile_id_t = typename base_layout_t::tile_id_t;
  using atom_id_t = typename base_layout_t::atom_id_t;
  using tile_end_iterator_t = thrust::transform_iterator<
      tile_end_fn,
      thrust::counting_iterator<tile_id_t>,
      atom_id_t>;

  static constexpr atom_id_t kAtomsPerTile = static_cast<atom_id_t>(K);

  base_layout_t base_;

  __host__ __device__ flat_uniform_occupancy() : base_() {}

  __host__ __device__ explicit flat_uniform_occupancy(base_layout_t base)
      : base_(base) {}

  /// Direct read access to the underlying base layout (advanced use; the
  /// kernel uses this to recover the *original* tile-id of an atom).
  __host__ __device__ const base_layout_t& base() const { return base_; }

  __host__ __device__ tile_id_t num_tiles() const {
    auto n = base_.num_atoms();
    return static_cast<tile_id_t>((n + kAtomsPerTile - 1) / kAtomsPerTile);
  }

  __host__ __device__ atom_id_t num_atoms() const { return base_.num_atoms(); }

  __host__ __device__ atom_id_t tile_begin(tile_id_t t) const {
    return static_cast<atom_id_t>(t) * kAtomsPerTile;
  }

  __host__ __device__ atom_id_t tile_end(tile_id_t t) const {
    auto end = static_cast<atom_id_t>(t + 1) * kAtomsPerTile;
    auto total = base_.num_atoms();
    return end < total ? end : total;
  }

  __host__ __device__ atom_id_t tile_size(tile_id_t t) const {
    return tile_end(t) - tile_begin(t);
  }

  /// Random-access iterator @c i where @c i[k] @c == @c tile_end(k).
  __host__ __device__ tile_end_iterator_t tile_end_iter() const {
    return thrust::make_transform_iterator(
        thrust::counting_iterator<tile_id_t>(0),
        tile_end_fn{kAtomsPerTile, base_.num_atoms()});
  }

  /// Post-partition tile-id that owns atom @c a (i.e., @c a / K ). O(1).
  __host__ __device__ tile_id_t tile_of(atom_id_t a) const {
    return static_cast<tile_id_t>(a / kAtomsPerTile);
  }
};

}  // namespace layout
}  // namespace loops
