/**
 * @file work_oriented.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Work-oriented scheduling algorithm (map even-share of work to
 * threads.)
 * @version 0.1
 * @date 2022-03-07
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <loops/grid_stride_range.hxx>

#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/pair.h>
#include <thrust/execution_policy.h>

namespace loops {
namespace schedule {

template <class T, class U>
__host__ __device__ constexpr auto div(T const& t, U const& u) {
  return (t + u - 1) / u;
}

template <typename atoms_type, typename atom_size_type>
class atom_traits<algorithms_t::work_oriented, atoms_type, atom_size_type> {
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

template <typename tiles_type, typename tile_size_type>
class tile_traits<algorithms_t::work_oriented, tiles_type, tile_size_type> {
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

template <std::size_t THREADS_PER_BLOCK,
          std::size_t ITEMS_PER_THREAD,
          typename tiles_type,
          typename atoms_type,
          typename tile_size_type,
          typename atom_size_type>
class setup<algorithms_t::work_oriented,
            THREADS_PER_BLOCK,
            ITEMS_PER_THREAD,
            tiles_type,
            atoms_type,
            tile_size_type,
            atom_size_type> : public tile_traits<algorithms_t::work_oriented,
                                                 tiles_type,
                                                 tile_size_type>,
                              public atom_traits<algorithms_t::work_oriented,
                                                 atoms_type,
                                                 atom_size_type> {
 public:
  using tiles_t = tiles_type;
  using atoms_t = atoms_type;
  using tiles_iterator_t = tiles_t*;
  using atoms_iterator_t = atoms_t*;
  using tile_size_t = tile_size_type;
  using atom_size_t = atom_size_type;

  using tile_traits_t =
      tile_traits<algorithms_t::work_oriented, tiles_type, tile_size_type>;
  using atom_traits_t =
      atom_traits<algorithms_t::work_oriented, atoms_type, atom_size_type>;

  enum : unsigned int {
    threads_per_block = THREADS_PER_BLOCK,
    items_per_thread = ITEMS_PER_THREAD,
    items_per_tile = threads_per_block * items_per_thread,
  };

  /**
   * @brief Construct a setup object for load balance schedule.
   *
   * @param tiles Tiles iterator.
   * @param num_tiles Number of tiles.
   * @param num_atoms Number of atoms.
   */
  __device__ __forceinline__ setup(tiles_iterator_t _tiles,
                                   tile_size_t _num_tiles,
                                   atom_size_t _num_atoms)
      : tile_traits_t(_num_tiles, _tiles),
        atom_traits_t(_num_atoms),
        total_work(_num_tiles + _num_atoms),
        num_threads(threads_per_block),  // items_per_tile * (div(total_work,
                                         // items_per_tile))),
        work_per_thread(div(total_work, num_threads)) {}

  __device__ __forceinline__ auto init() {
    thrust::counting_iterator<atoms_t> atoms_indices;

    std::size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    // int hid = (blockIdx.x * gridDim.y) + blockIdx.y;
    // if (threadIdx.x < 2) {
    std::size_t upper = min(work_per_thread * tid, total_work);
    std::size_t lower = min(upper + work_per_thread, total_work);
    auto st = search(upper, tile_traits_t::begin() + 1, atoms_indices,
                     tile_traits_t::size(), atom_traits_t::size());
    auto en = search(lower, tile_traits_t::begin() + 1, atoms_indices,
                     tile_traits_t::size(), atom_traits_t::size());
    // }
    return thrust::make_pair(st, en);
  }

  template <typename map_t>
  __device__ __forceinline__ step_range_t<tiles_t> tiles(map_t& m) const {
    return custom_stride_range(tiles_t(m.first.first), tiles_t(m.second.first),
                               tiles_t(1));
  }

  template <typename map_t>
  __device__ __forceinline__ step_range_t<atoms_t> atoms(tiles_t t, map_t& m) {
    auto num_atoms = tile_traits_t::begin()[(t + 1)];
    auto nz_start = m.first.second;
    m.first.second += (num_atoms - nz_start);
    return custom_stride_range(atoms_t(nz_start), num_atoms, atoms_t(1));
  }

  template <typename map_t>
  __device__ __forceinline__ step_range_t<tiles_t> remainder_tiles(
      map_t& m) const {
    return custom_stride_range(tiles_t(m.second.first), tiles_t(m.second.first),
                               tiles_t(1));
  }

  template <typename map_t>
  __device__ __forceinline__ step_range_t<atoms_t> remainder_atoms(
      map_t& m) const {
    return custom_stride_range(atoms_t(m.first.second),
                               (atoms_t)(m.second.second), atoms_t(1));
  }

 private:
  /**
   * @brief Thrust based 2D binary-search for merge-path algorithm.
   *
   * @param diagonal
   * @param a
   * @param b
   * @param a_len
   * @param b_len
   * @return A coordinate.
   */
  template <typename xit_t, typename yit_t>
  __device__ __forceinline__ auto search(const std::size_t& diagonal,
                                         const xit_t a,
                                         const yit_t b,
                                         const tile_size_t& a_len,
                                         const atom_size_t& b_len) {
    // Diagonal search range (in x-coordinate space)
    std::size_t x_min = max((std::size_t)(diagonal - b_len), std::size_t(0));
    std::size_t x_max = min(diagonal, (std::size_t)a_len);

    auto it = thrust::lower_bound(
        thrust::seq,                                    // Sequential impl
        thrust::counting_iterator<std::size_t>(x_min),  // Start iterator @x_min
        thrust::counting_iterator<std::size_t>(x_max),  // End iterator @x_max
        diagonal,                                       // ...
        [=] __device__(const std::size_t& idx, const std::size_t& diagonal) {
          return a[idx] <= b[diagonal - idx - 1];
        });

    return thrust::make_pair(min(*it, a_len), (diagonal - *it));
  }

  std::size_t total_work;
  std::size_t num_threads;
  std::size_t work_per_thread;
};

}  // namespace schedule
}  // namespace loops