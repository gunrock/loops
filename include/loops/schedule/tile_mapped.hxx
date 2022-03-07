/**
 * @file tile_mapped.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Tile-mapped schedule (map work to tiles, process using individual
 * threads within the tile.)
 * @version 0.1
 * @date 2022-02-04
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <loops/grid_stride_range.hxx>

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/distance.h>

#ifndef _CG_ABI_EXPERIMENTAL
#define _CG_ABI_EXPERIMENTAL
#endif

#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>

namespace loops {
namespace schedule {

namespace cg = cooperative_groups;
namespace cg_x = cooperative_groups::experimental;

template <typename atoms_type, typename atom_size_type>
class atom_traits<algroithms_t::tile_mapped, atoms_type, atom_size_type> {
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
class tile_traits<algroithms_t::tile_mapped, tiles_type, tile_size_type> {
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
          std::size_t THREADS_PER_TILE,
          typename tiles_type,
          typename atoms_type,
          typename tile_size_type,
          typename atom_size_type>
class setup<algroithms_t::tile_mapped,
            THREADS_PER_BLOCK,
            THREADS_PER_TILE,
            tiles_type,
            atoms_type,
            tile_size_type,
            atom_size_type>
    : public tile_traits<algroithms_t::tile_mapped, tiles_type, tile_size_type>,
      public atom_traits<algroithms_t::tile_mapped,
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
      tile_traits<algroithms_t::tile_mapped, tiles_type, tile_size_type>;
  using atom_traits_t =
      atom_traits<algroithms_t::tile_mapped, atoms_type, atom_size_type>;

  enum : unsigned int {
    threads_per_block = THREADS_PER_BLOCK,
    threads_per_tile = THREADS_PER_TILE,
    tiles_per_block = THREADS_PER_BLOCK / THREADS_PER_TILE,
  };

  /// Temporary storage buffer for schedule algorithm.
  struct __align__(32) storage_t {
    cg_x::block_tile_memory<4, threads_per_block> groups;
    atoms_t tile_aggregates[threads_per_block / threads_per_tile];
    atoms_t atoms_offsets[threads_per_block];
    tiles_t tiles_indices[threads_per_block];
  };

  storage_t& buffer;

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
      : buffer(_buffer),
        tile_traits_t(_num_tiles, _tiles),
        atom_traits_t(_num_atoms) {}

  __device__ __forceinline__ auto partition() {
    auto g = cg::this_grid();
    auto b = cg_x::this_thread_block(buffer.groups);
    auto p = cg_x::tiled_partition<threads_per_tile>(b);

    auto index = g.thread_rank();
    auto tile_index = p.thread_rank() + (p.meta_group_rank() * p.size());
    if (index < tile_traits_t::size()) {
      buffer.tiles_indices[tile_index] = index;
    } else {
      buffer.tiles_indices[tile_index] = -1;
    }
    return p;
  }

  template <typename partition_t>
  __device__ step_range_t<atoms_t> atom_accessor(partition_t& p) {
    atoms_t* p_st =
        buffer.atoms_offsets + (p.meta_group_rank() * threads_per_tile);
    auto g = cg::this_grid();
    auto index = g.thread_rank();
    atoms_t num_atoms = 0;
    if (index < tile_traits_t::size()) {
      num_atoms =
          tile_traits_t::begin()[index + 1] - tile_traits_t::begin()[index];
    }

    p_st[p.thread_rank()] = cg::exclusive_scan(p, num_atoms);
    p.sync();

    if (p.thread_rank() == p.size() - 1) {
      // Accumulate tiled aggregates.
      buffer.tile_aggregates[p.meta_group_rank()] =
          p_st[p.thread_rank()] + num_atoms;
    }

    p.sync();
    atoms_t aggregate_atoms = buffer.tile_aggregates[p.meta_group_rank()];
    return custom_stride_range(atoms_t(p.thread_rank()), aggregate_atoms,
                               atoms_t(p.size()));
  }

  template <typename partition_t>
  __device__ __forceinline__ int get_length(partition_t& p) {
    auto g = cg::this_grid();

    auto thread_id = g.thread_rank();
    auto local_id = p.thread_rank();

    int length = thread_id - local_id + p.size();
    if (tile_traits_t::size() < length)
      length = tile_traits_t::size();

    length -= thread_id - local_id;
    return length;
  }

  template <typename partition_t>
  __device__ __forceinline__ tiles_t tile_accessor(atoms_t& virtual_atom,
                                                   partition_t& p) {
    int length = get_length(p);
    atoms_t* p_st =
        buffer.atoms_offsets + (p.meta_group_rank() * threads_per_tile);
    auto it =
        thrust::upper_bound(thrust::seq, p_st, p_st + length, virtual_atom);
    auto x = thrust::distance(p_st, it) - 1;
    return x;
  }

  template <typename partition_t>
  __device__ __forceinline__ bool is_valid_accessor(tiles_t& tile_id,
                                                    partition_t& p) {
    return tile_id < get_length(p);
  }

  template <typename partition_t>
  __device__ __forceinline__ tiles_t tile_id(tiles_t& v_tile_id,
                                             partition_t& p) {
    return buffer.tiles_indices[v_tile_id + (p.meta_group_rank() * p.size())];
  }

  template <typename partition_t>
  __device__ __forceinline__ atoms_t atom_id(atoms_t& v_atom,
                                             tiles_t& tile_id,
                                             tiles_t& v_tile_id,
                                             partition_t& p) {
    atoms_t* p_st =
        buffer.atoms_offsets + (p.meta_group_rank() * threads_per_tile);
    return tile_traits_t::begin()[tile_id] + v_atom - p_st[v_tile_id];
  }

};  // namespace schedule

}  // namespace schedule
}  // namespace loops