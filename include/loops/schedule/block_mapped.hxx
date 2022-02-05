/**
 * @file block_mapped.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2022-02-04
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <loops/grid_stride_range.hxx>
#include <loops/schedule.hxx>

#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>

namespace loops {
namespace schedule {

template <typename key_t, typename index_t>
__host__ __device__ index_t rightmost(const key_t* keys,
                                      const key_t& key,
                                      const index_t count) {
  index_t begin = 0;
  index_t end = count;
  while (begin < end) {
    index_t mid = floor((begin + end) / 2);
    key_t key_ = keys[mid];
    bool pred = key_ > key;
    if (pred) {
      end = mid;
    } else {
      begin = mid + 1;
    }
  }
  return end - 1;
}

template <typename atoms_type, typename atom_size_type>
class atom_traits<algroithms_t::block_mapped, atoms_type, atom_size_type> {
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
class tile_traits<algroithms_t::block_mapped, tiles_type, tile_size_type> {
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

template <typename tiles_type,
          typename atoms_type,
          typename tile_size_type,
          typename atom_size_type>
class setup<algroithms_t::block_mapped,
            tiles_type,
            atoms_type,
            tile_size_type,
            atom_size_type> : public tile_traits<algroithms_t::block_mapped,
                                                 tiles_type,
                                                 tile_size_type>,
                              public atom_traits<algroithms_t::block_mapped,
                                                 atoms_type,
                                                 atom_size_type> {
 public:
  using tiles_t = tiles_type;
  using atoms_t = atoms_type;
  using tiles_iterator_t = tiles_t*;
  using atoms_iterator_t = tiles_t*;
  using tile_size_t = tile_size_type;
  using atom_size_t = atom_size_type;

  using storage_t = atoms_t;

  using tile_traits_t =
      tile_traits<algroithms_t::block_mapped, tiles_type, tile_size_type>;
  using atom_traits_t =
      atom_traits<algroithms_t::block_mapped, atoms_type, atom_size_type>;

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

  template <typename cg_block_tile_t>
  __device__ atoms_t work_per_partition(tile_size_t tile_id,
                                        cg_block_tile_t& partition) {
    atoms_t atoms_to_process = 0;
    if (tile_id < tile_traits_t::size()) {
      atoms_to_process =
          tile_traits_t::begin()[tile_id + 1] - tile_traits_t::begin()[tile_id];
    }
    return cooperative_groups::exclusive_scan(partition, atoms_to_process);
  }

  template <typename cg_block_tile_t>
  __device__ atoms_t balance(storage_t* st, cg_block_tile_t& partition) {
    auto tile_id = threadIdx.x + blockIdx.x * blockDim.x;
    st[partition.thread_rank()] = work_per_partition(tile_id, partition);
    partition.sync();
    return st[partition.size() - 1];
  }

  template <typename cg_block_tile_t>
  __device__ step_range_t<atoms_t> virtual_atoms(storage_t* st,
                                                 cg_block_tile_t& partition) {
    return grid_stride_range(atoms_t(partition.thread_rank()),
                             balance(st, partition));
  }

  __device__ tiles_t tile_id(storage_t* st,
                             atoms_t& virtual_atom,
                             tile_size_t& valid_search_size) {
    return rightmost(st, virtual_atom, valid_search_size);
  }

  __device__ atoms_t atom_id(storage_t* st, atoms_t& v_atom, tiles_t& tid) {
    return tile_traits_t::begin()[tid] + v_atom - st[tid];
  }
};

}  // namespace schedule
}  // namespace loops