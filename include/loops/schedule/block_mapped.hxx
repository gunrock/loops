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

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/distance.h>

#define _CG_ABI_EXPERIMENTAL

#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>

namespace loops {
namespace schedule {

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

template <std::size_t THREADS_PER_BLOCK,
          std::size_t THREADS_PER_TILE,
          typename tiles_type,
          typename atoms_type,
          typename tile_size_type,
          typename atom_size_type>
class setup<algroithms_t::block_mapped,
            THREADS_PER_BLOCK,
            THREADS_PER_TILE,
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
  using tile_storage_t = tiles_t;

  using tile_traits_t =
      tile_traits<algroithms_t::block_mapped, tiles_type, tile_size_type>;
  using atom_traits_t =
      atom_traits<algroithms_t::block_mapped, atoms_type, atom_size_type>;

  enum : unsigned int {
    threads_per_block = THREADS_PER_BLOCK,
    threads_per_tile = THREADS_PER_TILE,
    tiles_per_block = THREADS_PER_BLOCK / THREADS_PER_TILE,
  };

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
  __device__ step_range_t<atoms_t> virtual_atoms(storage_t* st,
                                                 storage_t* th_st,
                                                 storage_t* sh_aggregates,
                                                 cg_block_tile_t& partition) {
    storage_t* p_st = st + (partition.meta_group_rank() * threads_per_tile);

    p_st[partition.thread_rank()] =
        cooperative_groups::exclusive_scan(partition, th_st[0]);
    partition.sync();

#ifdef DEBUG_SCHEDULE
    // printf("threadIdx.x = %d, nnzs = %d, p_st[%d] : offset = %d\n",
    //        (int)threadIdx.x, (int)atom_to_process,
    //        (int)partition.thread_rank(), (int)p_st[partition.thread_rank()]);
#endif
    if (partition.thread_rank() == partition.size() - 1) {
#ifdef DEBUG_SCHEDULE
      // printf("Last Thread of Every Partition = %d, th_st[0] = %d, p_st[] =
      //        % d\n ", (int)partition.thread_rank(), (int)th_st[0],
      //        (int)p_st[partition.thread_rank()]);
#endif
      // Accumulate tiled aggregates.
      sh_aggregates[partition.meta_group_rank()] =
          p_st[partition.thread_rank()] + th_st[0];
    }
    partition.sync();
    atoms_t aggregate_atoms = sh_aggregates[partition.meta_group_rank()];
    return custom_stride_range(atoms_t(partition.thread_rank()),
                               aggregate_atoms, atoms_t(partition.size()));
  }

  template <typename cg_block_tile_t>
  __device__ __forceinline__ int get_length(cg_block_tile_t& partition) {
    auto g = cooperative_groups::this_grid();
    auto b = cooperative_groups::this_thread_block();

    auto thread_id = g.thread_rank();
    auto block_id = b.thread_rank();
    auto local_id = partition.thread_rank();

    int length = thread_id - local_id + partition.size();
    if (tile_traits_t::size() < length)
      length = tile_traits_t::size();

    length -= thread_id - local_id;
    return length;
  }

  template <typename cg_block_tile_t>
  __device__ tiles_t tile_id(storage_t* st,
                             atoms_t& virtual_atom,
                             cg_block_tile_t& partition) {
    int length = get_length(partition);
    storage_t* p_st = st + (partition.meta_group_rank() * threads_per_tile);
    auto it =
        thrust::upper_bound(thrust::seq, p_st, p_st + length, virtual_atom);
    auto x = thrust::distance(p_st, it) - 1;
    return x;
  }

  template <typename cg_block_tile_t>
  __device__ tiles_t is_valid_tile(tiles_t& tile_id,
                                   cg_block_tile_t& partition) {
    return tile_id < get_length(partition);
  }

  template <typename cg_block_tile_t>
  __device__ atoms_t atom_id(storage_t* st,
                             atoms_t& v_atom,
                             tiles_t& tile_id,
                             tiles_t& v_tile_id,
                             cg_block_tile_t& partition) {
#ifdef DEBUG_SCHEDULE
    // printf("Shared Memory: %d @ %d/%d\n",
    //        (int)(partition.meta_group_rank() * partition.size()),
    //        (int)partition.meta_group_rank(), (int)partition.size());
#endif
    storage_t* p_st = st + (partition.meta_group_rank() * threads_per_tile);
    return tile_traits_t::begin()[tile_id] + v_atom - p_st[v_tile_id];
  }
};  // namespace schedule

}  // namespace schedule
}  // namespace loops