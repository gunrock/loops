/**
 * @file convert.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Conversion functions for containers.
 * @version 0.1
 * @date 2022-07-19
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <thrust/binary_search.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/scatter.h>
#include <thrust/execution_policy.h>
#include <loops/memory.hxx>

namespace loops {
namespace detail {
using namespace memory;

/**
 * @brief Convert offsets to indices.
 *
 * @tparam space The memory space of offsets.
 * @tparam index_t The type of indices.
 * @tparam offset_t The type of offsets.
 * @param offsets The offsets.
 * @param size_of_offsets The size of offsets.
 * @param indices The indices.
 * @param size_of_indices The size of indices.
 * @param stream The stream.
 */
template <memory_space_t space, typename index_t, typename offset_t>
void offsets_to_indices(const offset_t* offsets,
                        const std::size_t size_of_offsets,
                        index_t* indices,
                        const std::size_t size_of_indices,
                        cudaStream_t stream = 0) {
  // Execution policy (determines where to run the kernel).
  using execution_policy_t =
      std::conditional_t<(space == memory_space_t::device),
                         decltype(thrust::cuda::par.on(stream)),
                         decltype(thrust::host)>;

  execution_policy_t exec;

  // Convert compressed offsets into uncompressed indices.
  thrust::fill(exec, indices + 0, indices + size_of_indices, offset_t(0));

  thrust::scatter_if(
      exec,                                    // execution policy
      thrust::counting_iterator<offset_t>(0),  // begin iterator
      thrust::counting_iterator<offset_t>(size_of_offsets - 1),  // end iterator
      offsets + 0,  // where to scatter
      thrust::make_transform_iterator(
          thrust::make_zip_iterator(
              thrust::make_tuple(offsets + 0, offsets + 1)),
          [=] __host__ __device__(const thrust::tuple<offset_t, offset_t>& t) {
            thrust::not_equal_to<offset_t> comp;
            return comp(thrust::get<0>(t), thrust::get<1>(t));
          }),
      indices + 0);

  thrust::inclusive_scan(exec, indices + 0, indices + size_of_indices,
                         indices + 0, thrust::maximum<offset_t>());
}

/**
 * @brief Converts "indices"-based array to "offsets"-based array.
 *
 * @tparam space The memory space of indices.
 * @tparam index_t The type of indices.
 * @tparam offset_t The type of offsets.
 * @param indices The indices.
 * @param size_of_indices The size of indices.
 * @param offsets The offsets.
 * @param size_of_offsets The size of offsets.
 * @param stream CUDA stream.
 */
template <memory_space_t space, typename index_t, typename offset_t>
void indices_to_offsets(index_t* indices,
                        std::size_t size_of_indices,
                        offset_t* offsets,
                        std::size_t size_of_offsets,
                        cudaStream_t stream = 0) {
  // Execution policy (determines where to run the kernel).
  using execution_policy_t =
      std::conditional_t<(space == memory_space_t::device),
                         decltype(thrust::cuda::par.on(stream)),
                         decltype(thrust::host)>;

  execution_policy_t exec;

  // Convert uncompressed indices into compressed offsets.
  thrust::lower_bound(exec, indices, indices + size_of_indices,
                      thrust::counting_iterator<offset_t>(0),
                      thrust::counting_iterator<offset_t>(size_of_offsets),
                      offsets + 0);
}

}  // namespace detail
}  // namespace loops