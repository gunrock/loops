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
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <loops/memory.hxx>

namespace loops {
namespace detail {
using namespace memory;

/**
 * @brief Convert offsets to indices.
 *
 * @tparam index_v_t The type of vector indices.
 * @tparam offset_v_t The type of vector offsets.
 * @param offsets The offsets.
 * @param indices The indices.
 */
template <typename index_v_t, typename offset_v_t>
void offsets_to_indices(const offset_v_t& offsets, index_v_t& indices) {
  using offset_t = typename offset_v_t::value_type;
  using index_t = typename index_v_t::value_type;

  // Convert compressed offsets into uncompressed indices.
  thrust::fill(indices.begin(), indices.end(), offset_t(0));

  thrust::scatter_if(
      thrust::counting_iterator<offset_t>(0),  // begin iterator
      thrust::counting_iterator<offset_t>(offsets.size() - 1),  // end iterator
      offsets.begin(),  // where to scatter
      thrust::make_transform_iterator(
          thrust::make_zip_iterator(
              thrust::make_tuple(offsets.begin(), offsets.begin() + 1)),
          [=] __host__ __device__(const thrust::tuple<offset_t, offset_t>& t) {
            thrust::not_equal_to<offset_t> comp;
            return comp(thrust::get<0>(t), thrust::get<1>(t));
          }),
      indices.begin());

  thrust::inclusive_scan(indices.begin(), indices.end(), indices.begin(),
                         thrust::maximum<offset_t>());
}

/**
 * @brief Converts "indices"-based array to "offsets"-based array.
 *
 * @tparam index_v_t The type of vector indices.
 * @tparam offset_v_t The type of vector offsets.
 * @param indices The indices.
 * @param offsets The offsets.
 */
template <typename index_v_t, typename offset_v_t>
void indices_to_offsets(const index_v_t& indices, offset_v_t& offsets) {
  using offset_t = typename offset_v_t::value_type;
  using index_t = typename index_v_t::value_type;

  // Convert uncompressed indices into compressed offsets.
  thrust::lower_bound(
      indices.begin(), indices.end(), thrust::counting_iterator<offset_t>(0),
      thrust::counting_iterator<offset_t>(offsets.size()), offsets.begin());
}

}  // namespace detail
}  // namespace loops