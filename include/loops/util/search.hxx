/**
 * @file search.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Simple search functionality.
 * @version 0.1
 * @date 2022-11-12
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <thrust/binary_search.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/distance.h>
#include <thrust/iterator/counting_iterator.h>

#include <loops/container/coordinate.hxx>

#pragma once

namespace loops {
namespace search {
/**
 * @brief Thrust based 2D binary-search for merge-path algorithm.
 *
 * @param diagonal Diagonal of the search.
 * @param a First iterator.
 * @param b Second iterator.
 * @param a_len Length of the first iterator.
 * @param b_len Length of the second iterator.
 * @return A coordinate.
 */
template <typename offset_t, typename xit_t, typename yit_t>
__device__ __forceinline__ auto _binary_search(const offset_t& diagonal,
                                               const xit_t a,
                                               const yit_t b,
                                               const offset_t& a_len,
                                               const offset_t& b_len) {
  using coord_idx_t = unsigned int;
  /// Diagonal search range (in x-coordinate space)
  /// Note that the subtraction can result into a negative number, in which
  /// case the max would result as 0. But if we use offset_t here, and it is
  /// an unsigned type, we would get strange behavior, possible an unwanted
  /// sign conversion that we do not want.
  int x_min = max(int(diagonal) - int(b_len), int(0));
  int x_max = min(int(diagonal), int(a_len));

  auto it = thrust::lower_bound(
      thrust::seq,                                 // Sequential impl
      thrust::counting_iterator<offset_t>(x_min),  // Start iterator @x_min
      thrust::counting_iterator<offset_t>(x_max),  // End iterator @x_max
      diagonal,                                    // ...
      [=] __device__(const offset_t& idx, const offset_t& diagonal) {
        return a[idx] <= b[diagonal - idx - 1];
      });

  return coordinate_t<coord_idx_t>{static_cast<coord_idx_t>(min(*it, a_len)),
                                   static_cast<coord_idx_t>(diagonal - *it)};
}
}  // namespace search
}  // namespace loops