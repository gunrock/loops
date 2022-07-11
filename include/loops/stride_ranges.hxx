/**
 * @file stride_ranges.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Stride ranges related functions.
 * @version 0.1
 * @date 2022-02-02
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once
#include <loops/range.hxx>

namespace loops {
template <typename T>
using step_range_t = typename range_proxy<T>::step_range_proxy;

/**
 * @brief Gride stride range defines a range that steps gridDim.x * blockDim.x
 * per step.
 *
 * @tparam T The type of the range.
 * @param begin The beginning of the range (0)
 * @param end End of the range, can be num_elements.
 * @return range_proxy<T>::step_range_proxy range returned.
 */
template <typename T>
__device__ __forceinline__ step_range_t<T> grid_stride_range(T begin, T end) {
  begin += blockDim.x * blockIdx.x + threadIdx.x;
  return range(begin, end).step(gridDim.x * blockDim.x);
}

/**
 * @brief Block stride range defines as a range that steps blockDim.x per step.
 *
 * @tparam T The type of the range.
 * @param begin The beginning of the range (e.g. 0)
 * @param end End of the range, can be num_elements.
 * @return step_range_t<T> range returned.
 */
template <typename T>
__device__ __forceinline__ step_range_t<T> block_stride_range(T begin, T end) {
  return range(begin, end).step(blockDim.x);
}

/**
 * @brief Custom stride range defines as a range that steps with a custom
 * stride.
 *
 * @tparam T The type of the range.
 * @param begin The beginning of the range (e.g. 0)
 * @param end End of the range, can be num_elements.
 * @param stride The stride of the range.
 * @return step_range_t<T> range returned.
 */
template <typename T>
__device__ __forceinline__ step_range_t<T> custom_stride_range(T begin,
                                                               T end,
                                                               T stride) {
  return range(begin, end).step(stride);
}
}  // namespace loops