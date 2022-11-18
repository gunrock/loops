/**
 * @file math.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Math related utilities.
 * @version 0.1
 * @date 2022-02-16
 *
 * @copyright Copyright (c) 2022
 *
 */
#pragma once

namespace loops {
namespace math {

/**
 * @brief Simple safe ceil division: (a + b - 1) / b. Handles overflow condition
 * as well.
 *
 * @tparam numerator_t Type of the dividend.
 * @tparam denominator_t Type of the divisor.
 * @param n Dividend.
 * @param d Divisor.
 * @return The quotient.
 */
template <class numerator_t, class denominator_t>
__host__ __device__ __forceinline__ constexpr numerator_t ceil_div(
    numerator_t const& n,
    denominator_t const& d) {
  return static_cast<numerator_t>(n / d + (n % d != 0 ? 1 : 0));
}

}  // namespace math
}  // namespace loops