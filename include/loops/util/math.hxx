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
 * @brief Simple safe ceil division: (a + b - 1) / b.
 *
 * @tparam type_t_t Type of the dividend.
 * @tparam type_u_t Type of the divisor.
 * @param t Dividend.
 * @param u Divisor.
 * @return The quotient.
 */
template <class type_t_t, class type_u_t>
__host__ __device__ constexpr auto ceil_div(type_t_t const& t,
                                            type_u_t const& u) {
  return (t + u - 1) / u;
}

}  // namespace math
}  // namespace loops