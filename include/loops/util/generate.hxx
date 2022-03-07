/**
 * @file generate.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2022-02-02
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <thrust/distance.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>

namespace loops {
namespace generate {
namespace random {

/**
 * @brief Generate a vector with uniform distribution of random numbers between
 * [begin, end].
 *
 * @tparam vector_t vector type.
 * @tparam vector_t::value_type value type.
 * @param input thrust host or device vector.
 * @param begin begin value (default: 0.0f).
 * @param end end value (default: 1.0f).
 */
template <typename iterator_t, typename type_t = float>
void uniform_distribution(iterator_t begin_it,
                          iterator_t end_it,
                          type_t begin = 0.0f,
                          type_t end = 1.0f) {
  int size = thrust::distance(begin_it, end_it);
  auto generate_random = [=] __host__ __device__(std::size_t i) -> type_t {
    thrust::minstd_rand rng;
    thrust::uniform_real_distribution<type_t> uniform(begin, end);
    rng.discard(i);
    return uniform(rng);
  };

  thrust::transform(thrust::make_counting_iterator<std::size_t>(0),
                    thrust::make_counting_iterator<std::size_t>(size), begin_it,
                    generate_random);
}
}  // namespace random
}  // namespace generate
}  // namespace loops