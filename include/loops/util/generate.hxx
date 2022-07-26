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
#include <chrono>
#include <thrust/random.h>
#include <thrust/distance.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

#include <loops/memory.hxx>
#include <loops/container/formats.hxx>

namespace loops {
namespace generate {
namespace random {

/**
 * @brief Hash function for random numbers.
 *
 * @param a The number to hash.
 * @return unsigned int The hashed number.
 */
__forceinline__ __host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

/**
 * @brief Generate a vector with uniform distribution of random numbers between
 * [begin, end].
 *
 * @tparam vector_t vector type.
 * @tparam vector_t::value_type value type.
 * @param input thrust host or device vector.
 * @param begin range begin value.
 * @param end range end value.
 * @param seed random seed.
 */
template <typename iterator_t, typename type_t>
void uniform_distribution(iterator_t begin_it,
                          iterator_t end_it,
                          type_t begin,
                          type_t end,
                          unsigned int useed = std::chrono::system_clock::now()
                                                   .time_since_epoch()
                                                   .count()) {
  int size = thrust::distance(begin_it, end_it);
  auto generate_random = [=] __host__ __device__(std::size_t i) -> type_t {
    unsigned int seed = hash(i) * useed;
    if (std::is_floating_point_v<type_t>) {
      thrust::default_random_engine rng(seed);
      thrust::uniform_real_distribution<type_t> uniform(begin, end);
      return uniform(rng);
    } else {
      thrust::default_random_engine rng(seed);
      thrust::uniform_int_distribution<type_t> uniform(begin, end);
      return uniform(rng);
    }
  };

  thrust::transform(thrust::make_counting_iterator<std::size_t>(0),
                    thrust::make_counting_iterator<std::size_t>(size), begin_it,
                    generate_random);
}

using namespace memory;

/**
 * @brief Generates a random Compressed Sparse Row (CSR) matrix.
 *
 * @tparam index_t index type.
 * @tparam offset_t offset type.
 * @tparam value_t value type.
 * @param rows number of rows.
 * @param columns number of columns.
 * @param sparsity sparsity ratio of the matrix.
 * @return csr_t<index_t, offset_t, value_t> random CSR matrix.
 */
template <typename index_t, typename offset_t, typename value_t>
void csr(std::size_t rows,
         std::size_t cols,
         float sparsity,
         csr_t<index_t, offset_t, value_t>& matrix) {
  std::size_t nnzs = sparsity * (rows * cols);
  coo_t<index_t, value_t, memory_space_t::host> coo(rows, cols, nnzs);

  // Generate Random indices and values.
  uniform_distribution(coo.row_indices.begin(), coo.row_indices.end(), 0,
                       index_t(rows));
  uniform_distribution(coo.col_indices.begin(), coo.col_indices.end(),
                       index_t(0), index_t(cols));
  uniform_distribution(coo.values.begin(), coo.values.end(), value_t(0.0),
                       value_t(1.0));

  // Remove duplicates.
  coo.remove_duplicates();
  matrix = coo;
}

}  // namespace random
}  // namespace generate
}  // namespace loops