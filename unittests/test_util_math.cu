/**
 * @file test_util_math.cu
 * @author Loops contributors
 * @brief Edge-case tests for @c loops::math utilities.
 *
 * @c math::ceil_div is used everywhere a grid size is computed; an
 * off-by-one here silently corrupts every kernel launch downstream.
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <catch2/catch_test_macros.hpp>

#include <loops/util/math.hxx>

#include <cstdint>

using namespace loops;

TEST_CASE("math::ceil_div basic divisible cases", "[util][math][ceil_div]") {
  CHECK(math::ceil_div(0, 1) == 0);
  CHECK(math::ceil_div(1, 1) == 1);
  CHECK(math::ceil_div(10, 5) == 2);
  CHECK(math::ceil_div(100, 10) == 10);
}

TEST_CASE("math::ceil_div rounds up on remainders",
          "[util][math][ceil_div]") {
  CHECK(math::ceil_div(1, 2) == 1);
  CHECK(math::ceil_div(3, 2) == 2);
  CHECK(math::ceil_div(7, 3) == 3);
  CHECK(math::ceil_div(11, 4) == 3);
  CHECK(math::ceil_div(127, 32) == 4);
  CHECK(math::ceil_div(128, 32) == 4);
  CHECK(math::ceil_div(129, 32) == 5);
}

TEST_CASE("math::ceil_div handles tiny numerator", "[util][math][ceil_div]") {
  CHECK(math::ceil_div(0, 100) == 0);
  CHECK(math::ceil_div(1, 100) == 1);
  CHECK(math::ceil_div(99, 100) == 1);
  CHECK(math::ceil_div(100, 100) == 1);
  CHECK(math::ceil_div(101, 100) == 2);
}

TEST_CASE("math::ceil_div handles std::size_t (CSR rows / block_size case)",
          "[util][math][ceil_div][size_t]") {
  // Mirror the call sites in @c spmv/*.cu where rows is std::size_t and
  // block_size is a constexpr.
  constexpr std::size_t block_size = 128;
  CHECK(math::ceil_div(static_cast<std::size_t>(0), block_size) == 0);
  CHECK(math::ceil_div(static_cast<std::size_t>(1), block_size) == 1);
  CHECK(math::ceil_div(static_cast<std::size_t>(127), block_size) == 1);
  CHECK(math::ceil_div(static_cast<std::size_t>(128), block_size) == 1);
  CHECK(math::ceil_div(static_cast<std::size_t>(129), block_size) == 2);
  CHECK(math::ceil_div(static_cast<std::size_t>(1024), block_size) == 8);
}

TEST_CASE("math::ceil_div is overflow-resistant near INT64_MAX",
          "[util][math][ceil_div][overflow]") {
  // The naive (a + b - 1) / b would overflow; the implementation in
  // @c include/loops/util/math.hxx uses (n / d + (n % d != 0)) instead.
  constexpr std::int64_t big = std::numeric_limits<std::int64_t>::max();
  CHECK(math::ceil_div(big, std::int64_t{1}) == big);
  CHECK(math::ceil_div(big - 1, std::int64_t{2}) == big / 2);
}
