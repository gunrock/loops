/**
 * @file test_util_range.cu
 * @author Loops contributors
 * @brief Iteration-shape tests for @c loops::range and friends.
 *
 * The schedule classes lean heavily on @c loops::range to materialize
 * tile / atom iterators. A bug in @c range or @c indices changes the
 * schedule's emitted (tile, atom) sequence and breaks every kernel.
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <catch2/catch_test_macros.hpp>

#include <loops/range.hxx>

#include <vector>

using namespace loops;

TEST_CASE("range(begin, end) yields begin..end-1", "[util][range]") {
  std::vector<int> got;
  for (auto i : range(0, 5))
    got.push_back(i);
  CHECK(got == std::vector<int>{0, 1, 2, 3, 4});
}

TEST_CASE("range(begin, end) is empty when begin == end", "[util][range]") {
  std::vector<int> got;
  for (auto i : range(7, 7))
    got.push_back(i);
  CHECK(got.empty());
}

TEST_CASE("range(begin, end).step(s) skips by s", "[util][range][step]") {
  std::vector<int> got;
  for (auto i : range(0, 10).step(2))
    got.push_back(i);
  CHECK(got == std::vector<int>{0, 2, 4, 6, 8});
}

TEST_CASE("range(begin, end).step(s) handles non-integer-multiple ends",
          "[util][range][step]") {
  std::vector<int> got;
  for (auto i : range(0, 9).step(3))
    got.push_back(i);
  CHECK(got == std::vector<int>{0, 3, 6});
}

TEST_CASE("range(begin, end) supports unsigned types", "[util][range]") {
  std::size_t count = 0;
  for (auto i :
       range(static_cast<std::size_t>(0), static_cast<std::size_t>(4))) {
    (void)i;
    ++count;
  }
  CHECK(count == 4);
}

TEST_CASE("range::indices yields container indices", "[util][range][indices]") {
  std::vector<float> v{1.0f, 2.0f, 3.0f};
  std::vector<std::size_t> got;
  for (auto i : indices(v))
    got.push_back(i);
  CHECK(got == std::vector<std::size_t>{0, 1, 2});
}

TEST_CASE("range::indices on empty container yields nothing",
          "[util][range][indices]") {
  std::vector<float> v;
  std::size_t count = 0;
  for (auto i : indices(v)) {
    (void)i;
    ++count;
  }
  CHECK(count == 0);
}
