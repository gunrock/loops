/**
 * @file test_layout_contract.hxx
 * @author Loops contributors
 * @brief Cross-layout invariant checkers used by every test_layout_*.cu .
 *
 * The layout contract documented in @c include/loops/container/layout.hxx
 * is the same for every layout, so the checks live here and each
 * per-layout test file just calls them with its specific layout view.
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <catch2/catch_test_macros.hpp>

namespace loops {
namespace testing {

/**
 * @brief Verify the universal layout contract on @c lay :
 *  - tile_begin(0) == 0
 *  - tile_end(num_tiles - 1) == num_atoms
 *  - tile_end(t) is monotonically non-decreasing
 *  - tile_end_iter()[k] == tile_end(k)
 *  - tile_size(t) == tile_end(t) - tile_begin(t)
 *  - num_atoms matches the caller-asserted expectation.
 */
template <typename layout_t>
void check_layout_invariants(const layout_t& lay,
                             typename layout_t::atom_id_t expected_atoms) {
  using tile_id_t = typename layout_t::tile_id_t;
  using atom_id_t = typename layout_t::atom_id_t;

  const auto T = lay.num_tiles();
  const auto A = lay.num_atoms();

  CHECK(A == expected_atoms);

  if (T == 0)
    return;

  CHECK(lay.tile_begin(tile_id_t{0}) == atom_id_t{0});
  CHECK(lay.tile_end(static_cast<tile_id_t>(T - 1)) == A);

  auto it = lay.tile_end_iter();
  atom_id_t last_end = 0;
  for (tile_id_t t = 0; t < T; ++t) {
    auto begin = lay.tile_begin(t);
    auto end = lay.tile_end(t);
    auto size = lay.tile_size(t);

    CHECK(begin <= end);
    CHECK(end - begin == size);
    CHECK(end >= last_end);  // monotonicity
    CHECK(it[t] == end);     // tile_end_iter agrees with tile_end

    last_end = end;
  }
}

/**
 * @brief Verify the optional @c tile_of round-trip:
 *
 *   For every non-empty tile @c t , @c tile_of(tile_begin(t)) == @c t and
 *   @c tile_of(tile_end(t) - 1) == @c t .
 */
template <typename layout_t>
void check_tile_of_round_trip(const layout_t& lay) {
  using tile_id_t = typename layout_t::tile_id_t;
  using atom_id_t = typename layout_t::atom_id_t;

  const auto T = lay.num_tiles();
  if (T == 0)
    return;

  for (tile_id_t t = 0; t < T; ++t) {
    auto begin = lay.tile_begin(t);
    auto end = lay.tile_end(t);
    if (begin == end)
      continue;
    CHECK(lay.tile_of(begin) == t);
    if (end > begin) {
      CHECK(lay.tile_of(static_cast<atom_id_t>(end - 1)) == t);
    }
  }
}

}  // namespace testing
}  // namespace loops
