/**
 * @file test_layout_ell.cu
 * @author Loops contributors
 * @brief Layout-contract conformance for @c loops::layout::ell .
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <catch2/catch_test_macros.hpp>

#include <loops/container/layout.hxx>

#include "test_layout_contract.hxx"

using namespace loops;
using namespace loops::testing;

TEST_CASE("layout::ell satisfies the layout contract", "[layout][ell]") {
  // 5 rows, pitch = 3 -> 15 atoms.
  layout::ell<int, int> lay(/*num_tiles=*/5, /*pitch=*/3);

  check_layout_invariants(lay, /*expected_atoms=*/15);
  check_tile_of_round_trip(lay);

  SECTION("tile_of is a / pitch") {
    CHECK(lay.tile_of(0) == 0);
    CHECK(lay.tile_of(2) == 0);
    CHECK(lay.tile_of(3) == 1);
    CHECK(lay.tile_of(14) == 4);
  }

  SECTION("uniform tile size = pitch for every row") {
    for (int t = 0; t < lay.num_tiles(); ++t) {
      CHECK(lay.tile_size(t) == 3);
      CHECK(lay.tile_begin(t) == t * 3);
      CHECK(lay.tile_end(t) == (t + 1) * 3);
    }
  }
}

TEST_CASE("layout::ell with pitch=1 collapses to one atom per tile",
          "[layout][ell][edge]") {
  layout::ell<int, int> lay(/*num_tiles=*/8, /*pitch=*/1);
  check_layout_invariants(lay, 8);
  check_tile_of_round_trip(lay);
  for (int t = 0; t < 8; ++t) {
    CHECK(lay.tile_size(t) == 1);
    CHECK(lay.tile_of(t) == t);
  }
}

TEST_CASE("layout::ell with zero rows is valid", "[layout][ell][edge]") {
  layout::ell<int, int> lay(0, 5);
  CHECK(lay.num_tiles() == 0);
  CHECK(lay.num_atoms() == 0);
  check_layout_invariants(lay, 0);
}

TEST_CASE("layout::ell with zero pitch yields zero atoms",
          "[layout][ell][edge]") {
  layout::ell<int, int> lay(4, 0);
  CHECK(lay.num_tiles() == 4);
  CHECK(lay.num_atoms() == 0);
  for (int t = 0; t < 4; ++t) CHECK(lay.tile_size(t) == 0);
  check_layout_invariants(lay, 0);
}
