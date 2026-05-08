/**
 * @file test_layout_dia.cu
 * @author Loops contributors
 * @brief Layout-contract conformance for @c loops::layout::dia .
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <catch2/catch_test_macros.hpp>

#include <loops/container/layout.hxx>

#include "test_layout_contract.hxx"

using namespace loops;
using namespace loops::testing;

TEST_CASE("layout::dia satisfies the layout contract", "[layout][dia]") {
  // 4 rows, 3 stored diagonals -> 12 atoms.
  layout::dia<int, int> lay(/*num_rows=*/4, /*num_diags=*/3);

  CHECK(lay.num_tiles() == 4);
  CHECK(lay.num_atoms() == 12);

  check_layout_invariants(lay, /*expected_atoms=*/12);
  check_tile_of_round_trip(lay);

  SECTION("every row has exactly num_diagonals atoms") {
    for (int r = 0; r < lay.num_tiles(); ++r) {
      CHECK(lay.tile_size(r) == 3);
      CHECK(lay.tile_begin(r) == r * 3);
      CHECK(lay.tile_end(r) == (r + 1) * 3);
    }
  }

  SECTION("tile_of is a / pitch") {
    CHECK(lay.tile_of(0) == 0);
    CHECK(lay.tile_of(2) == 0);
    CHECK(lay.tile_of(3) == 1);
    CHECK(lay.tile_of(11) == 3);
  }
}

TEST_CASE("layout::dia with zero rows is valid", "[layout][dia][edge]") {
  layout::dia<int, int> lay(0, 5);
  CHECK(lay.num_tiles() == 0);
  CHECK(lay.num_atoms() == 0);
  check_layout_invariants(lay, 0);
}

TEST_CASE("layout::dia with one diagonal collapses to one atom per row",
          "[layout][dia][edge]") {
  layout::dia<int, int> lay(8, 1);
  check_layout_invariants(lay, 8);
  check_tile_of_round_trip(lay);
  for (int r = 0; r < 8; ++r) {
    CHECK(lay.tile_size(r) == 1);
    CHECK(lay.tile_of(r) == r);
  }
}
