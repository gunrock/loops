/**
 * @file test_layout_coo.cu
 * @author Loops contributors
 * @brief Layout-contract conformance for @c loops::layout::coo .
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <catch2/catch_test_macros.hpp>

#include <loops/container/layout.hxx>

#include "test_layout_contract.hxx"

using namespace loops;
using namespace loops::testing;

TEST_CASE("layout::coo satisfies the layout contract", "[layout][coo]") {
  // 7 nonzeros, one tile per NZ (the degenerate-but-valid layout).
  layout::coo<int, int> lay(/*nnz=*/7);

  CHECK(lay.num_tiles() == 7);
  CHECK(lay.num_atoms() == 7);

  check_layout_invariants(lay, /*expected_atoms=*/7);
  check_tile_of_round_trip(lay);

  SECTION("every tile has exactly one atom") {
    for (int t = 0; t < lay.num_tiles(); ++t) {
      CHECK(lay.tile_size(t) == 1);
      CHECK(lay.tile_begin(t) == t);
      CHECK(lay.tile_end(t) == t + 1);
    }
  }

  SECTION("tile_of is the identity") {
    for (int a = 0; a < lay.num_atoms(); ++a) {
      CHECK(lay.tile_of(a) == a);
    }
  }
}

TEST_CASE("layout::coo with zero nonzeros is valid", "[layout][coo][edge]") {
  layout::coo<int, int> lay(0);
  CHECK(lay.num_tiles() == 0);
  CHECK(lay.num_atoms() == 0);
  check_layout_invariants(lay, 0);
}

TEST_CASE("layout::coo with one nonzero", "[layout][coo][edge]") {
  layout::coo<int, int> lay(1);
  CHECK(lay.num_tiles() == 1);
  CHECK(lay.num_atoms() == 1);
  CHECK(lay.tile_size(0) == 1);
  CHECK(lay.tile_of(0) == 0);
  check_layout_invariants(lay, 1);
  check_tile_of_round_trip(lay);
}
