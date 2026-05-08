/**
 * @file test_layout_csr.cu
 * @author Loops contributors
 * @brief Layout-contract conformance for @c loops::layout::csr .
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <catch2/catch_test_macros.hpp>

#include <loops/container/layout.hxx>

#include "test_layout_contract.hxx"

#include <thrust/host_vector.h>

#include <vector>

using namespace loops;
using namespace loops::testing;

TEST_CASE("layout::csr satisfies the layout contract", "[layout][csr]") {
  // Deliberately uneven row sizes including an empty row, to exercise
  // tile_of's binary search in the empty-row-skip case.
  std::vector<int> offsets_storage{0, 2, 2, 5, 7};
  thrust::host_vector<int> h_offsets(offsets_storage.begin(),
                                     offsets_storage.end());

  layout::csr<int, int> lay(h_offsets.data(),
                            /*num_tiles=*/4, /*num_atoms=*/7);

  check_layout_invariants(lay, /*expected_atoms=*/7);
  check_tile_of_round_trip(lay);

  SECTION("tile_of skips empty rows correctly") {
    // atoms 0..1 -> row 0, 2..4 -> row 2 (row 1 is empty), 5..6 -> row 3
    CHECK(lay.tile_of(0) == 0);
    CHECK(lay.tile_of(1) == 0);
    CHECK(lay.tile_of(2) == 2);
    CHECK(lay.tile_of(4) == 2);
    CHECK(lay.tile_of(5) == 3);
    CHECK(lay.tile_of(6) == 3);
  }

  SECTION("tile_size on the empty row is zero") {
    CHECK(lay.tile_size(0) == 2);
    CHECK(lay.tile_size(1) == 0);
    CHECK(lay.tile_size(2) == 3);
    CHECK(lay.tile_size(3) == 2);
  }
}

TEST_CASE("layout::csr handles a single-row matrix", "[layout][csr][edge]") {
  std::vector<int> offsets_storage{0, 5};
  thrust::host_vector<int> h_offsets(offsets_storage.begin(),
                                     offsets_storage.end());
  layout::csr<int, int> lay(h_offsets.data(), 1, 5);

  check_layout_invariants(lay, 5);
  check_tile_of_round_trip(lay);
  CHECK(lay.num_tiles() == 1);
  CHECK(lay.tile_size(0) == 5);
  for (int a = 0; a < 5; ++a) {
    CHECK(lay.tile_of(a) == 0);
  }
}

TEST_CASE("layout::csr handles all-empty rows", "[layout][csr][edge]") {
  // 4 rows, every row empty.
  std::vector<int> offsets_storage{0, 0, 0, 0, 0};
  thrust::host_vector<int> h_offsets(offsets_storage.begin(),
                                     offsets_storage.end());
  layout::csr<int, int> lay(h_offsets.data(), 4, 0);

  check_layout_invariants(lay, 0);
  CHECK(lay.num_atoms() == 0);
  for (int t = 0; t < 4; ++t)
    CHECK(lay.tile_size(t) == 0);
}
