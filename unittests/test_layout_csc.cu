/**
 * @file test_layout_csc.cu
 * @author Loops contributors
 * @brief Layout-contract conformance for @c loops::layout::csc .
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

TEST_CASE("layout::csc satisfies the layout contract", "[layout][csc]") {
  // 4 columns, nnz=7, last column empty -> exercises tile_of's empty-tail
  // case (mirror of the csr empty-row test).
  std::vector<int> offsets_storage{0, 2, 5, 7, 7};
  thrust::host_vector<int> h_offsets(offsets_storage.begin(),
                                     offsets_storage.end());
  layout::csc<int, int> lay(h_offsets.data(), /*num_tiles=*/4, /*nnz=*/7);

  check_layout_invariants(lay, /*expected_atoms=*/7);
  check_tile_of_round_trip(lay);

  SECTION("structurally identical to csr - same offsets math") {
    CHECK(lay.tile_size(0) == 2);
    CHECK(lay.tile_size(1) == 3);
    CHECK(lay.tile_size(2) == 2);
    CHECK(lay.tile_size(3) == 0);
    CHECK(lay.tile_of(0) == 0);
    CHECK(lay.tile_of(2) == 1);
    CHECK(lay.tile_of(6) == 2);
  }
}

TEST_CASE("layout::csc handles a single-column matrix",
          "[layout][csc][edge]") {
  std::vector<int> offsets_storage{0, 5};
  thrust::host_vector<int> h_offsets(offsets_storage.begin(),
                                     offsets_storage.end());
  layout::csc<int, int> lay(h_offsets.data(), 1, 5);

  check_layout_invariants(lay, 5);
  check_tile_of_round_trip(lay);
  CHECK(lay.num_tiles() == 1);
  CHECK(lay.tile_size(0) == 5);
}
