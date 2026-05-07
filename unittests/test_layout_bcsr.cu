/**
 * @file test_layout_bcsr.cu
 * @author Loops contributors
 * @brief Layout-contract conformance for @c loops::layout::bcsr .
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

TEST_CASE("layout::bcsr satisfies the layout contract", "[layout][bcsr]") {
  // 5 block-rows, 12 stored blocks total, the third block-row deliberately
  // empty so tile_of has to skip an empty entry.
  std::vector<int> offsets_storage{0, 3, 5, 5, 9, 12};
  thrust::host_vector<int> h_offsets(offsets_storage.begin(),
                                     offsets_storage.end());
  layout::bcsr<int, int> lay(h_offsets.data(), /*num_tiles=*/5,
                             /*num_atoms=*/12);

  check_layout_invariants(lay, /*expected_atoms=*/12);
  check_tile_of_round_trip(lay);

  SECTION("structurally identical to csr - same offsets math") {
    CHECK(lay.tile_size(0) == 3);
    CHECK(lay.tile_size(1) == 2);
    CHECK(lay.tile_size(2) == 0);
    CHECK(lay.tile_size(3) == 4);
    CHECK(lay.tile_size(4) == 3);
    CHECK(lay.tile_of(0) == 0);
    CHECK(lay.tile_of(4) == 1);
    CHECK(lay.tile_of(5) == 3);
    CHECK(lay.tile_of(11) == 4);
  }
}

TEST_CASE("layout::bcsr with one block-row containing all blocks",
          "[layout][bcsr][edge]") {
  std::vector<int> offsets_storage{0, 6};
  thrust::host_vector<int> h_offsets(offsets_storage.begin(),
                                     offsets_storage.end());
  layout::bcsr<int, int> lay(h_offsets.data(), 1, 6);

  check_layout_invariants(lay, 6);
  check_tile_of_round_trip(lay);
  CHECK(lay.num_tiles() == 1);
  for (int a = 0; a < 6; ++a) CHECK(lay.tile_of(a) == 0);
}
