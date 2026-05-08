/**
 * @file test_layout_flat_partitioner.cu
 * @author Loops contributors
 * @brief Layout-contract conformance for the
 * @c loops::layout::flat_uniform_occupancy adaptor.
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

TEST_CASE("layout::flat_uniform_occupancy satisfies the layout contract",
          "[layout][partitioner][flat]") {
  // Wrap a CSR base (4 rows, 7 nnz with one empty row) into K=2 windows.
  std::vector<int> offsets{0, 2, 2, 5, 7};
  thrust::host_vector<int> h_offsets(offsets.begin(), offsets.end());
  layout::csr<int, int> base(h_offsets.data(), 4, 7);

  using lay_t = layout::flat_uniform_occupancy<2, layout::csr<int, int>>;
  lay_t lay(base);

  CHECK(lay.num_tiles() == 4);  // ceil(7 / 2)
  CHECK(lay.num_atoms() == 7);

  check_layout_invariants(lay, /*expected_atoms=*/7);
  check_tile_of_round_trip(lay);

  SECTION("tile sizes are K except possibly the last one") {
    CHECK(lay.tile_size(0) == 2);
    CHECK(lay.tile_size(1) == 2);
    CHECK(lay.tile_size(2) == 2);
    CHECK(lay.tile_size(3) == 1);  // 7 mod 2 == 1
  }

  SECTION("base is preserved and queryable") {
    CHECK(lay.base().num_tiles() == 4);
    CHECK(lay.base().num_atoms() == 7);
    // Atom 2 of the partitioned layout (= flat atom 2 of the base CSR) sits
    // in CSR row 2 since row 1 is empty and rows 0 covers atoms 0..1.
    CHECK(lay.base().tile_of(2) == 2);
  }

  SECTION("partitioned tile_of is a / K") {
    for (int a = 0; a < 7; ++a) {
      CHECK(lay.tile_of(a) == a / 2);
    }
  }
}

TEST_CASE("layout::flat_uniform_occupancy with K dividing num_atoms exactly",
          "[layout][partitioner][flat][edge]") {
  // 4 rows, 8 nnz, K = 4 -> exactly 2 tiles, none short.
  std::vector<int> offsets{0, 2, 4, 6, 8};
  thrust::host_vector<int> h_offsets(offsets.begin(), offsets.end());
  layout::csr<int, int> base(h_offsets.data(), 4, 8);

  using lay_t = layout::flat_uniform_occupancy<4, layout::csr<int, int>>;
  lay_t lay(base);

  check_layout_invariants(lay, 8);
  check_tile_of_round_trip(lay);
  CHECK(lay.num_tiles() == 2);
  CHECK(lay.tile_size(0) == 4);
  CHECK(lay.tile_size(1) == 4);
}

TEST_CASE("layout::flat_uniform_occupancy with K > num_atoms",
          "[layout][partitioner][flat][edge]") {
  // 2 rows, 3 nnz, K = 16 -> a single short tile of size 3.
  std::vector<int> offsets{0, 1, 3};
  thrust::host_vector<int> h_offsets(offsets.begin(), offsets.end());
  layout::csr<int, int> base(h_offsets.data(), 2, 3);

  using lay_t = layout::flat_uniform_occupancy<16, layout::csr<int, int>>;
  lay_t lay(base);

  check_layout_invariants(lay, 3);
  check_tile_of_round_trip(lay);
  CHECK(lay.num_tiles() == 1);
  CHECK(lay.tile_size(0) == 3);
}

TEST_CASE("layout::flat_uniform_occupancy can wrap layout::ell",
          "[layout][partitioner][flat][edge]") {
  // ELL with 4 rows x pitch=3 -> 12 atoms; K=3 -> 4 tiles of 3.
  layout::ell<int, int> base(/*num_tiles=*/4, /*pitch=*/3);

  using lay_t = layout::flat_uniform_occupancy<3, layout::ell<int, int>>;
  lay_t lay(base);

  check_layout_invariants(lay, 12);
  check_tile_of_round_trip(lay);
  CHECK(lay.num_tiles() == 4);
  for (int t = 0; t < 4; ++t)
    CHECK(lay.tile_size(t) == 3);
  // Recovering the original ELL row of an atom still works through base().
  CHECK(lay.base().tile_of(0) == 0);
  CHECK(lay.base().tile_of(5) == 1);
  CHECK(lay.base().tile_of(11) == 3);
}
