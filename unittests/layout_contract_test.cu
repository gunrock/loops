/**
 * @file layout_contract_test.cu
 * @author Loops contributors
 * @brief Layout-contract conformance tests for in-tree layouts.
 *
 * These tests validate the cross-layout invariants documented in
 * include/loops/container/layout.hxx:
 *
 *   - tile_begin(0) == 0
 *   - tile_end(num_tiles - 1) == num_atoms
 *   - tile_end(t) is monotonically non-decreasing
 *   - tile_end_iter()[k] == tile_end(k)
 *   - tile_size(t) == tile_end(t) - tile_begin(t)
 *   - tile_of(a) returns t s.t. tile_begin(t) <= a < tile_end(t)
 *
 * Running these for every layout we add is the cheapest form of
 * insurance against future schedule-side bugs that show up only when
 * a particular layout violates an invariant.
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <loops/container/layout.hxx>
#include <loops/container/csr.hxx>

#include <thrust/host_vector.h>

#include <vector>

using namespace loops;

namespace {

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
      continue;  // empty tiles have no atoms to round-trip
    CHECK(lay.tile_of(begin) == t);
    if (end > begin) {
      CHECK(lay.tile_of(static_cast<atom_id_t>(end - 1)) == t);
    }
  }
}

}  // namespace

TEST_CASE("layout::csr satisfies the layout contract", "[layout][csr]") {
  // Build a small CSR with deliberately uneven row sizes (incl. an empty row)
  // to exercise tile_of's binary search around degenerate cases. We keep the
  // offsets array on the host because the layout view holds a raw pointer
  // that we later dereference from host code in the assertions.
  std::vector<int> offsets_storage{0, 2, 2, 5, 7};  // 4 rows, nnz=7, row 1 empty
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
}

TEST_CASE("layout::csc satisfies the layout contract", "[layout][csc]") {
  // 4 columns, nnz=7, last column empty, exercises tile_of empty-tail case.
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
    CHECK(lay.tile_size(3) == 0);  // empty tail tile
    CHECK(lay.tile_of(0) == 0);
    CHECK(lay.tile_of(2) == 1);
    CHECK(lay.tile_of(6) == 2);
  }
}

TEST_CASE("layout::coo satisfies the layout contract", "[layout][coo]") {
  // 7 nonzeros, one tile per NZ.
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

TEST_CASE("layout::ell satisfies the layout contract", "[layout][ell]") {
  // 5 rows, 3 atoms-per-row (uniform pitch). num_atoms = 15.
  layout::ell<int, int> lay(/*num_tiles=*/5, /*pitch=*/3);
  check_layout_invariants(lay, /*expected_atoms=*/15);
  check_tile_of_round_trip(lay);

  SECTION("tile_of is a / pitch") {
    CHECK(lay.tile_of(0) == 0);
    CHECK(lay.tile_of(2) == 0);
    CHECK(lay.tile_of(3) == 1);
    CHECK(lay.tile_of(14) == 4);
  }
}

TEST_CASE("layout::bcsr satisfies the layout contract", "[layout][bcsr]") {
  // 5 block-rows, 12 stored blocks, last block-row empty.
  std::vector<int> offsets_storage{0, 3, 5, 5, 9, 12};
  thrust::host_vector<int> h_offsets(offsets_storage.begin(),
                                     offsets_storage.end());
  layout::bcsr<int, int> lay(h_offsets.data(), /*num_tiles=*/5, /*nnz=*/12);

  check_layout_invariants(lay, /*expected_atoms=*/12);
  check_tile_of_round_trip(lay);

  SECTION("structurally identical to csr - same offsets math") {
    CHECK(lay.tile_size(0) == 3);
    CHECK(lay.tile_size(1) == 2);
    CHECK(lay.tile_size(2) == 0);  // empty block-row
    CHECK(lay.tile_size(3) == 4);
    CHECK(lay.tile_size(4) == 3);
    CHECK(lay.tile_of(0) == 0);
    CHECK(lay.tile_of(4) == 1);
    CHECK(lay.tile_of(5) == 3);  // skips empty block-row 2
    CHECK(lay.tile_of(11) == 4);
  }
}

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

TEST_CASE("layout::flat_uniform_occupancy satisfies the layout contract",
          "[layout][partitioner][flat]") {
  // Wrap a CSR base layout (4 rows, 7 nnz with one empty row) into a
  // flat-uniform-occupancy partitioner with K=2. We expect ceil(7/2)=4
  // tiles of 2 atoms each (last is size 1).
  std::vector<int> offsets{0, 2, 2, 5, 7};
  thrust::host_vector<int> h_offsets(offsets.begin(), offsets.end());
  layout::csr<int, int> base(h_offsets.data(), 4, 7);

  using lay_t = layout::flat_uniform_occupancy<2, layout::csr<int, int>>;
  lay_t lay(base);

  CHECK(lay.num_tiles() == 4);  // ceil(7/2)
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
    // The schedule sees 4 tiles (post-partition); the kernel can still
    // recover the underlying CSR row of any atom via base().tile_of(a).
    CHECK(lay.base().tile_of(2) == 2);  // atom 2 is in CSR row 2 (row 1 empty)
  }
}
