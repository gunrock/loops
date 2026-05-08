/**
 * @file test_container_coo.cu
 * @author Loops contributors
 * @brief Construction, conversion, and mutation tests for @c loops::coo_t .
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <catch2/catch_test_macros.hpp>

#include <loops/container/coo.hxx>
#include <loops/container/csr.hxx>
#include <loops/memory.hxx>

#include "test_helpers.hxx"

#include <algorithm>
#include <set>

using namespace loops;
using namespace loops::testing;

TEST_CASE("coo_t default constructor zeros every field", "[container][coo]") {
  coo_t<int, float> c;
  CHECK(c.rows == 0);
  CHECK(c.cols == 0);
  CHECK(c.nnzs == 0);
  CHECK(c.row_indices.size() == 0);
  CHECK(c.col_indices.size() == 0);
  CHECK(c.values.size() == 0);
}

TEST_CASE("coo_t dimensioned constructor sizes the storage vectors",
          "[container][coo]") {
  coo_t<int, float> c(8, 4, 17);
  CHECK(c.rows == 8);
  CHECK(c.cols == 4);
  CHECK(c.nnzs == 17);
  CHECK(c.row_indices.size() == 17);
  CHECK(c.col_indices.size() == 17);
  CHECK(c.values.size() == 17);
}

TEST_CASE("csr_t->coo_t conversion preserves every entry",
          "[container][coo][conversion]") {
  auto h_csr = make_banded_csr(6, 1, 2);
  coo_t<int, float, memory::memory_space_t::host> coo(h_csr);

  CHECK(coo.rows == h_csr.rows);
  CHECK(coo.cols == h_csr.cols);
  CHECK(coo.nnzs == h_csr.nnzs);

  // Every CSR (r, indices[k], values[k]) triple appears once in the COO.
  std::set<std::tuple<int, int, float>> coo_set;
  for (std::size_t a = 0; a < coo.nnzs; ++a) {
    coo_set.emplace(coo.row_indices[a], coo.col_indices[a], coo.values[a]);
  }
  for (std::size_t r = 0; r < h_csr.rows; ++r) {
    for (auto k = h_csr.offsets[r]; k < h_csr.offsets[r + 1]; ++k) {
      CHECK(coo_set.count(std::make_tuple(static_cast<int>(r), h_csr.indices[k],
                                          h_csr.values[k])) == 1);
    }
  }
}

TEST_CASE("coo_t::sort_by_row leaves rows monotonic",
          "[container][coo][mutation]") {
  // Start from a CSR (already row-monotonic), shuffle into a COO via the
  // CSR ctor, then explicitly sort_by_row and assert monotonicity.
  auto h_csr = make_banded_csr(8, 2, 2);
  coo_t<int, float, memory::memory_space_t::host> coo(h_csr);

  coo.sort_by_row();
  for (std::size_t a = 1; a < coo.nnzs; ++a) {
    CHECK(coo.row_indices[a] >= coo.row_indices[a - 1]);
  }
}

TEST_CASE("coo_t::sort_by_column leaves columns monotonic",
          "[container][coo][mutation]") {
  auto h_csr = make_banded_csr(8, 2, 2);
  coo_t<int, float, memory::memory_space_t::host> coo(h_csr);

  coo.sort_by_column();
  for (std::size_t a = 1; a < coo.nnzs; ++a) {
    CHECK(coo.col_indices[a] >= coo.col_indices[a - 1]);
  }
}
