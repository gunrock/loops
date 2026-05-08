/**
 * @file test_container_csr.cu
 * @author Loops contributors
 * @brief Construction, copy, and conversion tests for @c loops::csr_t .
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <catch2/catch_test_macros.hpp>

#include <loops/container/csr.hxx>
#include <loops/container/coo.hxx>
#include <loops/memory.hxx>

#include "test_helpers.hxx"

#include <vector>

using namespace loops;
using namespace loops::testing;

TEST_CASE("csr_t default constructor zeros every field", "[container][csr]") {
  csr_t<int, int, float> c;
  CHECK(c.rows == 0);
  CHECK(c.cols == 0);
  CHECK(c.nnzs == 0);
  CHECK(c.offsets.size() == 0);
  CHECK(c.indices.size() == 0);
  CHECK(c.values.size() == 0);
}

TEST_CASE("csr_t dimensioned constructor sizes the storage vectors",
          "[container][csr]") {
  csr_t<int, int, float> c(8, 4, 17);
  CHECK(c.rows == 8);
  CHECK(c.cols == 4);
  CHECK(c.nnzs == 17);
  CHECK(c.offsets.size() == 9);  // rows + 1
  CHECK(c.indices.size() == 17);
  CHECK(c.values.size() == 17);
}

TEST_CASE("csr_t survives a host->device->host round trip",
          "[container][csr][space]") {
  // Build a host CSR with a known shape, push to device, pull back, check
  // equality element-wise.
  auto h = make_banded_csr(10, 1, 1);
  csr_t<int, int, float, memory::memory_space_t::device> d(h);
  csr_t<int, int, float, memory::memory_space_t::host> back(d);

  CHECK(back.rows == h.rows);
  CHECK(back.cols == h.cols);
  CHECK(back.nnzs == h.nnzs);
  for (std::size_t i = 0; i < h.offsets.size(); ++i)
    CHECK(back.offsets[i] == h.offsets[i]);
  for (std::size_t i = 0; i < h.nnzs; ++i) {
    CHECK(back.indices[i] == h.indices[i]);
    CHECK(back.values[i] == h.values[i]);
  }
}

TEST_CASE("coo_t->csr_t conversion preserves entries (regression for "
          "col_indices member-name fix)",
          "[container][csr][conversion]") {
  // The COO-from-CSR ctor referenced the wrong member name (csr.col_indices
  // instead of csr.indices) until daa9807 ; this test pins the conversion
  // so the bug can't sneak back in.
  auto h_csr = make_banded_csr(6, 1, 2);

  // csr -> coo
  coo_t<int, float, memory::memory_space_t::host> coo(h_csr);
  CHECK(coo.rows == h_csr.rows);
  CHECK(coo.cols == h_csr.cols);
  CHECK(coo.nnzs == h_csr.nnzs);

  // Every COO triple must round-trip to the same (col, value) at the right
  // CSR row. Each CSR slot may be claimed at most once so a duplicate-emit
  // bug would surface as an unmatched residual after the loop.
  std::vector<int> csr_claimed(h_csr.nnzs, 0);
  for (std::size_t a = 0; a < coo.nnzs; ++a) {
    int r = coo.row_indices[a];
    int c = coo.col_indices[a];
    float v = coo.values[a];
    bool found = false;
    for (auto k = h_csr.offsets[r]; k < h_csr.offsets[r + 1]; ++k) {
      if (csr_claimed[k] == 0 && h_csr.indices[k] == c &&
          h_csr.values[k] == v) {
        csr_claimed[k] = 1;
        found = true;
        break;
      }
    }
    CHECK(found);
  }
  for (std::size_t k = 0; k < h_csr.nnzs; ++k) CHECK(csr_claimed[k] == 1);
}
