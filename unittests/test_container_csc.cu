/**
 * @file test_container_csc.cu
 * @author Loops contributors
 * @brief Construction and conversion tests for @c loops::csc_t .
 *
 * Pin the offsets-size fix (rows+1 -> cols+1) and the new csr->csc
 * convenience ctor so they don't regress.
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <catch2/catch_test_macros.hpp>

#include <loops/container/csc.hxx>
#include <loops/container/csr.hxx>
#include <loops/memory.hxx>

#include "test_helpers.hxx"

#include <vector>

using namespace loops;
using namespace loops::testing;

TEST_CASE("csc_t default constructor zeros every field", "[container][csc]") {
  csc_t<int, int, float> c;
  CHECK(c.rows == 0);
  CHECK(c.cols == 0);
  CHECK(c.nnzs == 0);
  CHECK(c.offsets.size() == 0);
  CHECK(c.indices.size() == 0);
  CHECK(c.values.size() == 0);
}

TEST_CASE("csc_t dimensioned constructor sizes offsets to cols+1 (not rows+1)",
          "[container][csc][regression]") {
  // Regression for the historical bug where csc_t(rows, cols, nnz) used
  // offsets(rows + 1). Tests the fix in 06ee828.
  csc_t<int, int, float> c(/*rows=*/3, /*cols=*/7, /*nnz=*/15);
  CHECK(c.rows == 3);
  CHECK(c.cols == 7);
  CHECK(c.nnzs == 15);
  CHECK(c.offsets.size() == 8);  // cols + 1
  CHECK(c.indices.size() == 15);
  CHECK(c.values.size() == 15);
}

TEST_CASE("csr_t->csc_t round-trips: each CSR entry shows up in the CSC",
          "[container][csc][conversion]") {
  // Build a known CSR, transpose to CSC, then verify every (r,c,v) triple
  // from the CSR is found in the CSC's column-stationary layout. CSC's
  // own offsets indexes by column, indices stores the row id of each NZ.
  auto h_csr = make_banded_csr(8, 2, 1);
  csc_t<int, int, float, memory::memory_space_t::host> csc(h_csr);

  CHECK(csc.rows == h_csr.rows);
  CHECK(csc.cols == h_csr.cols);
  CHECK(csc.nnzs == h_csr.nnzs);
  CHECK(csc.offsets.size() == h_csr.cols + 1);

  // CSR-by-row scan, look each (r,c,v) up in the CSC by column.
  for (std::size_t r = 0; r < h_csr.rows; ++r) {
    for (auto k = h_csr.offsets[r]; k < h_csr.offsets[r + 1]; ++k) {
      int c = h_csr.indices[k];
      float v = h_csr.values[k];
      bool found = false;
      for (auto a = csc.offsets[c]; a < csc.offsets[c + 1]; ++a) {
        if (csc.indices[a] == static_cast<int>(r) && csc.values[a] == v) {
          found = true;
          break;
        }
      }
      CHECK(found);
    }
  }
}

TEST_CASE("csc_t survives a host->device->host round trip",
          "[container][csc][space]") {
  auto h_csr = make_banded_csr(6, 1, 1);
  csc_t<int, int, float, memory::memory_space_t::host> csc_h(h_csr);
  csc_t<int, int, float, memory::memory_space_t::device> csc_d(csc_h);
  csc_t<int, int, float, memory::memory_space_t::host> back(csc_d);

  CHECK(back.rows == csc_h.rows);
  CHECK(back.cols == csc_h.cols);
  CHECK(back.nnzs == csc_h.nnzs);
  for (std::size_t i = 0; i < csc_h.offsets.size(); ++i)
    CHECK(back.offsets[i] == csc_h.offsets[i]);
  for (std::size_t i = 0; i < csc_h.nnzs; ++i) {
    CHECK(back.indices[i] == csc_h.indices[i]);
    CHECK(back.values[i] == csc_h.values[i]);
  }
}
