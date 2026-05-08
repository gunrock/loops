/**
 * @file test_container_ell.cu
 * @author Loops contributors
 * @brief Construction and conversion tests for @c loops::ell_t .
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <catch2/catch_test_macros.hpp>

#include <loops/container/ell.hxx>
#include <loops/container/csr.hxx>
#include <loops/memory.hxx>

#include "test_helpers.hxx"

using namespace loops;
using namespace loops::testing;

TEST_CASE("ell_t default constructor zeros every field", "[container][ell]") {
  ell_t<int, float> e;
  CHECK(e.rows == 0);
  CHECK(e.cols == 0);
  CHECK(e.nnzs == 0);
  CHECK(e.pitch == 0);
  CHECK(e.indices.size() == 0);
  CHECK(e.values.size() == 0);
}

TEST_CASE("ell_t dimensioned constructor pads with sentinel and zeros",
          "[container][ell]") {
  ell_t<int, float, memory::memory_space_t::host> e(/*rows=*/4, /*cols=*/8,
                                                    /*nnz=*/12, /*pitch=*/3);
  CHECK(e.rows == 4);
  CHECK(e.cols == 8);
  CHECK(e.nnzs == 12);
  CHECK(e.pitch == 3);
  CHECK(e.indices.size() == 4 * 3);
  CHECK(e.values.size() == 4 * 3);
  // All slots default to padding (sentinel index, zero value).
  for (std::size_t i = 0; i < e.indices.size(); ++i) {
    CHECK(e.indices[i] == static_cast<int>(-1));
    CHECK(e.values[i] == 0.0f);
  }
}

TEST_CASE("csr_t->ell_t conversion picks pitch = max row degree",
          "[container][ell][conversion]") {
  // A banded matrix with bandwidth (1,1) -> at most 3 entries per row;
  // the inner rows have all 3, the boundary rows have 2. Pitch should be 3.
  auto h_csr = make_banded_csr(5, 1, 1);
  ell_t<int, float, memory::memory_space_t::host> ell(h_csr);

  CHECK(ell.rows == h_csr.rows);
  CHECK(ell.cols == h_csr.cols);
  CHECK(ell.nnzs == h_csr.nnzs);
  CHECK(ell.pitch == 3);
  CHECK(ell.indices.size() == 5 * 3);

  // Each row's stored entries are exactly the CSR row's entries; padding
  // slots carry the sentinel.
  for (std::size_t r = 0; r < h_csr.rows; ++r) {
    auto deg = h_csr.offsets[r + 1] - h_csr.offsets[r];
    for (std::size_t k = 0; k < ell.pitch; ++k) {
      auto slot = r * ell.pitch + k;
      if (k < static_cast<std::size_t>(deg)) {
        CHECK(ell.indices[slot] == h_csr.indices[h_csr.offsets[r] + k]);
        CHECK(ell.values[slot] == h_csr.values[h_csr.offsets[r] + k]);
      } else {
        CHECK(ell.indices[slot] == ell_t<int, float>::sentinel());
        CHECK(ell.values[slot] == 0.0f);
      }
    }
  }
}

TEST_CASE("ell_t survives a host->device->host round trip",
          "[container][ell][space]") {
  auto h_csr = make_banded_csr(6, 1, 1);
  ell_t<int, float, memory::memory_space_t::host> ell_h(h_csr);
  ell_t<int, float, memory::memory_space_t::device> ell_d(ell_h);
  ell_t<int, float, memory::memory_space_t::host> back(ell_d);

  CHECK(back.rows == ell_h.rows);
  CHECK(back.cols == ell_h.cols);
  CHECK(back.nnzs == ell_h.nnzs);
  CHECK(back.pitch == ell_h.pitch);
  for (std::size_t i = 0; i < ell_h.indices.size(); ++i) {
    CHECK(back.indices[i] == ell_h.indices[i]);
    CHECK(back.values[i] == ell_h.values[i]);
  }
}
