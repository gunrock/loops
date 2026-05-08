/**
 * @file test_container_dia.cu
 * @author Loops contributors
 * @brief Construction and conversion tests for @c loops::dia_t .
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <catch2/catch_test_macros.hpp>

#include <loops/container/dia.hxx>
#include <loops/container/csr.hxx>
#include <loops/memory.hxx>

#include "test_helpers.hxx"

#include <set>

using namespace loops;
using namespace loops::testing;

TEST_CASE("dia_t default constructor zeros every field",
          "[container][dia]") {
  dia_t<int, int, float> d;
  CHECK(d.rows == 0);
  CHECK(d.cols == 0);
  CHECK(d.nnzs == 0);
  CHECK(d.stride == 0);
  CHECK(d.num_diagonals == 0);
  CHECK(d.diag_offsets.size() == 0);
  CHECK(d.values.size() == 0);
}

TEST_CASE("csr_t->dia_t: diagonal discovery picks up exactly the populated "
          "(c - r) values",
          "[container][dia][conversion]") {
  // A 1-lower / 2-upper banded matrix populates four diagonals per row:
  // offsets {-1, 0, 1, 2} (the boundary rows touch fewer).
  auto h_csr = make_banded_csr(5, 1, 2);
  dia_t<int, int, float, memory::memory_space_t::host> dia(h_csr);

  CHECK(dia.rows == 5);
  CHECK(dia.cols == 5);
  CHECK(dia.nnzs == h_csr.nnzs);
  CHECK(dia.stride == 5);

  // Recompute the expected diagonal set on the fly.
  std::set<int> expected;
  for (std::size_t r = 0; r < h_csr.rows; ++r) {
    for (auto k = h_csr.offsets[r]; k < h_csr.offsets[r + 1]; ++k) {
      expected.insert(h_csr.indices[k] - static_cast<int>(r));
    }
  }
  CHECK(dia.num_diagonals == expected.size());
  CHECK(dia.values.size() == dia.num_diagonals * dia.stride);

  // The diag_offsets are stored in (canonical) sorted order.
  for (std::size_t i = 1; i < dia.num_diagonals; ++i) {
    CHECK(dia.diag_offsets[i] > dia.diag_offsets[i - 1]);
  }
}

TEST_CASE("csr_t->dia_t: every CSR entry lands at values[d * stride + r]",
          "[container][dia][conversion]") {
  auto h_csr = make_banded_csr(6, 1, 1);
  dia_t<int, int, float, memory::memory_space_t::host> dia(h_csr);

  // For each CSR (r, c, v), find d s.t. diag_offsets[d] == c - r and check
  // values[d * stride + r] == v.
  for (std::size_t r = 0; r < h_csr.rows; ++r) {
    for (auto k = h_csr.offsets[r]; k < h_csr.offsets[r + 1]; ++k) {
      const int c = h_csr.indices[k];
      const float v = h_csr.values[k];
      const int off = c - static_cast<int>(r);

      std::size_t d = 0;
      bool found_diag = false;
      for (; d < dia.num_diagonals; ++d) {
        if (dia.diag_offsets[d] == off) {
          found_diag = true;
          break;
        }
      }
      REQUIRE(found_diag);
      CHECK(dia.values[d * dia.stride + r] == v);
    }
  }
}

TEST_CASE("dia_t survives a host->device->host round trip",
          "[container][dia][space]") {
  auto h_csr = make_banded_csr(4, 1, 1);
  dia_t<int, int, float, memory::memory_space_t::host> dia_h(h_csr);
  dia_t<int, int, float, memory::memory_space_t::device> dia_d(dia_h);
  dia_t<int, int, float, memory::memory_space_t::host> back(dia_d);

  CHECK(back.rows == dia_h.rows);
  CHECK(back.num_diagonals == dia_h.num_diagonals);
  CHECK(back.stride == dia_h.stride);
  for (std::size_t i = 0; i < dia_h.diag_offsets.size(); ++i)
    CHECK(back.diag_offsets[i] == dia_h.diag_offsets[i]);
  for (std::size_t i = 0; i < dia_h.values.size(); ++i)
    CHECK(back.values[i] == dia_h.values[i]);
}
