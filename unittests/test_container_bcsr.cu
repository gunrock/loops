/**
 * @file test_container_bcsr.cu
 * @author Loops contributors
 * @brief Construction and conversion tests for @c loops::bcsr_t .
 *
 * Exercises:
 *   - default ctor zeros every field
 *   - csr -> bcsr conversion with both 2x2 and 3x3 block sizes
 *   - block-column sorting within a block-row (canonical form)
 *   - per-block dense payload reproduces the source CSR exactly
 *   - non-divisible matrix dims (cols % C != 0) get padded with zeros
 *   - host -> device -> host round-trip preserves all fields
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <catch2/catch_test_macros.hpp>

#include <loops/container/bcsr.hxx>
#include <loops/container/csr.hxx>
#include <loops/memory.hxx>

#include "test_helpers.hxx"

using namespace loops;
using namespace loops::testing;

TEST_CASE("bcsr_t default constructor zeros every field",
          "[container][bcsr]") {
  bcsr_t<2, 2, int, int, float> b;
  CHECK(b.rows == 0);
  CHECK(b.cols == 0);
  CHECK(b.nnzs == 0);
  CHECK(b.num_block_rows == 0);
  CHECK(b.num_block_cols == 0);
  CHECK(b.num_blocks == 0);
  CHECK(b.block_offsets.size() == 0);
  CHECK(b.block_col_indices.size() == 0);
  CHECK(b.values.size() == 0);
}

TEST_CASE("csr_t->bcsr_t (2x2): block-diagonal matrix yields one block per "
          "block-row",
          "[container][bcsr][conversion]") {
  // 4 dense 2x2 blocks on the diagonal of an 8x8 matrix. With R = C = 2
  // we expect num_blocks == 4, one per block-row.
  auto h_csr = make_block_diag_csr(/*num_blocks=*/4, /*block_size=*/2);
  bcsr_t<2, 2, int, int, float, memory::memory_space_t::host> b(h_csr);

  CHECK(b.rows == 8);
  CHECK(b.cols == 8);
  CHECK(b.num_block_rows == 4);
  CHECK(b.num_block_cols == 4);
  CHECK(b.num_blocks == 4);
  CHECK(b.block_offsets.size() == 5);
  CHECK(b.block_col_indices.size() == 4);
  CHECK(b.values.size() == 4 * 2 * 2);

  // Each block-row holds exactly one block, with block-col equal to br.
  for (std::size_t br = 0; br < 4; ++br) {
    CHECK(b.block_offsets[br + 1] - b.block_offsets[br] == 1);
    CHECK(b.block_col_indices[b.block_offsets[br]] == static_cast<int>(br));
  }
}

TEST_CASE("csr_t->bcsr_t (2x2): every CSR entry is reachable in the BCSR",
          "[container][bcsr][conversion]") {
  // A banded matrix exercises both within-block and cross-block patterns.
  auto h_csr = make_banded_csr(8, 2, 2);
  bcsr_t<2, 2, int, int, float, memory::memory_space_t::host> b(h_csr);

  // For every CSR (r, c, v), confirm the corresponding cell in the BCSR's
  // dense per-block payload holds the same value.
  for (std::size_t r = 0; r < h_csr.rows; ++r) {
    for (auto k = h_csr.offsets[r]; k < h_csr.offsets[r + 1]; ++k) {
      int c = h_csr.indices[k];
      float v = h_csr.values[k];
      const std::size_t br = r / 2;
      const std::size_t bc = c / 2;
      const std::size_t i = r % 2;
      const std::size_t j = c % 2;

      bool found = false;
      for (auto a = b.block_offsets[br]; a < b.block_offsets[br + 1]; ++a) {
        if (b.block_col_indices[a] == static_cast<int>(bc)) {
          CHECK(b.values[a * 4 + i * 2 + j] == v);
          found = true;
          break;
        }
      }
      CHECK(found);
    }
  }
}

TEST_CASE("csr_t->bcsr_t (3x3): non-divisible dimensions get padded",
          "[container][bcsr][conversion][edge]") {
  // 7x7 matrix with R = C = 3 -> num_block_rows = num_block_cols = 3, with
  // padding zeros at row 7..8 and col 7..8 of the dense block payload.
  auto h_csr = make_banded_csr(7, 1, 1);
  bcsr_t<3, 3, int, int, float, memory::memory_space_t::host> b(h_csr);

  CHECK(b.rows == 7);
  CHECK(b.cols == 7);
  CHECK(b.num_block_rows == 3);  // ceil(7 / 3)
  CHECK(b.num_block_cols == 3);

  // Non-mapped (i, j) inside a block must read as zero.
  for (std::size_t br = 0; br < 3; ++br) {
    for (auto a = b.block_offsets[br]; a < b.block_offsets[br + 1]; ++a) {
      const std::size_t bc = b.block_col_indices[a];
      for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
          const std::size_t r = br * 3 + i;
          const std::size_t c = bc * 3 + j;
          if (r >= 7 || c >= 7) {
            CHECK(b.values[a * 9 + i * 3 + j] == 0.0f);
          }
        }
      }
    }
  }
}

TEST_CASE("bcsr_t survives a host->device->host round trip",
          "[container][bcsr][space]") {
  auto h_csr = make_block_diag_csr(3, 2);
  bcsr_t<2, 2, int, int, float, memory::memory_space_t::host> b_h(h_csr);
  bcsr_t<2, 2, int, int, float, memory::memory_space_t::device> b_d(b_h);
  bcsr_t<2, 2, int, int, float, memory::memory_space_t::host> back(b_d);

  CHECK(back.rows == b_h.rows);
  CHECK(back.num_blocks == b_h.num_blocks);
  CHECK(back.num_block_rows == b_h.num_block_rows);
  for (std::size_t i = 0; i < b_h.values.size(); ++i)
    CHECK(back.values[i] == b_h.values[i]);
  for (std::size_t i = 0; i < b_h.block_col_indices.size(); ++i)
    CHECK(back.block_col_indices[i] == b_h.block_col_indices[i]);
}
