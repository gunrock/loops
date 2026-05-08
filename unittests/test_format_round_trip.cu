/**
 * @file test_format_round_trip.cu
 * @author Loops contributors
 * @brief Cross-format equivalence: every container's csr-conversion ctor
 * preserves the dense-matrix view of the input CSR.
 *
 * For each format @c F we materialize the equivalent dense matrix from
 * the CSR @e and from the converted format, then compare element-wise.
 * Any divergence means the converter dropped, duplicated, or mis-located
 * a non-zero. The dense intermediate is the simplest possible oracle
 * (and trivially correct for the small matrices we use here).
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <catch2/catch_test_macros.hpp>

#include <loops/container/bcsr.hxx>
#include <loops/container/coo.hxx>
#include <loops/container/csc.hxx>
#include <loops/container/csr.hxx>
#include <loops/container/dia.hxx>
#include <loops/container/ell.hxx>
#include <loops/memory.hxx>

#include "test_helpers.hxx"

#include <cstddef>
#include <vector>

using namespace loops;
using namespace loops::testing;

namespace {

using dense_t = std::vector<std::vector<float>>;

dense_t dense_from_csr(const csr_host_t& csr) {
  dense_t m(csr.rows, std::vector<float>(csr.cols, 0.0f));
  for (std::size_t r = 0; r < csr.rows; ++r) {
    for (auto k = csr.offsets[r]; k < csr.offsets[r + 1]; ++k) {
      m[r][csr.indices[k]] = csr.values[k];
    }
  }
  return m;
}

template <typename coo_t>
dense_t dense_from_coo(const coo_t& coo) {
  dense_t m(coo.rows, std::vector<float>(coo.cols, 0.0f));
  for (std::size_t a = 0; a < coo.nnzs; ++a) {
    m[coo.row_indices[a]][coo.col_indices[a]] = coo.values[a];
  }
  return m;
}

template <typename csc_t>
dense_t dense_from_csc(const csc_t& csc) {
  dense_t m(csc.rows, std::vector<float>(csc.cols, 0.0f));
  for (std::size_t c = 0; c < csc.cols; ++c) {
    for (auto a = csc.offsets[c]; a < csc.offsets[c + 1]; ++a) {
      m[csc.indices[a]][c] = csc.values[a];
    }
  }
  return m;
}

template <typename ell_t>
dense_t dense_from_ell(const ell_t& ell) {
  dense_t m(ell.rows, std::vector<float>(ell.cols, 0.0f));
  for (std::size_t r = 0; r < ell.rows; ++r) {
    for (std::size_t k = 0; k < ell.pitch; ++k) {
      const std::size_t slot = r * ell.pitch + k;
      const int col = ell.indices[slot];
      if (col >= 0) {
        m[r][col] = ell.values[slot];
      }
    }
  }
  return m;
}

template <std::size_t R, std::size_t C, typename bcsr_t>
dense_t dense_from_bcsr(const bcsr_t& b) {
  dense_t m(b.rows, std::vector<float>(b.cols, 0.0f));
  for (std::size_t br = 0; br < b.num_block_rows; ++br) {
    for (auto a = b.block_offsets[br]; a < b.block_offsets[br + 1]; ++a) {
      const std::size_t bc = b.block_col_indices[a];
      for (std::size_t i = 0; i < R; ++i) {
        for (std::size_t j = 0; j < C; ++j) {
          const std::size_t r = br * R + i;
          const std::size_t c = bc * C + j;
          if (r < b.rows && c < b.cols) {
            m[r][c] = b.values[a * R * C + i * C + j];
          }
        }
      }
    }
  }
  return m;
}

template <typename dia_t>
dense_t dense_from_dia(const dia_t& d) {
  dense_t m(d.rows, std::vector<float>(d.cols, 0.0f));
  for (std::size_t i = 0; i < d.num_diagonals; ++i) {
    const int off = d.diag_offsets[i];
    for (std::size_t r = 0; r < d.rows; ++r) {
      const long c = static_cast<long>(r) + off;
      if (c >= 0 && c < static_cast<long>(d.cols)) {
        m[r][c] = d.values[i * d.stride + r];
      }
    }
  }
  return m;
}

void check_dense_equal(const dense_t& a, const dense_t& b) {
  REQUIRE(a.size() == b.size());
  for (std::size_t r = 0; r < a.size(); ++r) {
    REQUIRE(a[r].size() == b[r].size());
    for (std::size_t c = 0; c < a[r].size(); ++c) {
      CHECK(a[r][c] == b[r][c]);
    }
  }
}

}  // namespace

TEST_CASE("csr <-> coo dense-equivalence", "[round_trip][csr][coo]") {
  auto csr = make_banded_csr(8, 2, 1);
  coo_t<int, float, memory::memory_space_t::host> coo(csr);
  check_dense_equal(dense_from_csr(csr), dense_from_coo(coo));
}

TEST_CASE("csr -> csc dense-equivalence", "[round_trip][csr][csc]") {
  auto csr = make_banded_csr(8, 2, 1);
  csc_t<int, int, float, memory::memory_space_t::host> csc(csr);
  check_dense_equal(dense_from_csr(csr), dense_from_csc(csc));
}

TEST_CASE("csr -> ell dense-equivalence", "[round_trip][csr][ell]") {
  auto csr = make_banded_csr(8, 2, 1);
  ell_t<int, float, memory::memory_space_t::host> ell(csr);
  check_dense_equal(dense_from_csr(csr), dense_from_ell(ell));
}

TEST_CASE("csr -> bcsr (2x2) dense-equivalence", "[round_trip][csr][bcsr]") {
  auto csr = make_block_diag_csr(/*num_blocks=*/4, /*block_size=*/2);
  bcsr_t<2, 2, int, int, float, memory::memory_space_t::host> b(csr);
  check_dense_equal(dense_from_csr(csr), dense_from_bcsr<2, 2>(b));
}

TEST_CASE("csr -> bcsr (3x3) on non-divisible dims dense-equivalence",
          "[round_trip][csr][bcsr][edge]") {
  auto csr = make_banded_csr(7, 1, 1);
  bcsr_t<3, 3, int, int, float, memory::memory_space_t::host> b(csr);
  check_dense_equal(dense_from_csr(csr), dense_from_bcsr<3, 3>(b));
}

TEST_CASE("csr -> dia dense-equivalence", "[round_trip][csr][dia]") {
  auto csr = make_banded_csr(8, 1, 2);
  dia_t<int, int, float, memory::memory_space_t::host> d(csr);
  check_dense_equal(dense_from_csr(csr), dense_from_dia(d));
}

TEST_CASE("csr -> coo -> csc dense-equivalence (transpose-of-transpose)",
          "[round_trip][csr][coo][csc]") {
  auto csr = make_banded_csr(8, 2, 1);
  coo_t<int, float, memory::memory_space_t::host> coo(csr);
  csc_t<int, int, float, memory::memory_space_t::host> csc(coo);
  check_dense_equal(dense_from_csr(csr), dense_from_csc(csc));
}
