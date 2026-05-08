/**
 * @file test_helpers.hxx
 * @author Loops contributors
 * @brief Shared test fixtures: deterministic synthetic-matrix factories and
 * a CSR-equivalence predicate. Header-only so multiple test TUs include it
 * without link-time collisions.
 *
 * Design goal: tests should be *hermetic*. No file I/O, no dataset paths,
 * no environment-dependent state. Every matrix used in a test is built on
 * demand from a known recipe so failures are reproducible. The factories
 * cover the patterns the in-tree formats are designed to exploit:
 *
 *   - identity and diagonal      -> one-tile-per-row sanity checks
 *   - banded                     -> DIA / ELL sweet spot
 *   - block-diagonal             -> BCSR sweet spot
 *   - skewed (power-law-ish)     -> load-balancing stress
 *   - empty-rows                 -> binary-search edge case
 *   - random                     -> general-purpose smoke
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <catch2/catch_test_macros.hpp>

#include <loops/container/csr.hxx>
#include <loops/container/coo.hxx>
#include <loops/container/vector.hxx>
#include <loops/memory.hxx>

#include <cuda_runtime.h>

#include <thrust/host_vector.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <tuple>
#include <vector>

namespace loops {
namespace testing {

using csr_host_t = loops::csr_t<int, int, float, loops::memory::memory_space_t::host>;
using x_host_t = loops::vector_t<float, loops::memory::memory_space_t::host>;

/// Convert a (row_indices, col_indices, values) coordinate triple into a
/// canonical row-sorted CSR. Used by every factory below; centralized so
/// the same builder logic is exercised by every test path.
inline csr_host_t coords_to_csr(int rows,
                                int cols,
                                std::vector<int> row_idx,
                                std::vector<int> col_idx,
                                std::vector<float> values) {
  REQUIRE(row_idx.size() == col_idx.size());
  REQUIRE(row_idx.size() == values.size());

  const std::size_t nnz = values.size();

  // Sort by (row, col) to keep within-row order canonical.
  std::vector<std::size_t> perm(nnz);
  for (std::size_t i = 0; i < nnz; ++i) perm[i] = i;
  std::sort(perm.begin(), perm.end(), [&](std::size_t a, std::size_t b) {
    return std::tie(row_idx[a], col_idx[a]) <
           std::tie(row_idx[b], col_idx[b]);
  });

  csr_host_t csr(static_cast<std::size_t>(rows),
                 static_cast<std::size_t>(cols), nnz);

  for (std::size_t i = 0; i <= static_cast<std::size_t>(rows); ++i)
    csr.offsets[i] = 0;
  for (std::size_t i = 0; i < nnz; ++i)
    csr.offsets[row_idx[perm[i]] + 1]++;
  for (std::size_t i = 1; i <= static_cast<std::size_t>(rows); ++i)
    csr.offsets[i] += csr.offsets[i - 1];

  for (std::size_t i = 0; i < nnz; ++i) {
    csr.indices[i] = col_idx[perm[i]];
    csr.values[i] = values[perm[i]];
  }
  return csr;
}

/// I_n : the n-by-n identity. SpMV(I, x) == x for any x. Cheapest sanity
/// check that doesn't depend on summation order.
inline csr_host_t make_identity_csr(int n) {
  std::vector<int> ri(n), ci(n);
  std::vector<float> vs(n, 1.0f);
  for (int i = 0; i < n; ++i) {
    ri[i] = i;
    ci[i] = i;
  }
  return coords_to_csr(n, n, std::move(ri), std::move(ci), std::move(vs));
}

/// Tridiagonal-ish band: every row r holds nonzeros at columns
/// [r-lower, r+upper] intersected with [0, n).
inline csr_host_t make_banded_csr(int n, int lower, int upper) {
  std::vector<int> ri, ci;
  std::vector<float> vs;
  for (int r = 0; r < n; ++r) {
    int lo = std::max(0, r - lower);
    int hi = std::min(n - 1, r + upper);
    for (int c = lo; c <= hi; ++c) {
      ri.push_back(r);
      ci.push_back(c);
      // Distinct, deterministic, non-zero values keep summation
      // order-sensitivity from spuriously masking real bugs.
      vs.push_back(static_cast<float>(r * 1000 + c) * 0.001f + 0.5f);
    }
  }
  return coords_to_csr(n, n, std::move(ri), std::move(ci), std::move(vs));
}

/// Block-diagonal: @c num_blocks dense @c block_size x block_size blocks
/// stacked along the diagonal. Total dimensions: (num_blocks * block_size).
/// BCSR with R = C = block_size compresses this losslessly.
inline csr_host_t make_block_diag_csr(int num_blocks, int block_size) {
  const int n = num_blocks * block_size;
  std::vector<int> ri, ci;
  std::vector<float> vs;
  for (int b = 0; b < num_blocks; ++b) {
    int base = b * block_size;
    for (int i = 0; i < block_size; ++i) {
      for (int j = 0; j < block_size; ++j) {
        ri.push_back(base + i);
        ci.push_back(base + j);
        vs.push_back(0.5f + static_cast<float>(b) +
                     static_cast<float>(i * block_size + j) * 0.01f);
      }
    }
  }
  return coords_to_csr(n, n, std::move(ri), std::move(ci), std::move(vs));
}

/// Power-law-ish skewed degree distribution: row 0 has @c heavy_row_nnz
/// entries scattered across the matrix, every other row has @c light_row_nnz .
/// Stresses static row-mapped schedules vs load-balanced ones.
inline csr_host_t make_skewed_csr(int rows,
                                  int cols,
                                  int heavy_row_nnz,
                                  int light_row_nnz,
                                  std::uint64_t seed = 7u) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> col_dist(0, cols - 1);
  std::uniform_real_distribution<float> val_dist(0.5f, 1.5f);

  std::vector<int> ri, ci;
  std::vector<float> vs;
  auto add_row = [&](int r, int nnz) {
    std::vector<int> picks;
    picks.reserve(nnz);
    while (static_cast<int>(picks.size()) < nnz) {
      int c = col_dist(rng);
      if (std::find(picks.begin(), picks.end(), c) == picks.end())
        picks.push_back(c);
    }
    for (int c : picks) {
      ri.push_back(r);
      ci.push_back(c);
      vs.push_back(val_dist(rng));
    }
  };
  add_row(0, std::min(heavy_row_nnz, cols));
  for (int r = 1; r < rows; ++r) add_row(r, std::min(light_row_nnz, cols));
  return coords_to_csr(rows, cols, std::move(ri), std::move(ci),
                       std::move(vs));
}

/// Sprinkles entries with the requested density, then deliberately empties
/// every k-th row to force schedules through binary-search-on-empty paths.
inline csr_host_t make_empty_row_csr(int rows,
                                     int cols,
                                     float density,
                                     int empty_every,
                                     std::uint64_t seed = 11u) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dens_dist(0.0f, 1.0f);
  std::uniform_real_distribution<float> val_dist(0.5f, 1.5f);

  std::vector<int> ri, ci;
  std::vector<float> vs;
  for (int r = 0; r < rows; ++r) {
    if (empty_every > 0 && (r % empty_every) == 0) continue;
    for (int c = 0; c < cols; ++c) {
      if (dens_dist(rng) < density) {
        ri.push_back(r);
        ci.push_back(c);
        vs.push_back(val_dist(rng));
      }
    }
  }
  return coords_to_csr(rows, cols, std::move(ri), std::move(ci),
                       std::move(vs));
}

/// Uniformly random sparse matrix with deterministic seed. Removes
/// duplicate (row, col) entries by keeping the first occurrence.
inline csr_host_t make_random_csr(int rows,
                                  int cols,
                                  float density,
                                  std::uint64_t seed = 17u) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dens_dist(0.0f, 1.0f);
  std::uniform_real_distribution<float> val_dist(0.5f, 1.5f);

  std::vector<int> ri, ci;
  std::vector<float> vs;
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      if (dens_dist(rng) < density) {
        ri.push_back(r);
        ci.push_back(c);
        vs.push_back(val_dist(rng));
      }
    }
  }
  return coords_to_csr(rows, cols, std::move(ri), std::move(ci),
                       std::move(vs));
}

/// Deterministic input vector sized to match @c csr.cols .
inline x_host_t make_input_vector(const csr_host_t& csr,
                                  std::uint64_t seed = 23u) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(0.5f, 1.5f);
  x_host_t x(csr.cols);
  for (std::size_t i = 0; i < csr.cols; ++i) x[i] = dist(rng);
  return x;
}

/// Approximate-equality predicate for SpMV outputs. The 1e-3 absolute /
/// 1e-4 relative envelope is loose enough to absorb GPU summation-order
/// effects without flagging real arithmetic mistakes; tighten it via the
/// arguments for kernels that should be bit-exact.
inline bool nearly_equal(float a,
                         float b,
                         float rtol = 1e-4f,
                         float atol = 1e-3f) {
  return std::abs(a - b) <= atol + rtol * std::abs(b);
}

/// Shorthand for the common test pattern: pull a device output to host and
/// count how many entries disagree with the host reference.
inline std::size_t count_mismatches(const float* d_y,
                                    const std::vector<float>& y_ref,
                                    float rtol = 1e-4f,
                                    float atol = 1e-3f) {
  thrust::host_vector<float> h(y_ref.size());
  cudaMemcpy(h.data(), d_y, y_ref.size() * sizeof(float),
             cudaMemcpyDeviceToHost);
  std::size_t errs = 0;
  for (std::size_t i = 0; i < y_ref.size(); ++i) {
    if (!nearly_equal(h[i], y_ref[i], rtol, atol)) ++errs;
  }
  return errs;
}

/// Host-side reference SpMV ; lets tests build expected outputs from the
/// same factories that produce their GPU inputs.
inline std::vector<float> reference_spmv(const csr_host_t& csr,
                                         const x_host_t& x) {
  std::vector<float> y(csr.rows, 0.0f);
  for (std::size_t r = 0; r < csr.rows; ++r) {
    float s = 0.0f;
    for (auto k = csr.offsets[r]; k < csr.offsets[r + 1]; ++k)
      s += csr.values[k] * x[csr.indices[k]];
    y[r] = s;
  }
  return y;
}

}  // namespace testing
}  // namespace loops
