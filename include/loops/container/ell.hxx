/**
 * @file ell.hxx
 * @author Loops contributors
 * @brief Interface for the ELLPACK (ELL) sparse format.
 * @version 0.1
 * @date 2026-05-05
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <loops/container/vector.hxx>
#include <loops/container/formats.hxx>
#include <loops/container/csr.hxx>
#include <loops/memory.hxx>

#include <thrust/copy.h>
#include <thrust/host_vector.h>

#include <algorithm>

namespace loops {

using namespace memory;

/**
 * @brief ELLPACK (ELL) format.
 *
 * Stores up to `pitch` non-zeros per row in two row-major dense arrays of
 * size `rows * pitch`. Rows shorter than the matrix-wide max are padded
 * with the sentinel value `index_t(-1)` in the indices array; the matching
 * value entries are zero. This wastes O((pitch - avg_nnz) * rows) of memory
 * but enables predictable per-row access patterns and (for matrices with
 * roughly uniform row degree) good GPU memory throughput.
 *
 * @tparam index_t Type of the column indices.
 * @tparam value_t Type of the non-zero values.
 * @tparam space   Memory space (host / device / managed).
 */
template <typename index_t,
          typename value_t,
          memory_space_t space = memory_space_t::device>
struct ell_t {
  std::size_t rows;
  std::size_t cols;
  std::size_t nnzs;   /// true non-zero count (excludes padding)
  std::size_t pitch;  /// atoms per row; equals max-nnz-per-row over the matrix

  vector_t<index_t, space> indices;  /// length rows * pitch, row-major
  vector_t<value_t, space> values;   /// length rows * pitch, row-major

  static __host__ __device__ index_t sentinel() {
    return static_cast<index_t>(-1);
  }

  ell_t() : rows(0), cols(0), nnzs(0), pitch(0), indices(), values() {}

  ell_t(std::size_t r, std::size_t c, std::size_t nnz, std::size_t p)
      : rows(r),
        cols(c),
        nnzs(nnz),
        pitch(p),
        indices(r * p, sentinel()),
        values(r * p, value_t(0)) {}

  /// Cross-space copy.
  template <auto rhs_space>
  ell_t(const ell_t<index_t, value_t, rhs_space>& rhs)
      : rows(rhs.rows),
        cols(rhs.cols),
        nnzs(rhs.nnzs),
        pitch(rhs.pitch),
        indices(rhs.indices),
        values(rhs.values) {}

  /**
   * @brief Build an ELL view of a CSR matrix.
   *
   * Computes `pitch = max_r (offsets[r+1] - offsets[r])` and bucket-fills
   * the dense arrays with sentinel padding. The conversion runs on the
   * host; the result is materialized in the requested memory space.
   *
   * @param csr Source CSR matrix in any memory space.
   */
  template <typename offset_t, auto rhs_space>
  ell_t(const csr_t<index_t, offset_t, value_t, rhs_space>& csr) {
    csr_t<index_t, offset_t, value_t, memory_space_t::host> h(csr);

    rows = h.rows;
    cols = h.cols;
    nnzs = h.nnzs;

    std::size_t mpr = 0;
    for (std::size_t r = 0; r < rows; ++r) {
      auto deg = static_cast<std::size_t>(h.offsets[r + 1] - h.offsets[r]);
      mpr = std::max(mpr, deg);
    }
    pitch = mpr;

    thrust::host_vector<index_t> h_idx(rows * pitch, sentinel());
    thrust::host_vector<value_t> h_val(rows * pitch, value_t(0));

    for (std::size_t r = 0; r < rows; ++r) {
      auto begin = h.offsets[r];
      auto end = h.offsets[r + 1];
      for (auto k = begin; k < end; ++k) {
        const std::size_t slot = r * pitch + (k - begin);
        h_idx[slot] = h.indices[k];
        h_val[slot] = h.values[k];
      }
    }

    indices = vector_t<index_t, space>(rows * pitch);
    values = vector_t<value_t, space>(rows * pitch);
    thrust::copy(h_idx.begin(), h_idx.end(), indices.begin());
    thrust::copy(h_val.begin(), h_val.end(), values.begin());
  }
};  // struct ell_t

}  // namespace loops
