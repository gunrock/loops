/**
 * @file dia.hxx
 * @author Loops contributors
 * @brief Diagonal (DIA) format.
 * @version 0.1
 * @date 2026-05-06
 *
 * DIA stores the matrix as a set of nonzero diagonals. Each diagonal is
 * laid out as a length- @c stride vector (typically @c stride == @c rows ),
 * and a side @c diag_offsets[d] indexes which diagonal each row belongs
 * to:
 *
 *   diag_offsets[d] == 0  : main diagonal (M[r, r])
 *   diag_offsets[d] < 0   : sub-diagonal  (M[r, r + diag_offsets[d]])
 *   diag_offsets[d] > 0   : super-diagonal (M[r, r + diag_offsets[d]])
 *
 * @c values is a column-major dense buffer of shape
 * @c [num_diagonals, stride] : @c values[d * stride + r] is the value of
 * row @c r along diagonal @c d (or zero if @c (r, r + diag_offsets[d])
 * is out of bounds, in which case the column index is clamped and the
 * value is treated as zero).
 *
 * Padding rule: out-of-range entries are stored as zero. The kernel
 * still touches every (r, d) cell; the column-clamping prevents OOB
 * reads from @c x .
 *
 * Trade-off: DIA is *extremely* compact for banded matrices and lets
 * SpMV become a regular gather over @c x with no atomics, but it
 * degenerates badly for unstructured matrices (one diagonal per stored
 * @c (r - c) value, most cells zero). The container guards against
 * pathological inputs by surfacing the resulting @c num_diagonals so
 * tests can skip DIA on matrices where it doesn't make sense.
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <loops/container/formats.hxx>
#include <loops/container/vector.hxx>
#include <loops/memory.hxx>

#include <thrust/host_vector.h>

#include <algorithm>
#include <cstddef>
#include <set>
#include <vector>

namespace loops {

using namespace memory;

/**
 * @brief Diagonal (DIA) container.
 *
 * @tparam index_t  Type of column-index / diagonal-offset (signed!).
 * @tparam offset_t Type used for any offset arrays (currently unused; kept
 *                  for API symmetry with the other containers).
 * @tparam value_t  Non-zero value type.
 * @tparam space    Memory space (host/device) for storage.
 */
template <typename index_t,
          typename offset_t,
          typename value_t,
          memory_space_t space = memory_space_t::device>
struct dia_t {
  std::size_t rows;
  std::size_t cols;
  std::size_t nnzs;            /// Original (non-padded) nonzero count.
  std::size_t stride;          /// Length of each stored diagonal (rows).
  std::size_t num_diagonals;   /// Distinct (r - c) values stored.

  vector_t<index_t, space> diag_offsets;  /// length num_diagonals.
  vector_t<value_t, space> values;        /// num_diagonals * stride, column-major.

  dia_t()
      : rows(0),
        cols(0),
        nnzs(0),
        stride(0),
        num_diagonals(0),
        diag_offsets(),
        values() {}

  template <auto rhs_space>
  dia_t(const dia_t<index_t, offset_t, value_t, rhs_space>& rhs)
      : rows(rhs.rows),
        cols(rhs.cols),
        nnzs(rhs.nnzs),
        stride(rhs.stride),
        num_diagonals(rhs.num_diagonals),
        diag_offsets(rhs.diag_offsets),
        values(rhs.values) {}

  /**
   * @brief Build a DIA view of an existing CSR matrix on the host.
   *
   * Sweeps the CSR once to find the set of distinct @c (r - c) values
   * (the populated diagonals), then sweeps again to scatter values into
   * the dense @c [num_diagonals, stride] buffer.
   */
  template <auto rhs_space, typename csr_offset_t>
  dia_t(const csr_t<index_t, csr_offset_t, value_t, rhs_space>& csr)
      : rows(csr.rows), cols(csr.cols), nnzs(csr.nnzs) {
    stride = rows;

    csr_t<index_t, csr_offset_t, value_t, memory_space_t::host> h(csr);

    std::set<index_t> diag_set;
    for (std::size_t r = 0; r < rows; ++r) {
      const csr_offset_t a_lo = h.offsets[r];
      const csr_offset_t a_hi = h.offsets[r + 1];
      for (csr_offset_t a = a_lo; a < a_hi; ++a) {
        const index_t c = h.indices[a];
        diag_set.insert(static_cast<index_t>(c) -
                        static_cast<index_t>(r));  // (col - row)
      }
    }

    std::vector<index_t> h_diag_offsets(diag_set.begin(), diag_set.end());
    num_diagonals = h_diag_offsets.size();

    std::vector<value_t> h_values(num_diagonals * stride, value_t{0});

    // Reverse-lookup: diag-offset -> diagonal-index.
    std::vector<std::pair<index_t, std::size_t>> diag_idx;
    diag_idx.reserve(num_diagonals);
    for (std::size_t d = 0; d < num_diagonals; ++d) {
      diag_idx.emplace_back(h_diag_offsets[d], d);
    }
    std::sort(diag_idx.begin(), diag_idx.end());

    auto find_diag = [&](index_t off) -> std::size_t {
      auto it = std::lower_bound(diag_idx.begin(), diag_idx.end(),
                                  std::make_pair(off, std::size_t{0}),
                                  [](const auto& a, const auto& b) {
                                    return a.first < b.first;
                                  });
      return it->second;
    };

    for (std::size_t r = 0; r < rows; ++r) {
      const csr_offset_t a_lo = h.offsets[r];
      const csr_offset_t a_hi = h.offsets[r + 1];
      for (csr_offset_t a = a_lo; a < a_hi; ++a) {
        const index_t c = h.indices[a];
        const index_t off = static_cast<index_t>(c) -
                            static_cast<index_t>(r);
        const std::size_t d = find_diag(off);
        h_values[d * stride + r] = h.values[a];
      }
    }

    diag_offsets = vector_t<index_t, memory_space_t::host>(
        h_diag_offsets.begin(), h_diag_offsets.end());
    values = vector_t<value_t, memory_space_t::host>(h_values.begin(),
                                                      h_values.end());
  }
};  // struct dia_t

}  // namespace loops
