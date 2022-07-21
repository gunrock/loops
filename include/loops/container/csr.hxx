#pragma once

#include <loops/container/vector.hxx>
#include <loops/container/formats.hxx>
#include <loops/container/detail/convert.hxx>
#include <loops/memory.hxx>

#include <thrust/transform.h>

namespace loops {

using namespace memory;

/**
 * @brief Compressed Sparse Row (CSR) format.
 *
 * @tparam index_t Type of the nonzero elements indices.
 * @tparam offset_t Type of the row offsets.
 * @tparam value_t Type of the nonzero elements values.
 */
template <typename index_t,
          typename offset_t,
          typename value_t,
          memory_space_t space = memory_space_t::device>
struct csr_t {
  std::size_t rows;
  std::size_t cols;
  std::size_t nnzs;

  vector_t<offset_t, space> offsets;  /// Ap
  vector_t<index_t, space> indices;   /// Aj
  vector_t<value_t, space> values;    /// Ax

  /**
   * @brief Construct a new csr object with everything initialized to zero.
   *
   */
  csr_t() : rows(0), cols(0), nnzs(0), offsets(), indices(), values() {}

  /**
   * @brief Construct a new csr object with the given dimensions.
   *
   * @param r Number of rows.
   * @param c Number of columns.
   * @param nnz Number of non-zero elements.
   */
  csr_t(const std::size_t& r, const std::size_t& c, const std::size_t& nnz)
      : rows(r),
        cols(c),
        nnzs(nnz),
        offsets(r + 1),
        indices(nnz),
        values(nnz) {}

  /**
   * @brief Construct a new csr from another csr object on host.
   *
   * @param rhs csr_t<index_t, offset_t, value_t, rhs_space>
   */
  template <auto rhs_space>
  csr_t(const csr_t<index_t, offset_t, value_t, rhs_space>& rhs)
      : rows(rhs.rows),
        cols(rhs.cols),
        nnzs(rhs.nnzs),
        offsets(rhs.offsets),
        indices(rhs.indices),
        values(rhs.values) {}

  /**
   * @brief Construct a new csr object from coordinate format (COO).
   *
   * @param coo coo_t<index_t, value_t, auto>
   */
  template <auto rhs_space>
  csr_t(const coo_t<index_t, value_t, rhs_space>& coo)
      : rows(coo.rows),
        cols(coo.cols),
        nnzs(coo.nnzs),
        offsets(coo.rows + 1),
        indices(coo.col_indices),
        values(coo.values) {
    /// TODO: Do not need this copy for all cases.
    vector_t<index_t, space> _row_indices = coo.row_indices;
    detail::indices_to_offsets(_row_indices, offsets);
  }
};  // struct csr_t

}  // namespace loops