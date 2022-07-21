/**
 * @file csc.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Interface for Compressed Sparse-Column format.
 * @version 0.1
 * @date 2022-07-21
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <loops/container/vector.hxx>
#include <loops/memory.hxx>

namespace loops {

using namespace memory;

/**
 * @brief Compressed Sparse Column (CSC) format.
 *
 * @tparam index_t
 * @tparam offset_t
 * @tparam value_t
 */
template <typename index_t,
          typename offset_t,
          typename value_t,
          memory_space_t space = memory_space_t::device>
struct csc_t {
  std::size_t rows;
  std::size_t cols;
  std::size_t nnzs;

  vector_t<offset_t, space> offsets;  /// Aj
  vector_t<index_t, space> indices;   /// Ap
  vector_t<value_t, space> values;    /// Ax

  /**
   * @brief Construct a new csc object with everything initialized to zero.
   *
   */
  csc_t() : rows(0), cols(0), nnzs(0), offsets(), indices(), values() {}

  /**
   * @brief Construct a new csc object with the given dimensions.
   *
   * @param r Number of rows.
   * @param c Number of columns.
   * @param nnz Number of non-zero elements.
   */
  csc_t(std::size_t r, std::size_t c, std::size_t nnz)
      : rows(r),
        cols(c),
        nnzs(nnz),
        offsets(r + 1),
        indices(nnz),
        values(nnz) {}

  /**
   * @brief Construct a new csc from another csc object on host/device.
   *
   * @param rhs csc_t<index_t, offset_t, value_t, rhs_space>
   */
  template <auto rhs_space>
  csc_t(const csc_t<index_t, offset_t, value_t, rhs_space>& rhs)
      : rows(rhs.rows),
        cols(rhs.cols),
        nnzs(rhs.nnzs),
        offsets(rhs.offsets),
        indices(rhs.indices),
        values(rhs.values) {}

  /**
   * @brief Construct a new csc object from coordinate format (COO).
   * @note This constructor creates a copy of the input COO matrix.
   *
   * @param coo coo_t<index_t, value_t, auto>
   */
  template <auto rhs_space>
  csc_t(const coo_t<index_t, value_t, rhs_space>& coo)
      : rows(coo.rows), cols(coo.cols), nnzs(coo.nnzs), offsets(coo.cols + 1) {
    coo_t<index_t, value_t, space> __(coo);
    __.sort_by_column();
    indices = std::move(__.row_indices);
    values = std::move(__.values);
    detail::indices_to_offsets(__.col_indices, offsets);
  }

};  // struct csc_t

}  // namespace loops