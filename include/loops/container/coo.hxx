#pragma once

#include <loops/container/vector.hxx>
#include <loops/memory.hxx>

namespace loops {

using namespace memory;

/**
 * @brief Coordinate (COO) format.
 *
 * @tparam index_t
 * @tparam value_t
 */
template <typename index_t,
          typename value_t,
          memory_space_t space = memory_space_t::device>
struct coo_t {
  std::size_t rows;
  std::size_t cols;
  std::size_t nnzs;

  vector_t<index_t, space> row_indices;  // I
  vector_t<index_t, space> col_indices;  // J
  vector_t<value_t, space> values;       // V

  coo_t() : rows(0), cols(0), nnzs(0), row_indices(), col_indices(), values() {}

  coo_t(std::size_t r, std::size_t c, std::size_t nnz)
      : rows(r),
        cols(c),
        nnzs(nnz),
        row_indices(nnz),
        col_indices(nnz),
        values(nnz) {}

  coo_t(const coo_t<index_t, value_t, memory_space_t::device>& rhs)
      : rows(rhs.rows),
        cols(rhs.cols),
        nnzs(rhs.nnzs),
        row_indices(rhs.row_indices),
        col_indices(rhs.col_indices),
        values(rhs.values) {}

  coo_t(const coo_t<index_t, value_t, memory_space_t::host>& rhs)
      : rows(rhs.rows),
        cols(rhs.cols),
        nnzs(rhs.nnzs),
        row_indices(rhs.row_indices),
        col_indices(rhs.col_indices),
        values(rhs.values) {}
};  // struct coo_t

}  // namespace loops