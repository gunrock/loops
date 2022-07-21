#pragma once

#include <loops/container/vector.hxx>
#include <loops/memory.hxx>

#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

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

  vector_t<index_t, space> row_indices;  /// I
  vector_t<index_t, space> col_indices;  /// J
  vector_t<value_t, space> values;       /// V

  coo_t() : rows(0), cols(0), nnzs(0), row_indices(), col_indices(), values() {}

  coo_t(std::size_t r, std::size_t c, std::size_t nnz)
      : rows(r),
        cols(c),
        nnzs(nnz),
        row_indices(nnz),
        col_indices(nnz),
        values(nnz) {}

  template <auto rhs_space>
  coo_t(const coo_t<index_t, value_t, rhs_space>& rhs)
      : rows(rhs.rows),
        cols(rhs.cols),
        nnzs(rhs.nnzs),
        row_indices(rhs.row_indices),
        col_indices(rhs.col_indices),
        values(rhs.values) {}

  void sort_by_row() {
    auto begin = thrust::make_zip_iterator(
        thrust::make_tuple(row_indices.begin(), col_indices.begin()));
    auto end = thrust::make_zip_iterator(
        thrust::make_tuple(row_indices.end(), col_indices.end()));
    sort(begin, end);
  }

  void sort_by_column() {
    auto begin = thrust::make_zip_iterator(
        thrust::make_tuple(col_indices.begin(), row_indices.begin()));
    auto end = thrust::make_zip_iterator(
        thrust::make_tuple(col_indices.end(), row_indices.end()));
    sort(begin, end);
  }

 private:
  template <typename begin_it_t, typename end_it_t>
  void sort(begin_it_t& begin, end_it_t& end) {
    thrust::sort_by_key(begin, end, values.begin());
  }
};  // struct coo_t

}  // namespace loops