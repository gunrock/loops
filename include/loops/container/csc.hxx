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

  csc_t() : rows(0), cols(0), nnzs(0), offsets(), indices(), values() {}

  csc_t(std::size_t r, std::size_t c, std::size_t nnz)
      : rows(r),
        cols(c),
        nnzs(nnz),
        offsets(c + 1),
        indices(nnz),
        values(nnz) {}

  ~csc_t() {}

};  // struct csc_t

}  // namespace loops