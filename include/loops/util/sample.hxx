/**
 * @file sample.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2022-07-19
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <loops/memory.hxx>
#include <loops/container/formats.hxx>

namespace loops {
namespace sample {

using namespace memory;

/**
 * @brief Returns a small sample CSR matrix of size 4x4x4.
 *
 * @par Overview
 *
 * **Logical Matrix Representation**
 * \code
 * r/c  0 1 2 3
 * 0 [ 0 0 0 0 ]
 * 1 [ 5 8 0 0 ]
 * 2 [ 0 0 3 0 ]
 * 3 [ 0 6 0 0 ]
 * \endcode
 *
 * **Logical Graph Representation**
 * \code
 * (i, j) [w]
 * (1, 0) [5]
 * (1, 1) [8]
 * (2, 2) [3]
 * (3, 1) [6]
 * \endcode
 *
 * **CSR Matrix Representation**
 * \code
 * VALUES       = [ 5 8 3 6 ]
 * COLUMN_INDEX = [ 0 1 2 1 ]
 * ROW_OFFSETS  = [ 0 0 2 3 4 ]
 * \endcode
 *
 * @tparam space Memory space of the CSR matrix.
 * @tparam index_t Type of vertex.
 * @tparam offset_t Type of edge.
 * @tparam value_t Type of weight.
 * @return csr_t<index_t, offset_t, value_t> CSR matrix.
 */
template <memory_space_t space = memory_space_t::device,
          typename index_t = int,
          typename offset_t = int,
          typename value_t = float>
csr_t<index_t, offset_t, value_t, space> csr() {
  csr_t<index_t, offset_t, value_t, memory_space_t::host> matrix(4, 4, 4);

  // Row Offsets
  matrix.offsets[0] = 0;
  matrix.offsets[1] = 0;
  matrix.offsets[2] = 2;
  matrix.offsets[3] = 3;
  matrix.offsets[4] = 4;

  // Column Indices
  matrix.indices[0] = 0;
  matrix.indices[1] = 1;
  matrix.indices[2] = 2;
  matrix.indices[3] = 1;

  // Non-zero values
  matrix.values[0] = 5;
  matrix.values[1] = 8;
  matrix.values[2] = 3;
  matrix.values[3] = 6;

  if (space == memory_space_t::host) {
    return matrix;
  } else {
    csr_t<index_t, offset_t, value_t, memory_space_t::device> d_matrix(matrix);
    return d_matrix;
  }
}

}  // namespace sample
}  // namespace loops