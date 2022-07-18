#pragma once

#include <loops/container/vector.hxx>
#include <loops/container/formats.hxx>
#include <loops/memory.hxx>

#include <thrust/transform.h>

namespace loops {

using namespace memory;

/**
 * @brief Compressed Sparse Row (CSR) format.
 *
 * @tparam index_t
 * @tparam offset_t
 * @tparam value_t
 */
template <typename index_t,
          typename offset_t,
          typename value_t,
          memory_space_t space = memory_space_t::device>
struct csr_t {
  std::size_t rows;
  std::size_t cols;
  std::size_t nnzs;

  vector_t<offset_t, space> offsets;  // Ap
  vector_t<index_t, space> indices;   // Aj
  vector_t<value_t, space> values;    // Ax

  csr_t() : rows(0), cols(0), nnzs(0), offsets(), indices(), values() {}

  csr_t(std::size_t r, std::size_t c, std::size_t nnz)
      : rows(r),
        cols(c),
        nnzs(nnz),
        offsets(r + 1),
        indices(nnz),
        values(nnz) {}

  ~csr_t() {}

  /**
   * @brief Copy constructor.
   * @param rhs
   */
  template <typename _csr_t>
  csr_t(const _csr_t& rhs)
      : rows(rhs.rows),
        cols(rhs.cols),
        nnzs(rhs.nnzs),
        offsets(rhs.offsets),
        indices(rhs.indices),
        values(rhs.values) {}

  /**
   * @brief Convert a Coordinate Sparse Format into Compressed Sparse Row
   * Format.
   *
   * @tparam index_t
   * @tparam offset_t
   * @tparam value_t
   * @param coo
   * @return csr_t<space, index_t, offset_t, value_t>&
   */
  csr_t<index_t, offset_t, value_t, space> from_coo(
      const coo_t<index_t, value_t, memory_space_t::host>& coo) {
    rows = coo.rows;
    cols = coo.cols;
    nnzs = coo.nnzs;

    // Vectors for conversions.
    vector_t<offset_t, memory_space_t::host> Ap(rows + 1);
    vector_t<index_t, memory_space_t::host> Aj(nnzs);
    vector_t<value_t, memory_space_t::host> Ax(nnzs);

    // compute number of non-zero entries per row of A.
    for (std::size_t n = 0; n < nnzs; ++n) {
      ++Ap[coo.row_indices[n]];
    }

    // cumulative sum the nnz per row to get offsets[].
    for (std::size_t i = 0, sum = 0; i < rows; ++i) {
      index_t temp = Ap[i];
      Ap[i] = sum;
      sum += temp;
    }
    Ap[rows] = nnzs;

    // write coordinate column indices and nonzero values into CSR's
    // column indices and nonzero values.
    for (std::size_t n = 0; n < nnzs; ++n) {
      index_t row = coo.row_indices[n];
      index_t dest = Ap[row];

      Aj[dest] = coo.col_indices[n];
      Ax[dest] = coo.values[n];

      ++Ap[row];
    }

    for (std::size_t i = 0, last = 0; i <= rows; ++i) {
      index_t temp = Ap[i];
      Ap[i] = last;
      last = temp;
    }

    // If returning a device csr_t, move coverted data to device.
    offsets = Ap;
    indices = Aj;
    values = Ax;

    return *this;  // CSR representation (with possible duplicates)
  }

  void read_binary(std::string filename) {
    FILE* file = fopen(filename.c_str(), "rb");

    // Read metadata
    assert(fread(&rows, sizeof(std::size_t), 1, file) != 0);
    assert(fread(&cols, sizeof(std::size_t), 1, file) != 0);
    assert(fread(&nnzs, sizeof(std::size_t), 1, file) != 0);

    offsets.resize(rows + 1);
    indices.resize(nnzs);
    values.resize(nnzs);

    if (space == memory_space_t::device) {
      assert(space == memory_space_t::device);

      thrust::host_vector<offset_t> h_offsets(rows + 1);
      thrust::host_vector<index_t> h_indices(nnzs);
      thrust::host_vector<value_t> h_values(nnzs);

      assert(fread(memory::raw_pointer_cast(h_offsets.data()), sizeof(offset_t),
                   rows + 1, file) != 0);
      assert(fread(memory::raw_pointer_cast(h_indices.data()), sizeof(index_t),
                   nnzs, file) != 0);
      assert(fread(memory::raw_pointer_cast(h_values.data()), sizeof(value_t),
                   nnzs, file) != 0);

      // Copy data from host to device
      offsets = h_offsets;
      indices = h_indices;
      values = h_values;

    } else {
      assert(space == memory_space_t::host);

      assert(fread(memory::raw_pointer_cast(offsets.data()), sizeof(offset_t),
                   rows + 1, file) != 0);
      assert(fread(memory::raw_pointer_cast(indices.data()), sizeof(index_t),
                   nnzs, file) != 0);
      assert(fread(memory::raw_pointer_cast(values.data()), sizeof(value_t),
                   nnzs, file) != 0);
    }
  }

  void write_binary(std::string filename) {
    FILE* file = fopen(filename.c_str(), "wb");

    // Write metadata
    fwrite(&rows, sizeof(std::size_t), 1, file);
    fwrite(&cols, sizeof(std::size_t), 1, file);
    fwrite(&nnzs, sizeof(std::size_t), 1, file);

    // Write data
    if (space == memory_space_t::device) {
      assert(space == memory_space_t::device);

      thrust::host_vector<offset_t> h_offsets(offsets);
      thrust::host_vector<index_t> h_indices(indices);
      thrust::host_vector<value_t> h_values(values);

      fwrite(memory::raw_pointer_cast(h_offsets.data()), sizeof(offset_t),
             rows + 1, file);
      fwrite(memory::raw_pointer_cast(h_indices.data()), sizeof(index_t), nnzs,
             file);
      fwrite(memory::raw_pointer_cast(h_values.data()), sizeof(value_t), nnzs,
             file);
    } else {
      assert(space == memory_space_t::host);

      fwrite(memory::raw_pointer_cast(offsets.data()), sizeof(offset_t),
             rows + 1, file);
      fwrite(memory::raw_pointer_cast(indices.data()), sizeof(index_t), nnzs,
             file);
      fwrite(memory::raw_pointer_cast(values.data()), sizeof(value_t), nnzs,
             file);
    }

    fclose(file);
  }

};  // struct csr_t

}  // namespace loops