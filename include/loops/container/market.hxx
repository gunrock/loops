/**
 * @file matrix_market.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2020-10-09
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once

#include <string>
#include <limits>

#include <loops/container/detail/mmio.hxx>

#include <loops/util/filepath.hxx>
#include <loops/container/formats.hxx>
#include <loops/memory.hxx>
#include <loops/error.hxx>

namespace loops {

using namespace memory;

/**
 * @brief Matrix Market format supports two kind of formats, a sparse coordinate
 * format and a dense array format.
 *
 */
enum matrix_market_format_t { coordinate, array };

/**
 * @brief Data type defines the type of data presented in the file, things like,
 * are they real numbers, complex (real and imaginary), pattern (do not have
 * weights/nonzero-values), etc.
 *
 */
enum matrix_market_data_t { real, complex, pattern, integer };

/**
 * @brief Storage scheme defines the storage structure, symmetric matrix for
 * example will be symmetric over the diagonal. Skew is skew symmetric. Etc.
 *
 */
enum matrix_market_storage_scheme_t { general, hermitian, symmetric, skew };

/**
 * @brief Reads a MARKET graph from an input-stream
 * into a specified sparse format
 *
 * Here is an example of the matrix market format
 * +----------------------------------------------+
 * |%%MatrixMarket matrix coordinate real general | <--- header line
 * |%                                             | <--+
 * |% comments                                    |    |-- 0 or more comments
 * |%                                             | <--+
 * |  M N L                                       | <--- rows, columns, entries
 * |  I1 J1 A(I1, J1)                             | <--+
 * |  I2 J2 A(I2, J2)                             |    |
 * |  I3 J3 A(I3, J3)                             |    |-- L lines
 * |     . . .                                    |    |
 * |  IL JL A(IL, JL)                             | <--+
 * +----------------------------------------------+
 *
 * Indices are 1-based i.2. A(1,1) is the first element.
 */
template <typename index_t, typename offset_t, typename type_t>
struct matrix_market_t {
  // typedef FILE* file_t;
  // typedef MM_typecode matrix_market_code_t;

  using file_t = FILE*;
  using matrix_market_code_t = MM_typecode;

  std::string filename;
  std::string dataset;

  // Dataset characteristics
  matrix_market_code_t code;              // (ALL INFORMATION)
  matrix_market_format_t format;          // Sparse coordinate or dense array
  matrix_market_data_t data;              // Data type
  matrix_market_storage_scheme_t scheme;  // Storage scheme

  matrix_market_t() {}
  ~matrix_market_t() {}

  /**
   * @brief Loads the given .mtx file into a coordinate format, and returns the
   * coordinate array. This needs to be further extended to support dense
   * arrays, those are the only two formats mtx are written in.
   *
   * @param _filename input file name (.mtx)
   * @return coordinate sparse format
   */
  auto load(std::string _filename) {
    filename = _filename;
    dataset = extract_dataset(extract_filename(filename));

    file_t file;

    // Load MTX information
    if ((file = fopen(filename.c_str(), "r")) == NULL) {
      std::cerr << "File could not be opened: " << filename << std::endl;
      exit(1);
    }

    if (mm_read_banner(file, &code) != 0) {
      std::cerr << "Could not process Matrix Market banner" << std::endl;
      exit(1);
    }

    std::size_t num_rows, num_columns, num_nonzeros;
    if ((mm_read_mtx_crd_size(file, &num_rows, &num_columns, &num_nonzeros)) !=
        0) {
      std::cerr << "Could not read file info (M, N, NNZ)" << std::endl;
      exit(1);
    }

    error::throw_if_exception(
        num_rows >= std::numeric_limits<index_t>::max() ||
            num_columns >= std::numeric_limits<index_t>::max(),
        "index_t overflow");
    error::throw_if_exception(
        num_nonzeros >= std::numeric_limits<offset_t>::max(),
        "offset_t overflow");

    // mtx are generally written as coordinate format
    coo_t<index_t, type_t, memory_space_t::host> coo(
        (index_t)num_rows, (index_t)num_columns, (offset_t)num_nonzeros);

    if (mm_is_coordinate(code))
      format = matrix_market_format_t::coordinate;
    else
      format = matrix_market_format_t::array;

    if (mm_is_pattern(code)) {
      data = matrix_market_data_t::pattern;

      // pattern matrix defines sparsity pattern, but not values
      for (index_t i = 0; i < num_nonzeros; ++i) {
        std::size_t row_index{0}, col_index{0};
        auto num_assigned = fscanf(file, " %zu %zu \n", &row_index, &col_index);
        error::throw_if_exception(num_assigned != 2,
                                  "Could not read edge from market file");
        error::throw_if_exception(row_index == 0,
                                  "Market file is zero-indexed");
        error::throw_if_exception(col_index == 0,
                                  "Market file is zero-indexed");
        // set and adjust from 1-based to 0-based indexing
        coo.row_indices[i] = (index_t)row_index - 1;
        coo.col_indices[i] = (index_t)col_index - 1;
        coo.values[i] = (type_t)1.0;  // use value 1.0 for all nonzero entries
      }
    } else if (mm_is_real(code) || mm_is_integer(code)) {
      if (mm_is_real(code))
        data = matrix_market_data_t::real;
      else
        data = matrix_market_data_t::integer;

      for (index_t i = 0; i < coo.nnzs; ++i) {
        std::size_t row_index{0}, col_index{0};
        double weight{0.0};

        auto num_assigned =
            fscanf(file, " %zu %zu %lf \n", &row_index, &col_index, &weight);

        error::throw_if_exception(
            num_assigned != 3, "Could not read weighted edge from market file");
        error::throw_if_exception(row_index == 0,
                                  "Market file is zero-indexed");
        error::throw_if_exception(col_index == 0,
                                  "Market file is zero-indexed");

        coo.row_indices[i] = (index_t)row_index - 1;
        coo.col_indices[i] = (index_t)col_index - 1;
        coo.values[i] = (type_t)weight;
      }
    } else {
      std::cerr << "Unrecognized matrix market format type" << std::endl;
      exit(1);
    }

    if (mm_is_symmetric(code)) {  // duplicate off diagonal entries
      scheme = matrix_market_storage_scheme_t::symmetric;
      index_t off_diagonals = 0;
      for (index_t i = 0; i < coo.nnzs; ++i) {
        if (coo.row_indices[i] != coo.col_indices[i])
          ++off_diagonals;
      }

      index_t _nonzeros = 2 * off_diagonals + (coo.nnzs - off_diagonals);

      vector_t<index_t, memory_space_t::host> _I(_nonzeros);
      vector_t<index_t, memory_space_t::host> _J(_nonzeros);
      vector_t<type_t, memory_space_t::host> _V(_nonzeros);

      index_t ptr = 0;
      for (index_t i = 0; i < coo.nnzs; ++i) {
        if (coo.row_indices[i] != coo.col_indices[i]) {
          _I[ptr] = coo.row_indices[i];
          _J[ptr] = coo.col_indices[i];
          _V[ptr] = coo.values[i];
          ++ptr;
          _J[ptr] = coo.row_indices[i];
          _I[ptr] = coo.col_indices[i];
          _V[ptr] = coo.values[i];
          ++ptr;
        } else {
          _I[ptr] = coo.row_indices[i];
          _J[ptr] = coo.col_indices[i];
          _V[ptr] = coo.values[i];
          ++ptr;
        }
      }
      coo.row_indices = _I;
      coo.col_indices = _J;
      coo.values = _V;
      coo.nnzs = _nonzeros;
    }  // end symmetric case

    fclose(file);
    return coo;
  }
};
}  // namespace loops