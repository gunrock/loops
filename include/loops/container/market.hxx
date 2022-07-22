/**
 * @file matrix_market.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2022-02-03
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <string>

#include <loops/container/detail/mmio.hxx>
#include <loops/util/filepath.hxx>
#include <loops/container/formats.hxx>
#include <loops/memory.hxx>

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
template <typename vertex_t, typename edge_t, typename weight_t>
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

  // mtx are generally written as coordinate format
  coo_t<vertex_t, weight_t, memory_space_t::host> coo;

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
  coo_t<vertex_t, weight_t, memory_space_t::host>& load(std::string _filename) {
    filename = _filename;
    dataset = extract_dataset(extract_filename(filename));

    file_t file;

    /// Load MTX information from the file.
    if ((file = fopen(filename.c_str(), "r")) == NULL) {
      std::cerr << "File could not be opened: " << filename << std::endl;
      exit(1);
    }

    /// TODO: Add support for mtx with no banners.
    if (mm_read_banner(file, &code) != 0) {
      std::cerr << "Could not process Matrix Market banner" << std::endl;
      exit(1);
    }

    /// TODO: Update C-interface to support unsigned ints instead.
    int num_rows, num_columns, num_nonzeros;
    if ((mm_read_mtx_crd_size(file, &num_rows, &num_columns, &num_nonzeros)) !=
        0) {
      std::cerr << "Could not read file info (M, N, NNZ)" << std::endl;
      exit(1);
    }

    /// Allocate memory for the matrix.
    coo.rows = (std::size_t)num_rows;
    coo.cols = (std::size_t)num_columns;
    coo.nnzs = (std::size_t)num_nonzeros;
    coo.row_indices.resize(num_nonzeros);
    coo.col_indices.resize(num_nonzeros);
    coo.values.resize(num_nonzeros);

    if (mm_is_coordinate(code))
      format = matrix_market_format_t::coordinate;
    else
      format = matrix_market_format_t::array;

    /// Pattern matrices do not have nonzero values.
    if (mm_is_pattern(code)) {
      data = matrix_market_data_t::pattern;

      // pattern matrix defines sparsity pattern, but not values
      for (vertex_t i = 0; i < num_nonzeros; ++i) {
        vertex_t I = 0;
        vertex_t J = 0;
        assert(fscanf(file, " %d %d \n", &I, &J) == 2);

        // adjust from 1-based to 0-based indexing
        coo.row_indices[i] = (vertex_t)I - 1;
        coo.col_indices[i] = (vertex_t)J - 1;

        // use value 1.0 for all nonzero entries
        coo.values[i] = (weight_t)1.0;
      }
    }

    /// Real or Integer matrices have real or integer values.
    else if (mm_is_real(code) || mm_is_integer(code)) {
      if (mm_is_real(code))
        data = matrix_market_data_t::real;
      else
        data = matrix_market_data_t::integer;

      for (vertex_t i = 0; i < coo.nnzs; ++i) {
        vertex_t I = 0;
        vertex_t J = 0;
        double V = 0.0f;

        assert(fscanf(file, " %d %d %lf \n", &I, &J, &V) == 3);

        coo.row_indices[i] = (vertex_t)I - 1;
        coo.col_indices[i] = (vertex_t)J - 1;
        coo.values[i] = (weight_t)V;
      }
    } else {
      std::cerr << "Unrecognized matrix market format type" << std::endl;
      exit(1);
    }

    /// Symmetric matrices have symmetric halves.
    if (mm_is_symmetric(code)) {
      scheme = matrix_market_storage_scheme_t::symmetric;

      vertex_t off_diagonals = 0;
      for (vertex_t i = 0; i < coo.nnzs; ++i) {
        if (coo.row_indices[i] != coo.col_indices[i])
          ++off_diagonals;
      }

      // Duplicate off-diagonal entries for symmetric matrix.
      std::size_t _nonzeros = 2 * off_diagonals + (coo.nnzs - off_diagonals);
      coo_t<vertex_t, weight_t, memory_space_t::host> temp(coo.rows, coo.cols,
                                                           _nonzeros);

      vertex_t ptr = 0;
      for (vertex_t i = 0; i < coo.nnzs; ++i) {
        if (coo.row_indices[i] != coo.col_indices[i]) {
          temp.row_indices[ptr] = coo.row_indices[i];
          temp.col_indices[ptr] = coo.col_indices[i];
          temp.values[ptr] = coo.values[i];
          ++ptr;
          temp.col_indices[ptr] = coo.row_indices[i];
          temp.row_indices[ptr] = coo.col_indices[i];
          temp.values[ptr] = coo.values[i];
          ++ptr;
        } else {
          temp.row_indices[ptr] = coo.row_indices[i];
          temp.col_indices[ptr] = coo.col_indices[i];
          temp.values[ptr] = coo.values[i];
          ++ptr;
        }
      }

      // Move data to the original COO matrix.
      coo = temp;
    }  // end symmetric case

    fclose(file);

    return coo;
  }
};

}  // namespace loops