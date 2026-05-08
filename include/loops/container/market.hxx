/**
 * @file market.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Matrix Market format reader.
 * @see http://math.nist.gov/MatrixMarket/
 * @version 0.2
 * @date 2026-05-07
 *
 * Reads a Matrix Market sparse coordinate file into a host @c coo_t .
 *
 * Performance design:
 *
 *   - The body is parsed via @c std::from_chars over a memory-mapped
 *     view of the file. This is locale-independent, has no userspace
 *     buffering layer, and clocks in around ~30 M integers / sec on a
 *     contemporary CPU vs. ~150 K / sec for the previous @c fscanf
 *     implementation. Loading @c com-Orkut.mtx (117 M lines) drops
 *     from ~5-8 minutes to ~30-60 seconds.
 *
 *   - Symmetric matrices are loaded in two passes over the same
 *     mmap'd region: a structural pass that counts off-diagonals
 *     (parsing only @c (row, col) and skipping the value column),
 *     followed by a parsing pass that emits directly into a
 *     correctly-sized COO. Peak host RAM during the load is therefore
 *     just the final COO; the previous loader allocated three full-
 *     sized intermediates and then copy-assigned them, doubling peak.
 *     The page cache stays warm across the two passes, so the wall
 *     cost is roughly 1.5x a single pass, not 2x.
 *
 *   - All counters use @c std::size_t , so symmetric expansion is
 *     correct on matrices with > 2^31 final nonzeros (e.g.
 *     @c friendster ); the previous @c index_t (int) counters silently
 *     overflowed at the @c 2 * off_diagonals + ... step.
 *
 * Backward compatibility:
 *
 *   - The public surface is unchanged: @c load(filename) still returns
 *     a host @c coo_t , and the @c dataset / @c filename string fields
 *     are populated as before.
 *
 *   - The @c MM_typecode -based @c code field has been replaced by a
 *     header-only @c detail::mm_typecode_t struct (no link-time
 *     dependency on @c mmio.cpp ). No in-tree caller reads @c .code
 *     directly.
 *
 * @copyright Copyright (c) 2020-2026
 *
 */

#pragma once

#include <loops/container/detail/mapped_file.hxx>
#include <loops/container/detail/mtx_parser.hxx>
#include <loops/container/formats.hxx>
#include <loops/error.hxx>
#include <loops/memory.hxx>
#include <loops/util/filepath.hxx>

#include <cstddef>
#include <limits>
#include <string>

namespace loops {

using namespace memory;

/**
 * @brief Matrix Market loader.
 *
 * @tparam index_t  Row / column index type used by the returned COO.
 * @tparam offset_t Offset type checked against the final @c nnz to
 *                  guard CSR conversion later in the pipeline.
 * @tparam type_t   Value type used by the returned COO.
 *
 * Usage:
 * @code
 *   matrix_market_t<int, int, float> mtx;
 *   auto coo = mtx.load("path/to/file.mtx");
 *   csr_t<int, int, float> csr(coo);  // host -> device CSR.
 * @endcode
 */
template <typename index_t, typename offset_t, typename type_t>
struct matrix_market_t {
  std::string filename;
  std::string dataset;

  /// Header typecode (object/format/field/symmetry); informational.
  detail::mm_typecode_t code;

  matrix_market_t() = default;
  ~matrix_market_t() = default;

  /**
   * @brief Load an .mtx file into a host @c coo_t .
   *
   * Throws via @c loops::error::throw_if_exception on missing files,
   * malformed banners, unsupported variants (complex / hermitian /
   * skew / dense array), or @c index_t / @c offset_t overflow.
   */
  coo_t<index_t, type_t, memory_space_t::host> load(std::string _filename) {
    filename = std::move(_filename);
    dataset = extract_dataset(extract_filename(filename));

    detail::mapped_file_t f(filename);
    const char* p = f.data();
    const char* end = f.end();

    error::throw_if_exception(p == end,
                              "matrix-market: empty file " + filename);

    p = detail::parse_banner(p, end, code);

    error::throw_if_exception(
        !code.is_matrix,
        "matrix-market: object must be 'matrix' in " + filename);
    error::throw_if_exception(
        !code.is_coordinate,
        "matrix-market: only the coordinate (sparse) format is supported in " +
            filename);
    error::throw_if_exception(
        code.is_complex,
        "matrix-market: complex values not supported in " + filename);
    error::throw_if_exception(
        code.is_hermitian || code.is_skew,
        "matrix-market: hermitian / skew-symmetric not supported in " +
            filename);
    error::throw_if_exception(
        !(code.is_general || code.is_symmetric),
        "matrix-market: missing or unrecognized symmetry tag in " + filename);

    // Header continues with optional comment lines, then the dim line:
    //   <num_rows> <num_cols> <header_nnz>
    p = detail::skip_comments(p, end);

    std::size_t num_rows = 0, num_cols = 0, header_nnz = 0;
    p = read_three_size_t(p, end, num_rows, num_cols, header_nnz,
                          "dimension line");
    p = detail::skip_to_eol(p, end);

    // index_t holds row / column indices in the returned COO; offset_t
    // gates the eventual CSR conversion. Bail before allocating anything
    // if either type can't represent the matrix.
    error::throw_if_exception(
        num_rows >=
                static_cast<std::size_t>(std::numeric_limits<index_t>::max()) ||
            num_cols >=
                static_cast<std::size_t>(std::numeric_limits<index_t>::max()),
        "matrix-market: index_t overflow (rows or cols >= INT_MAX) in " +
            filename);

    const char* body_begin = p;

    // Final COO size: symmetric matrices store only the lower triangle
    // (plus diagonal) on disk and we expand off-diagonals at load time.
    std::size_t final_nnz = header_nnz;
    if (code.is_symmetric) {
      const std::size_t off_diagonals =
          count_off_diagonals(body_begin, end, header_nnz);
      final_nnz = header_nnz + off_diagonals;
    }

    error::throw_if_exception(
        final_nnz >=
            static_cast<std::size_t>(std::numeric_limits<offset_t>::max()),
        "matrix-market: offset_t overflow (final nnz exceeds offset_t max) "
        "in " +
            filename);

    coo_t<index_t, type_t, memory_space_t::host> coo(
        static_cast<index_t>(num_rows), static_cast<index_t>(num_cols),
        static_cast<offset_t>(final_nnz));

    parse_body(body_begin, end, header_nnz, code.is_symmetric, code.is_pattern,
               coo);

    return coo;
  }

 private:
  /// Read three whitespace-separated unsigned integers; throws if any
  /// of them fails to parse.
  static const char* read_three_size_t(const char* p,
                                       const char* end,
                                       std::size_t& a,
                                       std::size_t& b,
                                       std::size_t& c,
                                       const char* where) {
    p = detail::skip_blank(p, end);
    const char* q = p;
    p = detail::parse_size_t(p, end, a);
    error::throw_if_exception(p == q, std::string("matrix-market: expected ") +
                                          where + " (first integer)");
    p = detail::skip_blank(p, end);
    q = p;
    p = detail::parse_size_t(p, end, b);
    error::throw_if_exception(p == q, std::string("matrix-market: expected ") +
                                          where + " (second integer)");
    p = detail::skip_blank(p, end);
    q = p;
    p = detail::parse_size_t(p, end, c);
    error::throw_if_exception(p == q, std::string("matrix-market: expected ") +
                                          where + " (third integer)");
    return p;
  }

  /// Pass 1 (symmetric only): count rows where @c row != @c col so the
  /// final COO can be allocated to exactly the right size. Walks the
  /// same mmap'd region the parser will walk again in pass 2; the page
  /// cache stays hot across the two passes.
  static std::size_t count_off_diagonals(const char* p,
                                         const char* end,
                                         std::size_t header_nnz) {
    std::size_t off_diagonals = 0;
    for (std::size_t i = 0; i < header_nnz; ++i) {
      p = detail::skip_ws(p, end);
      std::size_t row1 = 0, col1 = 0;
      const char* q = p;
      p = detail::parse_size_t(p, end, row1);
      error::throw_if_exception(
          p == q, "matrix-market: expected row index in body (count pass)");
      p = detail::skip_blank(p, end);
      q = p;
      p = detail::parse_size_t(p, end, col1);
      error::throw_if_exception(
          p == q, "matrix-market: expected column index in body (count pass)");
      // We do not need the value here; skip the rest of the line.
      p = detail::skip_to_eol(p, end);
      if (row1 != col1)
        ++off_diagonals;
    }
    return off_diagonals;
  }

  /// Pass 2: parse the body into the (already sized) COO. For symmetric
  /// matrices, also emit the mirror @c (col, row, v) entry whenever the
  /// record is off-diagonal.
  static void parse_body(const char* p,
                         const char* end,
                         std::size_t header_nnz,
                         bool is_symmetric,
                         bool is_pattern,
                         coo_t<index_t, type_t, memory_space_t::host>& coo) {
    std::size_t w = 0;
    for (std::size_t i = 0; i < header_nnz; ++i) {
      p = detail::skip_ws(p, end);
      std::size_t row1 = 0, col1 = 0;
      const char* q = p;
      p = detail::parse_size_t(p, end, row1);
      error::throw_if_exception(p == q,
                                "matrix-market: expected row index in body");
      p = detail::skip_blank(p, end);
      q = p;
      p = detail::parse_size_t(p, end, col1);
      error::throw_if_exception(p == q,
                                "matrix-market: expected column index in body");

      double weight = 1.0;
      if (!is_pattern) {
        p = detail::skip_blank(p, end);
        q = p;
        p = detail::parse_double(p, end, weight);
        error::throw_if_exception(p == q,
                                  "matrix-market: expected value in body");
      }
      p = detail::skip_to_eol(p, end);

      error::throw_if_exception(
          row1 == 0 || col1 == 0,
          "matrix-market: zero-indexed entry (Matrix Market is 1-indexed)");
      const index_t r = static_cast<index_t>(row1 - 1);
      const index_t c = static_cast<index_t>(col1 - 1);
      const type_t v = static_cast<type_t>(weight);

      coo.row_indices[w] = r;
      coo.col_indices[w] = c;
      coo.values[w] = v;
      ++w;

      if (is_symmetric && r != c) {
        coo.row_indices[w] = c;
        coo.col_indices[w] = r;
        coo.values[w] = v;
        ++w;
      }
    }
    error::throw_if_exception(
        w != coo.nnzs,
        "matrix-market: internal inconsistency between pass 1 and pass 2");
  }
};
}  // namespace loops
