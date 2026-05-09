/**
 * @file bcsr.hxx
 * @author Loops contributors
 * @brief Block Compressed Sparse Row (BCSR) format.
 * @version 0.1
 * @date 2026-05-06
 *
 * BCSR partitions the matrix into a regular grid of R-by-C dense blocks
 * and stores only the *non-empty* blocks. The compression is row-wise at
 * the block level:
 *
 *   - num_block_rows = ceil(rows / R)
 *   - num_block_cols = ceil(cols / C)
 *   - block_offsets[br] = starting block id of block-row @c br
 *   - block_col_indices[b] = block-column for block @c b
 *   - values[b * R * C + i * C + j] = matrix[br*R + i, bc*C + j], where
 *     bc = block_col_indices[b]
 *
 * Padding rule: matrix dimensions are rounded up to multiples of @c R and
 * @c C ; out-of-range entries inside a stored block are zero. This keeps
 * SpMV branch-free at the block level and lets the kernel use compile-time
 * loop bounds for the per-block update.
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <loops/container/formats.hxx>
#include <loops/container/vector.hxx>
#include <loops/memory.hxx>

#include <thrust/host_vector.h>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace loops {

using namespace memory;

/**
 * @brief Block Compressed Sparse Row (BCSR) container.
 *
 * @tparam R Block height (rows per dense block) - compile-time.
 * @tparam C Block width  (cols per dense block) - compile-time.
 * @tparam index_t  Block-column-index type.
 * @tparam offset_t Block-offset type.
 * @tparam value_t  Non-zero value type.
 * @tparam space    Memory space (host/device) for storage.
 */
template <std::size_t R,
          std::size_t C,
          typename index_t,
          typename offset_t,
          typename value_t,
          memory_space_t space = memory_space_t::device>
struct bcsr_t {
  static_assert(R > 0 && C > 0, "BCSR block dims must be positive.");
  static constexpr std::size_t kBlockRows = R;
  static constexpr std::size_t kBlockCols = C;
  static constexpr std::size_t kBlockSize = R * C;

  std::size_t rows;            ///< Original (unpadded) row count.
  std::size_t cols;            ///< Original (unpadded) col count.
  std::size_t nnzs;            ///< Original (unpadded) non-zero count.
  std::size_t num_block_rows;  ///< ceil(rows / R).
  std::size_t num_block_cols;  ///< ceil(cols / C).
  std::size_t num_blocks;      ///< Stored (non-empty) blocks.

  vector_t<offset_t, space>
      block_offsets;  ///< Block-row offsets; length num_block_rows + 1.
  vector_t<index_t, space>
      block_col_indices;  ///< Block-column indices; length num_blocks.
  vector_t<value_t, space>
      values;  ///< Dense block values; length num_blocks * R * C.

  bcsr_t()
      : rows(0),
        cols(0),
        nnzs(0),
        num_block_rows(0),
        num_block_cols(0),
        num_blocks(0),
        block_offsets(),
        block_col_indices(),
        values() {}

  template <auto rhs_space>
  bcsr_t(const bcsr_t<R, C, index_t, offset_t, value_t, rhs_space>& rhs)
      : rows(rhs.rows),
        cols(rhs.cols),
        nnzs(rhs.nnzs),
        num_block_rows(rhs.num_block_rows),
        num_block_cols(rhs.num_block_cols),
        num_blocks(rhs.num_blocks),
        block_offsets(rhs.block_offsets),
        block_col_indices(rhs.block_col_indices),
        values(rhs.values) {}

  /**
   * @brief Build a BCSR view of an existing CSR matrix on the host.
   *
   * Sweeps each block-row, materializes the set of non-empty block columns,
   * scatters CSR nonzeros into the dense per-block payload, and emits the
   * compressed offsets/indices/values vectors. O((rows / R) * nnz_in_row)
   * per block-row. Runs on the host and copies the result up to @c space .
   */
  template <auto rhs_space, typename csr_offset_t>
  bcsr_t(const csr_t<index_t, csr_offset_t, value_t, rhs_space>& csr)
      : rows(csr.rows), cols(csr.cols), nnzs(csr.nnzs) {
    num_block_rows = (rows + R - 1) / R;
    num_block_cols = (cols + C - 1) / C;

    // Pull CSR onto host so we can index it linearly.
    csr_t<index_t, csr_offset_t, value_t, memory_space_t::host> h(csr);

    std::vector<offset_t> h_block_offsets(num_block_rows + 1, offset_t{0});
    std::vector<index_t> h_block_col_indices;
    std::vector<value_t> h_values;

    // Per-block-row scratch: which block columns have we already opened?
    std::vector<offset_t> bcol_to_local(num_block_cols, offset_t{-1});
    std::vector<index_t> opened_bcols;
    opened_bcols.reserve(num_block_cols);

    for (std::size_t br = 0; br < num_block_rows; ++br) {
      const std::size_t row_lo = br * R;
      const std::size_t row_hi = std::min(row_lo + R, rows);

      // Pass 1: discover all non-empty block columns in this block-row.
      opened_bcols.clear();
      for (std::size_t r = row_lo; r < row_hi; ++r) {
        const csr_offset_t a_lo = h.offsets[r];
        const csr_offset_t a_hi = h.offsets[r + 1];
        for (csr_offset_t a = a_lo; a < a_hi; ++a) {
          const index_t col = h.indices[a];
          const std::size_t bc = static_cast<std::size_t>(col) / C;
          if (bcol_to_local[bc] == offset_t{-1}) {
            bcol_to_local[bc] = static_cast<offset_t>(opened_bcols.size());
            opened_bcols.push_back(static_cast<index_t>(bc));
          }
        }
      }

      // Sort the discovered block columns so the resulting BCSR is
      // canonical (cheaper kernels, simpler comparisons in tests).
      std::sort(opened_bcols.begin(), opened_bcols.end());
      for (std::size_t k = 0; k < opened_bcols.size(); ++k) {
        bcol_to_local[opened_bcols[k]] = static_cast<offset_t>(k);
      }

      const std::size_t blocks_here = opened_bcols.size();
      const std::size_t base_block_id = h_values.size() / kBlockSize;

      h_values.resize(h_values.size() + blocks_here * kBlockSize, value_t{0});
      h_block_col_indices.insert(h_block_col_indices.end(),
                                 opened_bcols.begin(), opened_bcols.end());
      h_block_offsets[br + 1] =
          h_block_offsets[br] + static_cast<offset_t>(blocks_here);

      // Pass 2: scatter CSR entries into the dense block payload.
      for (std::size_t r = row_lo; r < row_hi; ++r) {
        const std::size_t i = r - row_lo;
        const csr_offset_t a_lo = h.offsets[r];
        const csr_offset_t a_hi = h.offsets[r + 1];
        for (csr_offset_t a = a_lo; a < a_hi; ++a) {
          const index_t col = h.indices[a];
          const std::size_t bc = static_cast<std::size_t>(col) / C;
          const std::size_t j = static_cast<std::size_t>(col) % C;
          const std::size_t local = bcol_to_local[bc];
          const std::size_t b = base_block_id + local;
          h_values[b * kBlockSize + i * C + j] = h.values[a];
        }
      }

      // Reset scratch in O(blocks_here), not O(num_block_cols).
      for (auto bc : opened_bcols) {
        bcol_to_local[bc] = offset_t{-1};
      }
    }

    num_blocks = h_block_col_indices.size();

    // Push to the configured memory space.
    block_offsets = vector_t<offset_t, memory_space_t::host>(
        h_block_offsets.begin(), h_block_offsets.end());
    block_col_indices = vector_t<index_t, memory_space_t::host>(
        h_block_col_indices.begin(), h_block_col_indices.end());
    values = vector_t<value_t, memory_space_t::host>(h_values.begin(),
                                                     h_values.end());
  }
};  // struct bcsr_t

}  // namespace loops
