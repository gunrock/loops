/**
 * @file original.cuh
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Sparse Matrix-Vector Multiplication kernels.
 * @version 0.1
 * @date 2022-02-03
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <loops/schedule.hxx>
#include <loops/container/formats.hxx>
#include <loops/container/vector.hxx>
#include <loops/util/launch.hxx>
#include <loops/util/device.hxx>
#include <loops/memory.hxx>
#include <iostream>

namespace loops {
namespace algorithms {
namespace spmv {

template <typename index_t, typename offset_t, typename type_t>
__global__ void __original(const std::size_t rows,
                           const std::size_t cols,
                           const std::size_t nnz,
                           const offset_t* offsets,
                           const index_t* indices,
                           const type_t* values,
                           const type_t* x,
                           type_t* y) {
  for (auto row = blockIdx.x * blockDim.x + threadIdx.x;
       row < rows;                    // boundary condition
       row += gridDim.x * blockDim.x  // step
  ) {
    type_t sum = 0;
    for (offset_t nz = offsets[row]; nz < offsets[row + 1]; ++nz)
      sum += values[nz] * x[indices[nz]];

    // Output
    y[row] = sum;
  }
}

/**
 * @brief Sparse-Matrix Vector Multiplication API.
 *
 * @tparam index_t Type of column indices.
 * @tparam offset_t Type of row offsets.
 * @tparam type_t Type of values.
 * @param csr CSR matrix (GPU).
 * @param x Input vector x (GPU).
 * @param y Output vector y (GPU).
 * @param stream CUDA stream.
 */
template <typename index_t, typename offset_t, typename type_t>
void original(csr_t<index_t, offset_t, type_t>& csr,
              vector_t<type_t>& x,
              vector_t<type_t>& y,
              cudaStream_t stream = 0) {
  // Create a schedule.
  constexpr std::size_t block_size = 128;

  /// Set-up kernel launch parameters and run the kernel.
  std::size_t grid_size = (csr.rows + block_size - 1) / block_size;
  launch::non_cooperative(stream, __original<index_t, offset_t, type_t>,
                          grid_size, block_size, csr.rows, csr.cols, csr.nnzs,
                          csr.offsets.data().get(), csr.indices.data().get(),
                          csr.values.data().get(), x.data().get(),
                          y.data().get());

  cudaStreamSynchronize(stream);
}

}  // namespace spmv
}  // namespace algorithms
}  // namespace loops