/**
 * @file spmv.cu
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Sparse Matrix-Vector Multiplication example.
 * @version 0.1
 * @date 2022-02-03
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "spmv.hxx"

// template <typename index_t, typename offset_t, typename type_t>
// __global__ void spmv(const std::size_t rows,
//                      const std::size_t cols,
//                      const std::size_t nnz,
//                      const offset_t* offsets,
//                      const index_t* indices,
//                      const type_t* values,
//                      const type_t* x,
//                      type_t* y) {
//   /// Equivalent to:
//   /// row = blockIdx.x * blockDim.x + threadIdx.x; (init)
//   /// row < rows; (boundary condition)
//   /// row += gridDim.x * blockDim.x. (step)
//   for (auto row : grid_stride_range(std::size_t(0), rows)) {
//     type_t sum = 0;

//     /// Equivalent to:
//     /// for (offset_t nz = offset; nz < end; ++nz)
//     for (auto nz : range(offsets[row], offsets[row + 1])) {
//       sum += values[nz] * x[indices[nz]];
//     }

//     // Output
//     y[row] = sum;
//   }
// }

using namespace loops;

template <typename setup_t,
          typename index_t,
          typename offset_t,
          typename type_t>
__global__ void spmv(setup_t config,
                     const std::size_t rows,
                     const std::size_t cols,
                     const std::size_t nnz,
                     const offset_t* offsets,
                     const index_t* indices,
                     const type_t* values,
                     const type_t* x,
                     type_t* y) {
  /// Equivalent to:
  /// row = blockIdx.x * blockDim.x + threadIdx.x; (init)
  /// row < rows; (boundary condition)
  /// row += gridDim.x * blockDim.x. (step)
  for (auto row : config.tiles()) {
    type_t sum = 0;

    /// Equivalent to:
    /// for (offset_t nz = offset; nz < end; ++nz)
    for (auto nz : config.atoms(row)) {
      sum += values[nz] * x[indices[nz]];
    }

    // Output
    y[row] = sum;
  }
}

int main(int argc, char** argv) {
  using index_t = int;
  using offset_t = int;
  using type_t = float;

  // ... I/O parameters, mtx, etc.
  parameters_t parameters(argc, argv);

  csr_t<index_t, offset_t, type_t> csr;
  matrix_market_t<index_t, offset_t, type_t> mtx;
  csr.from_coo(mtx.load(parameters.filename));

  // Input and output vectors.
  vector_t<type_t> x(csr.rows);
  vector_t<type_t> y(csr.rows);

  // Generate random numbers between [0, 1].
  generate::random::uniform_distribution(x.begin(), x.end());

  // Create a schedule.
  using setup_t = schedule::setup<schedule::algroithms_t::thread_mapped,
                                  index_t, offset_t, std::size_t, std::size_t>;

  setup_t config(csr.offsets.data().get(), csr.rows, csr.nnzs);

  // Compute the spmv.
  constexpr std::size_t block_size = 256;
  std::size_t grid_size = (csr.rows + block_size - 1) / block_size;
  spmv<<<grid_size, block_size>>>(
      config, csr.rows, csr.cols, csr.nnzs, csr.offsets.data().get(),
      csr.indices.data().get(), csr.values.data().get(), x.data().get(),
      y.data().get());
  cudaDeviceSynchronize();

  if (parameters.validate) {
    auto h_y = cpu::spmv(csr, x);
    vector_t<type_t, memory::memory_space_t::device> d_y(y);
    bool success = std::equal(h_y.begin(), h_y.end(), d_y.begin());

    std::cout << "Matrix:\t\t" << extract_filename(parameters.filename)
              << std::endl;
    std::cout << "Dimensions:\t" << csr.rows << " x " << csr.cols << " ("
              << csr.nnzs << ")" << std::endl;
    std::cout << "Validation:\t" << (success ? "passed" : "failed")
              << std::endl;

    if (parameters.verbose) {
      std::cout << "y:\t\t";
      thrust::copy(y.begin(), (y.size() < 10) ? y.end() : y.begin() + 10,
                   std::ostream_iterator<type_t>(std::cout, " "));
      std::cout << std::endl;
    }
  }
}