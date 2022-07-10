/**
 * @file original.cu
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Sparse Matrix-Vector Multiplication example.
 * @version 0.1
 * @date 2022-02-03
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "spmv.hxx"
using namespace loops;

template <typename index_t, typename offset_t, typename type_t>
__global__ void spmv(const std::size_t rows,
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
  generate::random::uniform_distribution(x.begin(), x.end(), 1, 10);

  // Create a schedule.
  constexpr std::size_t block_size = 128;

  /// Set-up kernel launch parameters and run the kernel.
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  std::size_t grid_size = (csr.rows + block_size - 1) / block_size;
  launch::non_cooperative(
      stream, spmv<index_t, offset_t, type_t>, grid_size, block_size, csr.rows,
      csr.cols, csr.nnzs, csr.offsets.data().get(), csr.indices.data().get(),
      csr.values.data().get(), x.data().get(), y.data().get());

  cudaStreamSynchronize(stream);

  /// Validation code, can be safely ignored.
  if (parameters.validate) {
    auto h_y = cpu::spmv(csr, x);

    std::size_t errors = util::equal(
        y.data().get(), h_y.data(), csr.rows,
        [](const type_t a, const type_t b) { return std::abs(a - b) > 1e-2; },
        parameters.verbose);

    std::cout << "Matrix:\t\t" << extract_filename(parameters.filename)
              << std::endl;
    std::cout << "Dimensions:\t" << csr.rows << " x " << csr.cols << " ("
              << csr.nnzs << ")" << std::endl;
    std::cout << "Errors:\t\t" << errors << std::endl;
  }
}