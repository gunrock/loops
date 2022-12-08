/**
 * @file group_mapped.cu
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Sparse Matrix-Vector Multiplication example.
 * @version 0.1
 * @date 2022-02-03
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "helpers.hxx"
#include <loops/algorithms/spmv/group_mapped.cuh>

using namespace loops;

int main(int argc, char** argv) {
  using index_t = int;
  using offset_t = int;
  using type_t = float;

  // ... I/O parameters, mtx, etc.
  parameters_t parameters(argc, argv);
  matrix_market_t<index_t, offset_t, type_t> mtx;
  csr_t<index_t, offset_t, type_t> csr(mtx.load(parameters.filename));

  // Input and output vectors.
  vector_t<type_t> x(csr.cols);
  vector_t<type_t> y(csr.rows);

  // Generate random numbers between [0, 1].
  generate::random::uniform_distribution(x.begin(), x.end(), 1, 10);

  // Run the benchmark.
  util::timer_t timer;
  timer.start();
  algorithms::spmv::group_mapped(csr, x, y);
  timer.stop();

  std::cout << "group_mapped," << mtx.dataset << "," << csr.rows << ","
            << csr.cols << "," << csr.nnzs << "," << timer.milliseconds()
            << std::endl;

  // Validation.
  if (parameters.validate)
    cpu::validate(parameters, csr, x, y);
}