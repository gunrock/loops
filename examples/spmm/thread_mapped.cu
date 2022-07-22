/**
 * @file thread_mapped.cu
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Sparse Matrix-Matrix Multiplication example.
 * @version 0.1
 * @date 2022-02-03
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "helpers.hxx"
#include <loops/algorithms/spmm/thread_mapped.cuh>

using namespace loops;

int main(int argc, char** argv) {
  using index_t = int;
  using offset_t = int;
  using type_t = float;

  // ... I/O parameters, mtx, etc.
  parameters_t parameters(argc, argv);

  matrix_market_t<index_t, offset_t, type_t> mtx;
  csr_t<index_t, offset_t, type_t> csr(mtx.load(parameters.filename));

  // Input and output matrices.
  std::size_t n = 10;
  matrix_t<type_t> B(csr.cols, n);
  matrix_t<type_t> C(csr.rows, n);

  // Generate random numbers between [0, 10].
  generate::random::uniform_distribution(B.m_data.begin(), B.m_data.end(), 1,
                                         10);

  // Run the benchmark.
  util::timer_t timer;
  timer.start();
  algorithms::spmm::thread_mapped(csr, B, C);
  timer.stop();

  std::cout << "Elapsed (ms):\t" << timer.milliseconds() << std::endl;
}