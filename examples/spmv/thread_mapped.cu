/**
 * @file thread_mapped.cu
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Sparse Matrix-Vector Multiplication example.
 * @version 0.1
 * @date 2022-02-03
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "helpers.hxx"
#include <loops/algorithms/spmv/thread_mapped.cuh>

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
  
  // Warm-up run.
  algorithms::spmv::thread_mapped(csr, x, y);

  // Run the benchmark.
  util::timer_t timer;
  timer.start();
  // Compute y = Ax
  int num_runs = 100;
  for (auto i = 0; i < num_runs; ++i) {
    algorithms::spmv::thread_mapped(csr, x, y);
  }
  timer.stop();
  double time = timer.milliseconds()/num_runs;

  std::cout << "thread_mapped," << mtx.dataset << "," << csr.rows << ","
            << csr.cols << "," << csr.nnzs << "," << std::fixed << std::setprecision(6) << time
            << std::endl;

  // Validation.
  if (parameters.validate)
    cpu::validate(parameters, csr, x, y);
}
