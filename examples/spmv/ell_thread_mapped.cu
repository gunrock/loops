/**
 * @file ell_thread_mapped.cu
 * @author Loops contributors
 * @brief SpMV on an ELL matrix using the thread-mapped schedule.
 *
 * The kernel/schedule code is unchanged from the CSR thread_mapped example;
 * only the layout view is swapped (CSR -> ELL). Validation still uses the
 * CSR reference, since the math is independent of storage.
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "helpers.hxx"
#include <loops/algorithms/spmv/ell_thread_mapped.cuh>

using namespace loops;

int main(int argc, char** argv) {
  using index_t = int;
  using offset_t = int;
  using type_t = float;

  parameters_t parameters(argc, argv);

  matrix_market_t<index_t, offset_t, type_t> mtx;
  csr_t<index_t, offset_t, type_t> csr(mtx.load(parameters.filename));
  ell_t<index_t, type_t> ell(csr);

  vector_t<type_t> x(csr.cols);
  vector_t<type_t> y(csr.rows);
  generate::random::uniform_distribution(x.begin(), x.end(), 1, 10, /*seed=*/42u);

  util::timer_t timer;
  timer.start();
  algorithms::spmv::ell_thread_mapped(ell, x, y);
  timer.stop();

  std::cout << "ell_thread_mapped," << mtx.dataset << "," << csr.rows << ","
            << csr.cols << "," << csr.nnzs << ",pitch=" << ell.pitch << ","
            << timer.milliseconds() << std::endl;

  if (parameters.validate)
    cpu::validate(parameters, csr, x, y);
}
