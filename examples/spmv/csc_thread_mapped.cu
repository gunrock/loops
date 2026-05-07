/**
 * @file csc_thread_mapped.cu
 * @author Loops contributors
 * @brief SpMV over a CSC matrix via the thread_mapped schedule.
 * @version 0.1
 * @date 2026-05-06
 *
 * Loads a matrix as CSR, transposes it into CSC (csr -> coo -> csc), then
 * runs column-stationary SpMV with atomic-add by row. Demonstrates the
 * symmetric layout: same offsets-array contract as CSR, just a different
 * kernel-side interpretation.
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "helpers.hxx"

#include <loops/container/csc.hxx>
#include <loops/algorithms/spmv/csc_thread_mapped.cuh>

using namespace loops;

int main(int argc, char** argv) {
  using index_t = int;
  using offset_t = int;
  using type_t = float;

  parameters_t parameters(argc, argv);
  matrix_market_t<index_t, offset_t, type_t> mtx;
  csr_t<index_t, offset_t, type_t> csr(mtx.load(parameters.filename));
  csc_t<index_t, offset_t, type_t> csc(csr);

  vector_t<type_t> x(csr.cols);
  vector_t<type_t> y(csr.rows);
  generate::random::uniform_distribution(x.begin(), x.end(), 1, 10);

  auto timer = algorithms::spmv::csc_thread_mapped(csc, x, y);

  std::cout << "csc_thread_mapped," << mtx.dataset << "," << csr.rows << ","
            << csr.cols << "," << csr.nnzs << "," << timer.milliseconds()
            << std::endl;

  if (parameters.validate)
    cpu::validate(parameters, csr, x, y);
}
